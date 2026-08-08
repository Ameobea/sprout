use crate::metrics::{cache as cache_metrics, http_server, recommend as recommend_metrics};
use crate::recommend::{self, ModelData, RecommendRequest};
use axum::body::Bytes;
use axum::extract::{Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub struct LruCache {
    map: HashMap<String, Bytes>,
    order: Vec<String>,
    max_size: usize,
    bytes: usize,
}

impl LruCache {
    pub fn new(max_size: usize) -> Self {
        LruCache { map: HashMap::new(), order: Vec::new(), max_size, bytes: 0 }
    }
    fn touch(&mut self, key: &str) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            let k = self.order.remove(pos);
            self.order.push(k);
        }
    }
    fn sync_gauges(&self) {
        cache_metrics::entries().set(self.map.len() as u64);
        cache_metrics::size_bytes().set(self.bytes as u64);
    }
    pub fn get(&mut self, key: &str) -> Option<Bytes> {
        let v = self.map.get(key).cloned();
        if v.is_some() {
            self.touch(key);
        }
        v
    }
    pub fn put(&mut self, key: String, value: Bytes) {
        if let Some(old) = self.map.get(&key) {
            self.bytes -= old.len();
            self.touch(&key);
        } else {
            if self.map.len() >= self.max_size {
                let oldest = self.order.remove(0);
                let old = self.map.remove(&oldest).unwrap();
                self.bytes -= oldest.len() + old.len();
                cache_metrics::evictions_total().inc();
            }
            self.bytes += key.len();
            self.order.push(key.clone());
        }
        self.bytes += value.len();
        self.map.insert(key, value);
        self.sync_gauges();
    }
    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
        self.bytes = 0;
        self.sync_gauges();
    }
    pub fn len(&self) -> usize {
        self.map.len()
    }
}

pub struct AppState {
    pub models: HashMap<&'static str, Arc<ModelData>>,
    pub default_model: &'static str,
    pub cache: Mutex<LruCache>,
    pub infer_lock: Arc<tokio::sync::Mutex<()>>,
}

impl AppState {
    fn resolve(&self, requested: Option<&str>) -> Option<&Arc<ModelData>> {
        self.models.get(requested.unwrap_or(self.default_model))
    }
}

fn err(status: StatusCode, detail: &str) -> Response {
    (status, Json(json!({ "detail": detail }))).into_response()
}

fn json_bytes(body: Bytes) -> Response {
    ([(header::CONTENT_TYPE, "application/json")], body).into_response()
}

/// Canonical cache key: defaults filled by serde, keys sorted by serde_json's BTreeMap.
fn cache_key(req: &RecommendRequest) -> String {
    serde_json::to_value(req).unwrap().to_string()
}

async fn track_request(req: Request, next: Next) -> Response {
    let endpoint = match req.uri().path() {
        "/recommend" => "recommend",
        "/health" => "health",
        "/models" => "models",
        "/corpus" => "corpus",
        "/cache/stats" => "cache_stats",
        "/cache/clear" => "cache_clear",
        _ => "other",
    };
    http_server::requests_total(endpoint).inc();
    let timer = http_server::response_time_seconds(endpoint).start_timer();
    let resp = next.run(req).await;
    drop(timer);
    let status = resp.status();
    if status.is_client_error() || status.is_server_error() {
        http_server::requests_failed_total(endpoint, status.as_u16()).inc();
    }
    resp
}

fn record_request_shape(model: &'static str, req: &RecommendRequest) {
    recommend_metrics::requests_by_model_total(model).inc();
    recommend_metrics::profile_size().observe(req.profile.len() as f64);
    if req.include_profile_holdout {
        recommend_metrics::feature_requests_total("profile_holdout").inc();
    }
    if req.include_contribution_analysis {
        recommend_metrics::feature_requests_total("contribution_analysis").inc();
    }
    if req.use_alt_ranking {
        recommend_metrics::feature_requests_total("alt_ranking").inc();
    }
    if req.niche_boost_factor > 0.0 {
        recommend_metrics::feature_requests_total("niche_boost").inc();
    }
    if req.include_raw_logits {
        recommend_metrics::feature_requests_total("raw_logits").inc();
    }
}

async fn recommend_handler(State(state): State<Arc<AppState>>, Json(req): Json<RecommendRequest>) -> Response {
    let md = match state.resolve(req.model.as_deref()) {
        Some(m) => m.clone(),
        None => {
            recommend_metrics::unknown_model_total().inc();
            return err(StatusCode::NOT_FOUND, &format!("unknown model: {}", req.model.as_deref().unwrap_or("")));
        }
    };
    record_request_shape(md.name, &req);
    let key = format!("{}|{}", md.name, cache_key(&req));
    if let Some(cached) = state.cache.lock().unwrap().get(&key) {
        cache_metrics::hits_total().inc();
        return json_bytes(cached);
    }
    cache_metrics::misses_total().inc();
    if req.profile.is_empty() {
        recommend_metrics::invalid_profile_total().inc();
        return err(StatusCode::BAD_REQUEST, "profile must be a non-empty list");
    }

    let prep = match recommend::preprocess(&md, &req.profile) {
        Ok(p) => p,
        Err(e) => {
            recommend_metrics::invalid_profile_total().inc();
            return err(StatusCode::BAD_REQUEST, &e);
        }
    };

    let lock = state.infer_lock.clone();
    let result = tokio::task::spawn_blocking(move || {
        let t0 = Instant::now();
        let _guard = lock.blocking_lock();
        recommend_metrics::queue_wait_seconds().observe(t0.elapsed().as_nanos() as u64);
        let timer = recommend_metrics::inference_time_seconds(md.name).start_timer();
        let (recommendations, profile_holdout, rating_stack, contribution_baseline) =
            recommend::run_inference(&md, &prep, &req);
        timer.stop_and_record();
        recommend::RecommendResponse {
            recommendations,
            profile_holdout,
            normalization_stats: recommend::norm_stats_out(&prep.stats),
            rating_stack,
            contribution_baseline,
        }
    })
    .await;

    match result {
        Ok(resp) => {
            let body = Bytes::from(serde_json::to_vec(&resp).unwrap());
            recommend_metrics::response_size_bytes().observe(body.len() as f64);
            state.cache.lock().unwrap().put(key, body.clone());
            json_bytes(body)
        }
        Err(e) => {
            recommend_metrics::inference_panics_total().inc();
            err(StatusCode::INTERNAL_SERVER_ERROR, &format!("inference panicked: {e}"))
        }
    }
}

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

async fn models_list(State(state): State<Arc<AppState>>) -> Json<Value> {
    let list: Vec<Value> = state
        .models
        .values()
        .map(|m| {
            json!({
                "name": m.name,
                "serving": format!("{:?}", m.serving).to_lowercase(),
                "corpus_size": m.corpus_ids.len(),
                "default": m.name == state.default_model,
            })
        })
        .collect();
    Json(json!({ "models": list }))
}

async fn corpus(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(q): axum::extract::Query<HashMap<String, String>>,
) -> Response {
    match state.resolve(q.get("model").map(String::as_str)) {
        Some(md) => Json(json!({
            "model": md.name,
            "corpus_ids": md.corpus_ids,
            "corpus_size": md.corpus_ids.len(),
        }))
        .into_response(),
        None => err(StatusCode::NOT_FOUND, "unknown model"),
    }
}

async fn cache_stats(State(state): State<Arc<AppState>>) -> Json<Value> {
    let cache = state.cache.lock().unwrap();
    Json(json!({ "size": cache.len(), "max_size": cache.max_size }))
}

async fn cache_clear(State(state): State<Arc<AppState>>) -> Json<Value> {
    state.cache.lock().unwrap().clear();
    Json(json!({ "status": "ok", "message": "Cache cleared" }))
}

pub fn build_router(models: HashMap<&'static str, Arc<ModelData>>, default_model: &'static str) -> Router {
    let state = Arc::new(AppState {
        models,
        default_model,
        cache: Mutex::new(LruCache::new(500)),
        infer_lock: Arc::new(tokio::sync::Mutex::new(())),
    });
    Router::new()
        .route("/recommend", post(recommend_handler))
        .route("/health", get(health))
        .route("/models", get(models_list))
        .route("/corpus", get(corpus))
        .route("/cache/stats", get(cache_stats))
        .route("/cache/clear", post(cache_clear))
        .layer(middleware::from_fn(track_request))
        .with_state(state)
}
