use crate::recommend::{self, ModelData, RecommendRequest};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub struct LruCache {
    map: HashMap<String, Arc<Value>>,
    order: Vec<String>,
    max_size: usize,
}

impl LruCache {
    pub fn new(max_size: usize) -> Self {
        LruCache { map: HashMap::new(), order: Vec::new(), max_size }
    }
    fn touch(&mut self, key: &str) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            let k = self.order.remove(pos);
            self.order.push(k);
        }
    }
    pub fn get(&mut self, key: &str) -> Option<Arc<Value>> {
        let v = self.map.get(key).cloned();
        if v.is_some() {
            self.touch(key);
        }
        v
    }
    pub fn put(&mut self, key: String, value: Arc<Value>) {
        if self.map.contains_key(&key) {
            self.touch(&key);
        } else {
            if self.map.len() >= self.max_size {
                let oldest = self.order.remove(0);
                self.map.remove(&oldest);
            }
            self.order.push(key.clone());
        }
        self.map.insert(key, value);
    }
    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }
    pub fn len(&self) -> usize {
        self.map.len()
    }
}

pub struct AppState {
    pub md: Arc<ModelData>,
    pub cache: Mutex<LruCache>,
    pub infer_lock: Arc<tokio::sync::Mutex<()>>,
}

fn err(status: StatusCode, detail: &str) -> Response {
    (status, Json(json!({ "detail": detail }))).into_response()
}

/// Canonical cache key: defaults filled by serde, keys sorted by serde_json's BTreeMap.
fn cache_key(req: &RecommendRequest) -> String {
    serde_json::to_value(req).unwrap().to_string()
}

async fn recommend_handler(State(state): State<Arc<AppState>>, Json(req): Json<RecommendRequest>) -> Response {
    let key = cache_key(&req);
    if let Some(cached) = state.cache.lock().unwrap().get(&key) {
        return Json((*cached).clone()).into_response();
    }
    if req.profile.is_empty() {
        return err(StatusCode::BAD_REQUEST, "profile must be a non-empty list");
    }

    let md = state.md.clone();
    let prep = match recommend::preprocess(&md, &req.profile) {
        Ok(p) => p,
        Err(e) => return err(StatusCode::BAD_REQUEST, &e),
    };

    let lock = state.infer_lock.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _guard = lock.blocking_lock();
        let (recommendations, profile_holdout) = recommend::run_inference(&md, &prep, &req);
        recommend::RecommendResponse {
            recommendations,
            profile_holdout,
            normalization_stats: recommend::norm_stats_out(&prep.stats),
        }
    })
    .await;

    match result {
        Ok(resp) => {
            let value = Arc::new(serde_json::to_value(&resp).unwrap());
            state.cache.lock().unwrap().put(key, value.clone());
            Json((*value).clone()).into_response()
        }
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, &format!("inference panicked: {e}")),
    }
}

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

async fn corpus(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({ "corpus_ids": state.md.corpus_ids, "corpus_size": state.md.corpus_ids.len() }))
}

async fn cache_stats(State(state): State<Arc<AppState>>) -> Json<Value> {
    let cache = state.cache.lock().unwrap();
    Json(json!({ "size": cache.len(), "max_size": cache.max_size }))
}

async fn cache_clear(State(state): State<Arc<AppState>>) -> Json<Value> {
    state.cache.lock().unwrap().clear();
    Json(json!({ "status": "ok", "message": "Cache cleared" }))
}

pub fn build_router(md: Arc<ModelData>) -> Router {
    let state = Arc::new(AppState {
        md,
        cache: Mutex::new(LruCache::new(500)),
        infer_lock: Arc::new(tokio::sync::Mutex::new(())),
    });
    Router::new()
        .route("/recommend", post(recommend_handler))
        .route("/health", get(health))
        .route("/corpus", get(corpus))
        .route("/cache/stats", get(cache_stats))
        .route("/cache/clear", post(cache_clear))
        .with_state(state)
}
