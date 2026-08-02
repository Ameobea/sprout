use foundations::telemetry::settings::TelemetrySettings;
use foundations::telemetry::TelemetryConfig;
use model_server_rs::engine::{Engine, Precision};
use model_server_rs::kernels::DEFAULT_CFG;
use model_server_rs::metrics::recommend as recommend_metrics;
use model_server_rs::recommend::{ModelData, ServingFamily};
use model_server_rs::server::build_router;
use model_server_rs::weights::Params;
use model_server_rs::CORPUS;
use serde::Deserialize;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

fn envv(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.into())
}

#[derive(Deserialize)]
struct ModelCfg {
    name: String,
    weights: String,
    corpus: String,
    metadata: String,
    #[serde(default)]
    train_counts: Option<String>,
    #[serde(default)]
    serving: Option<String>,
    #[serde(default)]
    default: bool,
}

fn load_popularity(path: &str, corpus_ids: &[i64]) -> Option<Vec<f32>> {
    let mut rdr = csv::Reader::from_path(path).ok()?;
    let headers = rdr.headers().ok()?.clone();
    let id_col = headers.iter().position(|h| h == "id")?;
    let count_col = headers.iter().position(|h| h == "rating_count")?;
    let mut counts: HashMap<i64, f32> = HashMap::new();
    for rec in rdr.records().flatten() {
        if let (Some(Ok(id)), Some(Ok(count))) =
            (rec.get(id_col).map(str::parse::<i64>), rec.get(count_col).map(str::parse::<f32>))
        {
            counts.insert(id, count);
        }
    }
    let mut pop: Vec<f32> = corpus_ids.iter().map(|id| counts.get(id).copied().unwrap_or(0.0)).collect();
    let total: f32 = pop.iter().sum();
    if total <= 0.0 {
        return None;
    }
    pop.iter_mut().for_each(|p| *p /= total);
    Some(pop)
}

fn load_model(cfg: &ModelCfg, threads: usize, pin: bool, prec: Precision) -> Arc<ModelData> {
    let corpus_ids: Vec<i64> = serde_json::from_str(&std::fs::read_to_string(&cfg.corpus).unwrap()).unwrap();
    assert_eq!(corpus_ids.len(), CORPUS, "{}: corpus size mismatch", cfg.name);
    let id_to_idx: HashMap<i64, u32> = corpus_ids.iter().enumerate().map(|(i, &id)| (id, i as u32)).collect();

    let serving = match cfg.serving.as_deref() {
        Some("logq") => ServingFamily::Logq,
        Some("legacy") | None => ServingFamily::Legacy,
        Some(other) => panic!("{}: unknown serving family {other}", cfg.name),
    };

    let train_counts: Option<Vec<f32>> = cfg.train_counts.as_ref().map(|p| {
        let counts: Vec<f64> = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        assert_eq!(counts.len(), CORPUS, "{}: train_counts size mismatch", cfg.name);
        counts.into_iter().map(|c| c as f32).collect()
    });
    if serving == ServingFamily::Logq {
        assert!(train_counts.is_some(), "{}: logq serving requires train_counts", cfg.name);
    }
    let log_pop = train_counts.as_ref().map(|cs| cs.iter().map(|&c| c.max(1.0).ln()).collect());

    let popularity = load_popularity(&cfg.metadata, &corpus_ids);
    if serving == ServingFamily::Legacy && popularity.is_none() {
        eprintln!("warning: {}: no popularity from {}; legacy niche boost disabled", cfg.name, cfg.metadata);
    }

    let t0 = std::time::Instant::now();
    let params = Params::load(Path::new(&cfg.weights));
    let pins: Vec<usize> = (0..threads).collect();
    let engine = Engine::new(&params, threads, DEFAULT_CFG, pin.then_some(&pins[..]), prec);
    drop(params);
    println!("[{}] loaded + packed in {:.2}s ({serving:?})", cfg.name, t0.elapsed().as_secs_f64());

    Arc::new(ModelData {
        name: Box::leak(cfg.name.clone().into_boxed_str()),
        serving,
        engine,
        corpus_ids,
        id_to_idx,
        popularity,
        train_counts,
        log_pop,
    })
}

fn main() {
    let port: u16 = envv("PORT", "8000").parse().unwrap();
    let metrics_port: u16 = envv("METRICS_PORT", "5709").parse().unwrap();
    let threads: usize = envv("INFER_THREADS", "8").parse().unwrap();
    let pin = envv("PIN_CORES", "1") == "1";
    let prec = match envv("PRECISION", "f32").as_str() {
        "bf16" => Precision::Bf16,
        _ => Precision::F32,
    };

    let cfgs: Vec<ModelCfg> = match std::env::var("MODELS_CONFIG") {
        Ok(path) => serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap(),
        Err(_) => vec![ModelCfg {
            name: "default".into(),
            weights: envv("MODEL_PATH", "/opt/model/jax_model.msgpack"),
            corpus: envv("CORPUS_PATH", "/opt/model/corpus_ids.json"),
            metadata: envv("METADATA_PATH", "/opt/model/processed-metadata.csv"),
            train_counts: None,
            serving: None,
            default: true,
        }],
    };
    assert!(!cfgs.is_empty(), "MODELS_CONFIG must list at least one model");

    let default_name: String =
        cfgs.iter().find(|c| c.default).map(|c| c.name.clone()).unwrap_or_else(|| cfgs[0].name.clone());
    let models: HashMap<&'static str, Arc<ModelData>> =
        cfgs.iter().map(|c| load_model(c, threads, pin, prec)).map(|m| (m.name, m)).collect();
    let default_model = models
        .get(default_name.as_str())
        .unwrap_or_else(|| panic!("default model {default_name} not in config"))
        .name;
    let n_models = models.len();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let mut service_info = foundations::service_info!();
        service_info.name_in_metrics = "anime_model_server".into();
        let mut tele_settings = TelemetrySettings::default();
        tele_settings.server.addr = SocketAddr::from(([0, 0, 0, 0], metrics_port)).into();
        let tele_driver = foundations::telemetry::init(TelemetryConfig {
            service_info: &service_info,
            settings: &tele_settings,
            custom_server_routes: vec![],
        })
        .unwrap();
        if let Some(addr) = tele_driver.server_addr() {
            println!("telemetry server listening on http://{addr}");
        }
        tokio::task::spawn(tele_driver);
        // must run after telemetry::init or the whole registry binds to the "undefined" service name
        recommend_metrics::models_loaded().set(n_models as u64);

        let app = build_router(models, default_model);
        let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await.unwrap();
        println!("listening on 0.0.0.0:{port}");
        axum::serve(listener, app).await.unwrap();
    });
}
