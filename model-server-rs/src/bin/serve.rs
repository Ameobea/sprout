use foundations::telemetry::settings::TelemetrySettings;
use foundations::telemetry::TelemetryConfig;
use model_server_rs::engine::{Engine, Precision};
use model_server_rs::kernels::DEFAULT_CFG;
use model_server_rs::recommend::ModelData;
use model_server_rs::server::build_router;
use model_server_rs::weights::Params;
use model_server_rs::CORPUS;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

fn envv(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.into())
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

fn main() {
    let model_path = envv("MODEL_PATH", "/opt/model/jax_model.msgpack");
    let corpus_path = envv("CORPUS_PATH", "/opt/model/corpus_ids.json");
    let metadata_path = envv("METADATA_PATH", "/opt/model/processed-metadata.csv");
    let port: u16 = envv("PORT", "8000").parse().unwrap();
    let metrics_port: u16 = envv("METRICS_PORT", "5709").parse().unwrap();
    let threads: usize = envv("INFER_THREADS", "8").parse().unwrap();
    let pin = envv("PIN_CORES", "1") == "1";
    let prec = match envv("PRECISION", "f32").as_str() {
        "bf16" => Precision::Bf16,
        _ => Precision::F32,
    };

    let corpus_ids: Vec<i64> = serde_json::from_str(&std::fs::read_to_string(&corpus_path).unwrap()).unwrap();
    assert_eq!(corpus_ids.len(), CORPUS, "corpus_ids.json size mismatch");
    let id_to_idx: HashMap<i64, u32> = corpus_ids.iter().enumerate().map(|(i, &id)| (id, i as u32)).collect();

    let popularity = load_popularity(&metadata_path, &corpus_ids);
    if popularity.is_none() {
        eprintln!("warning: no popularity distribution loaded from {metadata_path}; niche boost disabled");
    }

    let t0 = std::time::Instant::now();
    let params = Params::load(Path::new(&model_path));
    let pins: Vec<usize> = (0..threads).collect();
    let engine = Engine::new(&params, threads, DEFAULT_CFG, pin.then_some(&pins[..]), prec);
    drop(params);
    println!(
        "model loaded + packed in {:.2}s ({threads} threads, pin={pin}, {prec:?})",
        t0.elapsed().as_secs_f64()
    );

    let md = Arc::new(ModelData { engine, corpus_ids, id_to_idx, popularity });

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

        let app = build_router(md);
        let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await.unwrap();
        println!("listening on 0.0.0.0:{port}");
        axum::serve(listener, app).await.unwrap();
    });
}
