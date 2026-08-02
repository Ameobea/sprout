use model_server_rs::engine::{Engine, HoldoutDelta, Precision};
use model_server_rs::kernels::KernCfg;
use model_server_rs::{norm, refimpl, weights::Params, CORPUS};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct ForwardCase {
    idxs: Vec<u32>,
    vals: Vec<f32>,
    logits: Vec<f32>,
    ratings: Vec<f32>,
}

#[derive(Deserialize)]
struct NormCase {
    scores: Vec<f32>,
    normed: Vec<f32>,
    mu: f32,
    sigma: f32,
    alpha: f32,
    zscore: Vec<f32>,
    absolute: Vec<f32>,
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn golden_path() -> String {
    std::env::var("GOLDEN_PATH").unwrap_or("testdata/forward_golden.json".into())
}

#[test]
fn norm_matches_python() {
    let cases: Vec<NormCase> =
        serde_json::from_str(&std::fs::read_to_string("testdata/norm_golden.json").unwrap()).unwrap();
    for c in cases {
        let (normed, stats) = norm::normalize_ratings(&c.scores);
        assert!(max_abs_diff(&normed, &c.normed) < 1e-5, "normed mismatch for {:?}", c.scores);
        assert!((stats.mu - c.mu).abs() < 1e-4, "mu {} vs {}", stats.mu, c.mu);
        assert!((stats.sigma - c.sigma).abs() < 1e-4);
        assert!((stats.alpha - c.alpha).abs() < 1e-5);
        assert!(max_abs_diff(&stats.zscore, &c.zscore) < 1e-5);
        assert!(max_abs_diff(&stats.absolute, &c.absolute) < 1e-5);
    }
}

#[test]
fn forward_ref_matches_numpy_f64() {
    let model_path = std::env::var("MODEL_PATH").unwrap_or("../data/jax_model.msgpack".into());
    if !Path::new(&model_path).exists() {
        eprintln!("skipping: no model at {model_path}");
        return;
    }
    let params = Params::load(Path::new(&model_path));
    let cases: Vec<ForwardCase> =
        serde_json::from_str(&std::fs::read_to_string(golden_path()).unwrap()).unwrap();
    for c in &cases {
        let x = refimpl::make_dense_profile(&c.idxs, &c.vals);
        let (logits, ratings) = refimpl::forward_ref(&params, &x);
        let dl = max_abs_diff(&logits, &c.logits);
        let dr = max_abs_diff(&ratings, &c.ratings);
        println!("n={} max_diff logits={dl:.2e} ratings={dr:.2e}", c.idxs.len());
        assert!(dl < 0.02, "logits diverged: {dl}");
        assert!(dr < 0.01, "ratings diverged: {dr}");
    }
}

#[test]
fn bf16_engine_close_to_golden() {
    let model_path = std::env::var("MODEL_PATH").unwrap_or("../data/jax_model.msgpack".into());
    if !Path::new(&model_path).exists() {
        return;
    }
    let params = Params::load(Path::new(&model_path));
    let cases: Vec<ForwardCase> =
        serde_json::from_str(&std::fs::read_to_string(golden_path()).unwrap()).unwrap();
    let engine = Engine::new(&params, 4, KernCfg { nr: 32, mr: 8 }, None, Precision::Bf16);
    for c in &cases {
        let items: Vec<(u32, f32)> = c.idxs.iter().copied().zip(c.vals.iter().copied()).collect();
        let out = engine.forward(&items, None);
        let dl = max_abs_diff(out.logits_row(0), &c.logits);
        let dr = max_abs_diff(out.ratings_row(0), &c.ratings);
        println!("bf16 n={} max_diff logits={dl:.3} ratings={dr:.4}", items.len());
        assert!(dl < 0.5, "bf16 logits diverged: {dl}");
        assert!(dr < 0.2, "bf16 ratings diverged: {dr}");
    }
}

#[test]
fn engine_matches_golden_and_ref_holdout() {
    let model_path = std::env::var("MODEL_PATH").unwrap_or("../data/jax_model.msgpack".into());
    if !Path::new(&model_path).exists() {
        eprintln!("skipping: no model at {model_path}");
        return;
    }
    let params = Params::load(Path::new(&model_path));
    let cases: Vec<ForwardCase> =
        serde_json::from_str(&std::fs::read_to_string(golden_path()).unwrap()).unwrap();

    for cfg in [KernCfg { nr: 64, mr: 6 }, KernCfg { nr: 32, mr: 8 }] {
        let engine = Engine::new(&params, 4, cfg, None, Precision::F32);
        for c in &cases {
            let items: Vec<(u32, f32)> = c.idxs.iter().copied().zip(c.vals.iter().copied()).collect();
            let out = engine.forward(&items, None);
            let dl = max_abs_diff(out.logits_row(0), &c.logits);
            let dr = max_abs_diff(out.ratings_row(0), &c.ratings);
            println!("cfg={cfg:?} n={} max_diff logits={dl:.2e} ratings={dr:.2e}", items.len());
            assert!(dl < 0.02, "engine logits diverged: {dl}");
            assert!(dr < 0.01, "engine ratings diverged: {dr}");
        }

        // holdout rows must match a from-scratch forward on the reduced profile
        let c = &cases[2];
        let items: Vec<(u32, f32)> = c.idxs.iter().copied().zip(c.vals.iter().copied()).collect();
        let deltas: Vec<HoldoutDelta> = items
            .iter()
            .map(|&(idx, val)| HoldoutDelta { idx, presence_removed: true, dval: val })
            .collect();
        let out = engine.forward(&items, Some(&deltas));
        assert_eq!(out.rows, items.len() + 1);
        for (i, _) in items.iter().enumerate() {
            let reduced: Vec<(u32, f32)> =
                items.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, &it)| it).collect();
            let mut x = vec![0.0f32; CORPUS * 2];
            for &(idx, v) in &reduced {
                x[idx as usize] = 1.0;
                x[CORPUS + idx as usize] = v;
            }
            let (ref_logits, ref_ratings) = refimpl::forward_ref(&params, &x);
            let dl = max_abs_diff(out.logits_row(i), &ref_logits);
            let dr = max_abs_diff(out.ratings_row(i), &ref_ratings);
            assert!(dl < 0.02 && dr < 0.01, "holdout row {i} diverged: {dl} {dr}");
        }
    }
}
