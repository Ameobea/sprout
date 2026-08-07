use model_server_rs::engine::{Engine, HoldoutDelta, Precision};
use model_server_rs::kernels::KernCfg;
use model_server_rs::{norm, recommend, refimpl, weights::Params, CORPUS};
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
    std::env::var("GOLDEN_PATH").unwrap_or("testdata/forward_golden_2026logq.json".into())
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
    let model_path = std::env::var("MODEL_PATH").unwrap_or("../data/aug2026/jax_model_fresh_logq.msgpack".into());
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
    let model_path = std::env::var("MODEL_PATH").unwrap_or("../data/aug2026/jax_model_fresh_logq.msgpack".into());
    if !Path::new(&model_path).exists() {
        return;
    }
    let params = Params::load(Path::new(&model_path));
    let cases: Vec<ForwardCase> =
        serde_json::from_str(&std::fs::read_to_string(golden_path()).unwrap()).unwrap();
    let engine = Engine::new(&params, 4, KernCfg { nr: 32, mr: 8 }, None, Precision::Bf16);
    for c in &cases {
        let items: Vec<(u32, f32)> = c.idxs.iter().copied().zip(c.vals.iter().copied()).collect();
        let out = engine.forward(&items, None, None);
        let dl = max_abs_diff(out.logits_row(0), &c.logits);
        let dr = max_abs_diff(out.ratings_row(0), &c.ratings);
        println!("bf16 n={} max_diff logits={dl:.3} ratings={dr:.4}", items.len());
        assert!(dl < 0.5, "bf16 logits diverged: {dl}");
        assert!(dr < 0.2, "bf16 ratings diverged: {dr}");
    }
}

#[test]
fn engine_matches_golden_and_ref_holdout() {
    let model_path = std::env::var("MODEL_PATH").unwrap_or("../data/aug2026/jax_model_fresh_logq.msgpack".into());
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
            let out = engine.forward(&items, None, None);
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
            .map(|&(idx, val)| HoldoutDelta { idx, presence_removed: true, dval: val, dabs: 0.0 })
            .collect();
        let out = engine.forward(&items, None, Some(&deltas));
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

#[derive(Deserialize)]
struct GraftCase {
    idxs: Vec<u32>,
    vals: Vec<f32>,
    logits_head: Vec<f32>,
    ratings_head: Vec<f32>,
    logits_sum: f64,
    ratings_sum: f64,
    ease_head: Vec<f32>,
}

#[test]
fn graft_engine_matches_numpy_f64() {
    let model_path =
        std::env::var("GRAFT_MODEL_PATH").unwrap_or("../data/aug2026/jax_model_hybrid_concat.msgpack".into());
    let b_path = std::env::var("EASE_B_PATH").unwrap_or("../data/aug2026/serve/ease_B6k_lam200.f32bin".into());
    let golden = std::env::var("GRAFT_GOLDEN_PATH")
        .unwrap_or("../data/aug2026/serve/forward_golden_graft_concat.json".into());
    if !Path::new(&model_path).exists() || !Path::new(&b_path).exists() {
        eprintln!("skipping: graft model or B not present");
        return;
    }
    let params = Params::load(Path::new(&model_path));
    assert!(params.ease_proj.is_some(), "expected graft checkpoint");
    let bytes = std::fs::read(&b_path).unwrap();
    let ease_b: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let c: GraftCase = serde_json::from_str(&std::fs::read_to_string(&golden).unwrap()).unwrap();

    let engine =
        Engine::new_with_ease(&params, Some(ease_b), 4, KernCfg { nr: 32, mr: 8 }, None, Precision::F32);
    let items: Vec<(u32, f32)> = c.idxs.iter().copied().zip(c.vals.iter().copied()).collect();
    let out = engine.forward(&items, None, None);
    let logits = out.logits_row(out.rows - 1);
    let ratings = out.ratings_row(out.rows - 1);

    let e = out.ease_full_raw.as_ref().unwrap();
    let de = max_abs_diff(&e[..c.ease_head.len()], &c.ease_head);
    let dl = max_abs_diff(&logits[..c.logits_head.len()], &c.logits_head);
    let dr = max_abs_diff(&ratings[..c.ratings_head.len()], &c.ratings_head);
    let ls: f64 = logits.iter().map(|&v| v as f64).sum();
    let rs: f64 = ratings.iter().map(|&v| v as f64).sum();
    println!("graft max_diff ease={de:.2e} logits={dl:.2e} ratings={dr:.2e} sums Δ={:.3e}/{:.3e}",
        (ls - c.logits_sum).abs(), (rs - c.ratings_sum).abs());
    assert!(de < 1e-3, "ease channel diverged: {de}");
    assert!(dl < 0.02, "graft logits diverged: {dl}");
    assert!(dr < 0.01, "graft ratings diverged: {dr}");
    assert!((ls - c.logits_sum).abs() / c.logits_sum.abs().max(1.0) < 1e-3);
}

#[test]
fn rating_stack_matches_numpy_f64() {
    let golden_p = "../data/aug2026/serve/rating_stack_golden.json";
    let b_p = "../data/aug2026/serve/rating_resid_B6k.f32bin";
    let im_p = "../data/aug2026/serve/rating_imean6k.f32bin";
    if !Path::new(golden_p).exists() || !Path::new(b_p).exists() {
        eprintln!("skipping: rating stack artifacts missing");
        return;
    }
    #[derive(Deserialize)]
    struct StackCase {
        idxs: Vec<u32>,
        raw: Vec<f32>,
        updated_at: Vec<i64>,
        base_ratings: Vec<f32>,
        sigma: f32,
        w: f32,
        era_slope: f32,
        era_corr_now: f32,
        scores_head: Vec<f32>,
        final_head: Vec<f32>,
        final_sum: f64,
    }
    let c: StackCase = serde_json::from_str(&std::fs::read_to_string(golden_p).unwrap()).unwrap();
    let load = |p: &str, n: usize| -> Vec<f32> {
        let bytes = std::fs::read(p).unwrap();
        assert_eq!(bytes.len(), n * 4);
        bytes.chunks_exact(4).map(|ch| f32::from_le_bytes(ch.try_into().unwrap())).collect()
    };
    let b = load(b_p, CORPUS * CORPUS);
    let imean = load(im_p, CORPUS);

    let (normalized, stats) = norm::normalize_ratings(&c.raw);
    assert!((stats.sigma - c.sigma).abs() < 1e-4);
    let enc_items: Vec<(u32, f32)> = c.idxs.iter().copied().zip(normalized.iter().copied()).collect();
    let enc_rated: Vec<bool> = c.raw.iter().map(|&r| r > 0.0).collect();
    let mut rated_mask = vec![false; CORPUS];
    for &(i, _) in &enc_items {
        rated_mask[i as usize] = true;
    }
    let prep = recommend::Prepped {
        anime_ids: c.idxs.iter().map(|&i| i as i64).collect(),
        corpus_indices: c.idxs.clone(),
        original: c.raw.clone(),
        normalized,
        stats,
        enc_items,
        enc_rated,
        enc_abs: vec![0.0; c.idxs.len()],
        updated_at: c.updated_at.clone(),
        deltas: Vec::new(),
        rated_mask,
    };
    let st = recommend::compute_rating_stack(&b, &imean, &prep, 1.0, true, &c.base_ratings).unwrap();
    assert!((st.w - c.w).abs() < 1e-5, "w {} vs {}", st.w, c.w);
    assert!(max_abs_diff(&st.scores[..16], &c.scores_head) < 2e-3, "resid scores mismatch");
    assert!((st.era_slope - c.era_slope).abs() < 1e-3, "slope {} vs {}", st.era_slope, c.era_slope);
    assert!(
        (st.out.era_correction_now - c.era_corr_now).abs() < 1e-3,
        "corr {} vs {}",
        st.out.era_correction_now,
        c.era_corr_now
    );
    assert!(max_abs_diff(&st.ratings[..16], &c.final_head) < 2e-3, "final ratings mismatch");
    let sum: f64 = st.ratings.iter().map(|&v| v as f64).sum();
    assert!((sum - c.final_sum).abs() < 0.05, "final sum {} vs {}", sum, c.final_sum);
}

#[derive(Deserialize)]
struct RcCase {
    idxs: Vec<u32>,
    vals: Vec<f32>,
    #[serde(rename = "abs")]
    abs_vals: Vec<f32>,
    logits_head: Vec<f32>,
    ratings_head: Vec<f32>,
    logits_sum: f64,
    ratings_sum: f64,
}

#[test]
fn rc_3ch_graft_engine_matches_numpy_f64() {
    let model_path = std::env::var("RC_MODEL_PATH").unwrap_or("../data/aug2026/rc_full_seed0.msgpack".into());
    let b_path = std::env::var("EASE_B_PATH").unwrap_or("../data/aug2026/serve/ease_B6k_lam200.f32bin".into());
    let golden = std::env::var("RC_GOLDEN_PATH").unwrap_or("../data/aug2026/serve/forward_golden_rc.json".into());
    if !Path::new(&model_path).exists() || !Path::new(&golden).exists() {
        eprintln!("skipping: RC checkpoint or golden not present");
        return;
    }
    let params = Params::load(Path::new(&model_path));
    assert_eq!(params.in_channels(), 3, "expected 3-channel RC checkpoint");
    assert!(params.ease_proj.is_some(), "expected graft checkpoint");
    let bytes = std::fs::read(&b_path).unwrap();
    let ease_b: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let cases: Vec<RcCase> = serde_json::from_str(&std::fs::read_to_string(&golden).unwrap()).unwrap();

    let engine =
        Engine::new_with_ease(&params, Some(ease_b), 4, KernCfg { nr: 32, mr: 8 }, None, Precision::F32);
    for c in &cases {
        let items: Vec<(u32, f32)> = c.idxs.iter().copied().zip(c.vals.iter().copied()).collect();
        let out = engine.forward(&items, Some(&c.abs_vals), None);
        let logits = out.logits_row(out.rows - 1);
        let ratings = out.ratings_row(out.rows - 1);
        let dl = max_abs_diff(&logits[..c.logits_head.len()], &c.logits_head);
        let dr = max_abs_diff(&ratings[..c.ratings_head.len()], &c.ratings_head);
        let sl: f64 = logits[..CORPUS].iter().map(|&v| v as f64).sum();
        let sr: f64 = ratings[..CORPUS].iter().map(|&v| v as f64).sum();
        println!("rc n={} dl={dl:.2e} dr={dr:.2e}", items.len());
        assert!(dl < 0.02 && dr < 0.01, "RC head diverged: {dl} {dr}");
        assert!((sl - c.logits_sum).abs() < 1.5, "logits sum {sl} vs {}", c.logits_sum);
        assert!((sr - c.ratings_sum).abs() < 0.5, "ratings sum {sr} vs {}", c.ratings_sum);
    }

    // holdout rows: removing entry i must equal a fresh forward on the reduced profile
    let c = &cases[3];
    let items: Vec<(u32, f32)> = c.idxs.iter().copied().zip(c.vals.iter().copied()).collect();
    let deltas: Vec<HoldoutDelta> = items
        .iter()
        .zip(&c.abs_vals)
        .map(|(&(idx, val), &a)| HoldoutDelta { idx, presence_removed: true, dval: val, dabs: a })
        .collect();
    let out = engine.forward(&items, Some(&c.abs_vals), Some(&deltas));
    for i in 0..items.len().min(8) {
        let reduced: Vec<(u32, f32)> =
            items.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, &it)| it).collect();
        let red_abs: Vec<f32> = c.abs_vals.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, &a)| a).collect();
        let fresh = engine.forward(&reduced, Some(&red_abs), None);
        let dl = max_abs_diff(out.logits_row(i), fresh.logits_row(0));
        let dr = max_abs_diff(out.ratings_row(i), fresh.ratings_row(0));
        assert!(dl < 0.02 && dr < 0.01, "RC holdout row {i} diverged: {dl} {dr}");
    }
}
