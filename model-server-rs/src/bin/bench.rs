use model_server_rs::engine::{Engine, HoldoutDelta, Precision};
use model_server_rs::kernels::KernCfg;
use model_server_rs::weights::Params;
use model_server_rs::{CORPUS, HIDDEN};
use std::path::Path;
use std::time::Instant;

fn parse_arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::args()
        .skip_while(|a| a != name)
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn has_flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

fn make_profile(n: usize, seed: u64) -> Vec<(u32, f32)> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    let mut rng = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut idxs: Vec<u32> = Vec::with_capacity(n);
    while idxs.len() < n {
        let i = (rng() % CORPUS as u64) as u32;
        if !idxs.contains(&i) {
            idxs.push(i);
        }
    }
    idxs.sort_unstable();
    idxs.iter().map(|&i| (i, ((rng() % 4000) as f32 / 1000.0) - 2.0)).collect()
}

fn holdout_deltas(items: &[(u32, f32)]) -> Vec<HoldoutDelta> {
    items.iter().map(|&(idx, val)| HoldoutDelta { idx, presence_removed: true, dval: val }).collect()
}

fn bench_one(engine: &Engine, n: usize, iters: usize, holdout: bool) -> (f64, f64) {
    let items = make_profile(n, n as u64);
    let deltas = holdout_deltas(&items);
    let hd = if holdout { Some(&deltas[..]) } else { None };
    let mut times = Vec::with_capacity(iters);
    let mut sink = 0.0f32;
    for _ in 0..iters {
        let t0 = Instant::now();
        let out = engine.forward(&items, hd);
        times.push(t0.elapsed().as_secs_f64() * 1e3);
        sink += out.logits_row(out.rows - 1)[0];
    }
    std::hint::black_box(sink);
    times.sort_by(f64::total_cmp);
    (times[times.len() / 2], times[times.len() * 95 / 100])
}

const ROW_MACS: f64 = (HIDDEN * 512 + 512 * 1024 * 2 + 1024 * 2048 * 2 + 2048 * 6000 * 2) as f64;

fn main() {
    let model_path = std::env::var("MODEL_PATH").unwrap_or("../data/jax_model.msgpack".into());
    let threads: usize = parse_arg("--threads", 8);
    let nr: usize = parse_arg("--nr", 64);
    let mr: usize = parse_arg("--mr", 6);
    let iters: usize = parse_arg("--iters", 30);
    let pin = has_flag("--pin");
    let prec = if has_flag("--bf16") { Precision::Bf16 } else { Precision::F32 };

    println!("loading model...");
    let t0 = Instant::now();
    let params = Params::load(Path::new(&model_path));
    println!("loaded in {:.2}s", t0.elapsed().as_secs_f64());

    let pins: Vec<usize> = (0..threads).collect();
    let t0 = Instant::now();
    let engine = Engine::new(&params, threads, KernCfg { nr, mr }, pin.then_some(&pins[..]), prec);
    println!("packed in {:.2}s", t0.elapsed().as_secs_f64());
    drop(params);

    println!("\ncfg: nr={nr} mr={mr} threads={threads} pin={pin} iters={iters} prec={prec:?}");
    println!("{:>6} {:>9} {:>9} {:>9}", "rows", "p50 ms", "p95 ms", "GFLOP/s");
    for &n in &[1usize, 10, 50, 150, 400, 1000] {
        let (p50, p95) = bench_one(&engine, n, iters, n > 1);
        let rows = if n > 1 { n + 1 } else { 1 };
        let gflops = 2.0 * ROW_MACS * rows as f64 / (p50 / 1e3) / 1e9;
        println!("{rows:>6} {p50:>9.3} {p95:>9.3} {gflops:>9.1}");
    }

    #[cfg(feature = "bench-compare")]
    faer_compare::run(&model_path, iters);
    #[cfg(not(feature = "bench-compare"))]
    let _ = iters;
}

#[cfg(feature = "bench-compare")]
mod faer_compare {
    use super::*;
    use faer::linalg::matmul::matmul;
    use faer::{Accum, Mat, Par};
    use model_server_rs::simd::swish_slice;

    pub fn run(model_path: &str, iters: usize) {
        let params = Params::load(Path::new(model_path));
        let par = Par::rayon(0);
        let to_mat = |w: &model_server_rs::weights::Mat| -> Mat<f32> {
            Mat::from_fn(w.k, w.n, |i, j| w.w[i * w.n + j])
        };
        let bott = to_mat(&params.bott.w);
        let iu1 = to_mat(&params.item_up1.w);
        let iu2 = to_mat(&params.item_up2.w);
        let iout = to_mat(&params.item_out.w);
        let ru1 = to_mat(&params.rat_up1.w);
        let ru2 = to_mat(&params.rat_up2.w);
        let rout = to_mat(&params.rat_out.w);

        println!("\nfaer decoder chain (same shapes, generic GEMM)");
        println!("{:>6} {:>9} {:>9}", "rows", "p50 ms", "GFLOP/s");
        for &rows in &[1usize, 11, 51, 151, 401, 1001] {
            let a = Mat::<f32>::from_fn(rows, HIDDEN, |i, j| ((i * 7 + j * 13) % 100) as f32 * 0.01 - 0.5);
            let mut times = Vec::new();
            for _ in 0..iters.min(20) {
                let t0 = Instant::now();
                let mut z = Mat::<f32>::zeros(rows, 512);
                matmul(z.as_mut(), Accum::Replace, a.as_ref(), bott.as_ref(), 1.0f32, par);
                let mut d1 = Mat::<f32>::zeros(rows, 1024);
                matmul(d1.as_mut(), Accum::Replace, z.as_ref(), iu1.as_ref(), 1.0f32, par);
                swish_mat(&mut d1);
                let mut d1b = Mat::<f32>::zeros(rows, 2048);
                matmul(d1b.as_mut(), Accum::Replace, d1.as_ref(), iu2.as_ref(), 1.0f32, par);
                swish_mat(&mut d1b);
                let mut logits = Mat::<f32>::zeros(rows, 6000);
                matmul(logits.as_mut(), Accum::Replace, d1b.as_ref(), iout.as_ref(), 1.0f32, par);
                let mut d2 = Mat::<f32>::zeros(rows, 1024);
                matmul(d2.as_mut(), Accum::Replace, z.as_ref(), ru1.as_ref(), 1.0f32, par);
                swish_mat(&mut d2);
                let mut d2b = Mat::<f32>::zeros(rows, 2048);
                matmul(d2b.as_mut(), Accum::Replace, d2.as_ref(), ru2.as_ref(), 1.0f32, par);
                swish_mat(&mut d2b);
                let mut ratings = Mat::<f32>::zeros(rows, 6000);
                matmul(ratings.as_mut(), Accum::Replace, d2b.as_ref(), rout.as_ref(), 1.0f32, par);
                times.push(t0.elapsed().as_secs_f64() * 1e3);
                std::hint::black_box((logits[(0, 0)], ratings[(0, 0)]));
            }
            times.sort_by(f64::total_cmp);
            let p50 = times[times.len() / 2];
            let gflops = 2.0 * ROW_MACS * rows as f64 / (p50 / 1e3) / 1e9;
            println!("{rows:>6} {p50:>9.3} {gflops:>9.1}");
        }
    }

    fn swish_mat(m: &mut Mat<f32>) {
        for j in 0..m.ncols() {
            swish_slice(m.col_as_slice_mut(j));
        }
    }
}
