//! SCRATCH BENCH — NOT a production path. Nothing in `serve` references this.
//! Measures the cost of widening the encoder input from 2 channels (12000 dims)
//! to 3 (18000) / 5 (30000) against the real checkpoints and real profile
//! index/flag distributions.

use model_server_rs::engine::{Engine, HoldoutDelta, Precision};
use model_server_rs::kernels::DEFAULT_CFG;
use model_server_rs::simd::swish_slice;
use model_server_rs::weights::Params;
use model_server_rs::{CORPUS, HIDDEN};
use rmpv::Value;
use std::arch::x86_64::*;
use std::path::Path;
use std::time::Instant;

const DATA: &str = "../data";

struct Wide {
    k: usize,
    w: Vec<f32>,
    b: Vec<f32>,
}

fn load_dense0(path: &str) -> Wide {
    let bytes = std::fs::read(path).unwrap();
    let root = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
    let get = |m: &Value, key: &str| -> Value {
        let Value::Map(e) = m else { panic!("map") };
        e.iter().find(|(k, _)| k.as_str() == Some(key)).unwrap().1.clone()
    };
    let arr = |v: &Value| -> (Vec<usize>, Vec<f32>) {
        let Value::Ext(1, p) = v else { panic!("ext1") };
        let Value::Array(parts) = rmpv::decode::read_value(&mut &p[..]).unwrap() else { panic!() };
        let shape: Vec<usize> =
            parts[0].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect();
        let Value::Binary(b) = &parts[2] else { panic!() };
        (shape, b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    };
    let d0 = get(&root, "Dense_0");
    let (ks, kd) = arr(&get(&d0, "kernel"));
    let (_, bd) = arr(&get(&d0, "bias"));
    assert_eq!(ks[1], HIDDEN);
    let m = Wide { k: ks[0], w: kd, b: bd };
    match std::env::var("THP").as_deref() {
        Ok("huge") => advise(&m.w, libc::MADV_HUGEPAGE),
        Ok("nohuge") => advise(&m.w, libc::MADV_NOHUGEPAGE),
        _ => {}
    }
    m
}

fn advise(w: &[f32], flag: i32) {
    let base = w.as_ptr() as usize & !0x1fffff;
    let end = (w.as_ptr() as usize + w.len() * 4 + 0x1fffff) & !0x1fffff;
    unsafe { libc::madvise(base as *mut libc::c_void, end - base, flag) };
}

fn anon_huge_kb() -> u64 {
    std::fs::read_to_string("/proc/self/smaps_rollup")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with("AnonHugePages:"))
        .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
        .unwrap_or(0)
}

#[target_feature(enable = "avx512f")]
unsafe fn axpy_c<const C: usize>(out: &mut [f32], rows: [*const f32; C], coefs: [f32; C]) {
    let cv: [__m512; C] = std::array::from_fn(|j| _mm512_set1_ps(coefs[j]));
    let mut i = 0;
    while i < out.len() {
        let mut o = _mm512_loadu_ps(out.as_ptr().add(i));
        for j in 0..C {
            o = _mm512_fmadd_ps(cv[j], _mm512_loadu_ps(rows[j].add(i)), o);
        }
        _mm512_storeu_ps(out.as_mut_ptr().add(i), o);
        i += 16;
    }
}

/// Identical body to axpy_c but WITHOUT the target_feature attribute, reproducing the
/// codegen of the current engine.rs::axpy2 (intrinsics emitted as out-of-line calls with
/// every __m512 shuffled through the stack in 128-bit pieces).
unsafe fn axpy_slow<const C: usize>(out: &mut [f32], rows: [*const f32; C], coefs: [f32; C]) {
    let cv: [__m512; C] = std::array::from_fn(|j| _mm512_set1_ps(coefs[j]));
    let mut i = 0;
    while i < out.len() {
        let mut o = _mm512_loadu_ps(out.as_ptr().add(i));
        for j in 0..C {
            o = _mm512_fmadd_ps(cv[j], _mm512_loadu_ps(rows[j].add(i)), o);
        }
        _mm512_storeu_ps(out.as_mut_ptr().add(i), o);
        i += 16;
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    Ch2,
    Ch3Sparse,
    Ch3Dense,
    Ch5Sparse,
    Ch5Dense,
}

struct Profile {
    idx: Vec<u32>,
    val: Vec<f32>,
    rated: Vec<bool>,
    dropped: Vec<bool>,
    watching: Vec<bool>,
}

fn enc_gather(m: &Wide, p: &Profile, mode: Mode, fast: bool, out: &mut [f32]) {
    out.copy_from_slice(&m.b);
    let row = |c: usize, i: u32| unsafe { m.w.as_ptr().add((c * CORPUS + i as usize) * HIDDEN) };
    for (j, &i) in p.idx.iter().enumerate() {
        let mut ptrs = [std::ptr::null(); 5];
        let mut cs = [0.0f32; 5];
        let mut n = 0;
        let mut push = |ptr, c| {
            ptrs[n] = ptr;
            cs[n] = c;
            n += 1;
        };
        push(row(0, i), 1.0);
        push(row(1, i), p.val[j]);
        match mode {
            Mode::Ch2 => {}
            Mode::Ch3Dense => push(row(2, i), p.rated[j] as u8 as f32),
            Mode::Ch3Sparse => {
                if p.rated[j] {
                    push(row(2, i), 1.0)
                }
            }
            Mode::Ch5Dense => {
                push(row(2, i), p.rated[j] as u8 as f32);
                push(row(3, i), p.dropped[j] as u8 as f32);
                push(row(4, i), p.watching[j] as u8 as f32);
            }
            Mode::Ch5Sparse => {
                if p.rated[j] {
                    push(row(2, i), 1.0)
                }
                if p.dropped[j] {
                    push(row(3, i), 1.0)
                }
                if p.watching[j] {
                    push(row(4, i), 1.0)
                }
            }
        }
        drop(push);
        unsafe {
            match (n, fast) {
                (2, true) => axpy_c::<2>(out, [ptrs[0], ptrs[1]], [cs[0], cs[1]]),
                (3, true) => axpy_c::<3>(out, [ptrs[0], ptrs[1], ptrs[2]], [cs[0], cs[1], cs[2]]),
                (4, true) => {
                    axpy_c::<4>(out, [ptrs[0], ptrs[1], ptrs[2], ptrs[3]], [cs[0], cs[1], cs[2], cs[3]])
                }
                (_, true) => axpy_c::<5>(out, ptrs, cs),
                (2, false) => axpy_slow::<2>(out, [ptrs[0], ptrs[1]], [cs[0], cs[1]]),
                (3, false) => axpy_slow::<3>(out, [ptrs[0], ptrs[1], ptrs[2]], [cs[0], cs[1], cs[2]]),
                (4, false) => {
                    axpy_slow::<4>(out, [ptrs[0], ptrs[1], ptrs[2], ptrs[3]], [cs[0], cs[1], cs[2], cs[3]])
                }
                (_, false) => axpy_slow::<5>(out, ptrs, cs),
            }
        }
    }
    swish_slice(out);
}

// xorshift, matching the style of the existing bench
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Empirical corpus-index distribution from the 300 real fixture profiles,
/// as a CDF over 6000 indices (loaded from the scratch json).
fn load_empirical() -> (Vec<f64>, f64, f64, f64) {
    let txt = std::fs::read_to_string(
        std::env::var("REAL_PROFILES").unwrap_or_else(|_| "/tmp/real_profiles.json".into()),
    )
    .unwrap();
    let v: serde_json::Value = serde_json::from_str(&txt).unwrap();
    let mut hist = vec![0.0f64; CORPUS];
    let (mut tot, mut r, mut d, mut w) = (0.0, 0.0, 0.0, 0.0);
    for p in v.as_array().unwrap() {
        let idx = p["idx"].as_array().unwrap();
        for (j, i) in idx.iter().enumerate() {
            hist[i.as_u64().unwrap() as usize] += 1.0;
            tot += 1.0;
            r += p["rated"][j].as_u64().unwrap() as f64;
            d += p["dropped"][j].as_u64().unwrap() as f64;
            w += p["watching"][j].as_u64().unwrap() as f64;
        }
    }
    let mut acc = 0.0;
    for h in hist.iter_mut() {
        acc += *h / tot;
        *h = acc;
    }
    (hist, r / tot, d / tot, w / tot)
}

fn gen_profiles(cdf: &[f64], rates: (f64, f64, f64), n: usize, count: usize, seed: u64) -> Vec<Profile> {
    let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
    (0..count)
        .map(|_| {
            let mut idx: Vec<u32> = Vec::with_capacity(n);
            while idx.len() < n {
                let u = rng.f();
                let i = cdf.partition_point(|&c| c < u).min(CORPUS - 1) as u32;
                if !idx.contains(&i) {
                    idx.push(i);
                }
            }
            idx.sort_unstable();
            let val: Vec<f32> = (0..n).map(|_| (rng.f() as f32) * 4.0 - 2.0).collect();
            let rated: Vec<bool> = (0..n).map(|_| rng.f() < rates.0).collect();
            let dropped: Vec<bool> = (0..n).map(|_| rng.f() < rates.1).collect();
            let watching: Vec<bool> = (0..n).map(|_| rng.f() < rates.2).collect();
            Profile { idx, val, rated, dropped, watching }
        })
        .collect()
}

fn pct(v: &mut Vec<f64>, q: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() as f64 * q) as usize).min(v.len() - 1)]
}

/// `cold`: run a full decoder pass (engine.forward with an empty profile) between
/// encoder timings. That is exactly what a real request does after the gather, and it
/// streams ~62MB of packed bf16 weights through the 32MB L3, so the next gather starts
/// cold — as it always does in production. Without it the gather reads a warm L3 and
/// looks ~6x faster than it is.
fn bench_enc(m: &Wide, profs: &[Profile], mode: Mode, fast: bool, iters: usize, cold: Option<&Engine>) -> (f64, f64) {
    let mut out = vec![0.0f32; HIDDEN];
    let mut ts = Vec::with_capacity(iters);
    let mut sink = 0.0f32;
    for it in 0..iters {
        let p = &profs[it % profs.len()];
        if let Some(e) = cold {
            let o = e.forward(&[], None);
            sink += o.logits_row(0)[0];
        }
        let t0 = Instant::now();
        enc_gather(m, p, mode, fast, &mut out);
        ts.push(t0.elapsed().as_secs_f64() * 1e3);
        sink += out[0];
    }
    std::hint::black_box(sink);
    (pct(&mut ts, 0.5), pct(&mut ts, 0.99))
}

/// Fixed (n-independent) cost of a rows=1 request: everything after the gather.
fn bench_decoder_fixed(e: &Engine, iters: usize) -> (f64, f64) {
    let mut ts = Vec::with_capacity(iters);
    let mut sink = 0.0f32;
    for _ in 0..iters {
        let t0 = Instant::now();
        let o = e.forward(&[], None);
        ts.push(t0.elapsed().as_secs_f64() * 1e3);
        sink += o.logits_row(0)[0];
    }
    std::hint::black_box(sink);
    (pct(&mut ts, 0.5), pct(&mut ts, 0.99))
}

fn bench_fwd(e: &Engine, profs: &[Profile], holdout: bool, iters: usize) -> (f64, f64) {
    let items: Vec<Vec<(u32, f32)>> = profs
        .iter()
        .map(|p| p.idx.iter().cloned().zip(p.val.iter().cloned()).collect())
        .collect();
    let deltas: Vec<Vec<HoldoutDelta>> = items
        .iter()
        .map(|is| {
            is.iter().map(|&(idx, dval)| HoldoutDelta { idx, presence_removed: true, dval }).collect()
        })
        .collect();
    let mut ts = Vec::with_capacity(iters);
    let mut sink = 0.0f32;
    for it in 0..iters {
        let k = it % profs.len();
        let hd = if holdout { Some(&deltas[k][..]) } else { None };
        let t0 = Instant::now();
        let out = e.forward(&items[k], hd);
        ts.push(t0.elapsed().as_secs_f64() * 1e3);
        sink += out.logits_row(out.rows - 1)[0];
    }
    std::hint::black_box(sink);
    (pct(&mut ts, 0.5), pct(&mut ts, 0.99))
}

/// Byte-for-byte replica of engine.rs::axpy2 (slice args, no target_feature) so the
/// gather replica can be compared against the production loop on equal codegen terms.
unsafe fn axpy2_replica(out: &mut [f32], row_p: &[f32], row_v: &[f32], pc: f32, vc: f32) {
    let pcv = _mm512_set1_ps(pc);
    let vcv = _mm512_set1_ps(vc);
    let mut i = 0;
    while i < out.len() {
        let mut o = _mm512_loadu_ps(out.as_ptr().add(i));
        o = _mm512_fmadd_ps(pcv, _mm512_loadu_ps(row_p.as_ptr().add(i)), o);
        o = _mm512_fmadd_ps(vcv, _mm512_loadu_ps(row_v.as_ptr().add(i)), o);
        _mm512_storeu_ps(out.as_mut_ptr().add(i), o);
        i += 16;
    }
}

/// Same body as axpy2_replica but with the target_feature attribute, to isolate whether
/// the 3.5x gap is the attribute (codegen) or the slice-vs-pointer argument form.
#[target_feature(enable = "avx512f")]
unsafe fn axpy2_tf(out: &mut [f32], row_p: &[f32], row_v: &[f32], pc: f32, vc: f32) {
    let pcv = _mm512_set1_ps(pc);
    let vcv = _mm512_set1_ps(vc);
    let mut i = 0;
    while i < out.len() {
        let mut o = _mm512_loadu_ps(out.as_ptr().add(i));
        o = _mm512_fmadd_ps(pcv, _mm512_loadu_ps(row_p.as_ptr().add(i)), o);
        o = _mm512_fmadd_ps(vcv, _mm512_loadu_ps(row_v.as_ptr().add(i)), o);
        _mm512_storeu_ps(out.as_mut_ptr().add(i), o);
        i += 16;
    }
}

fn enc_gather_tf(m: &Wide, p: &Profile, out: &mut [f32]) {
    out.copy_from_slice(&m.b);
    for (j, &i) in p.idx.iter().enumerate() {
        let a = i as usize * HIDDEN;
        let b = (CORPUS + i as usize) * HIDDEN;
        unsafe { axpy2_tf(out, &m.w[a..a + HIDDEN], &m.w[b..b + HIDDEN], 1.0, p.val[j]) };
    }
    swish_slice(out);
}

fn enc_gather_replica(m: &Wide, p: &Profile, out: &mut [f32]) {
    out.copy_from_slice(&m.b);
    for (j, &i) in p.idx.iter().enumerate() {
        let a = i as usize * HIDDEN;
        let b = (CORPUS + i as usize) * HIDDEN;
        unsafe {
            axpy2_replica(out, &m.w[a..a + HIDDEN], &m.w[b..b + HIDDEN], 1.0, p.val[j]);
        }
    }
    swish_slice(out);
}

fn diag(engine: &Engine, m2: &Wide, profs: &[Profile], iters: usize) {
    let items: Vec<Vec<(u32, f32)>> =
        profs.iter().map(|p| p.idx.iter().cloned().zip(p.val.iter().cloned()).collect()).collect();
    let mut out = vec![0.0f32; HIDDEN];
    let (mut ta, mut tb, mut tc, mut td, mut te) = (vec![], vec![], vec![], vec![], vec![]);
    let mut sink = 0.0f32;
    for it in 0..iters {
        let k = it % profs.len();
        let t = Instant::now();
        let o = engine.forward(&items[k], None);
        ta.push(t.elapsed().as_secs_f64() * 1e3);
        sink += o.logits_row(0)[0];

        let t = Instant::now();
        let o = engine.forward(&[], None);
        tb.push(t.elapsed().as_secs_f64() * 1e3);
        sink += o.logits_row(0)[0];

        let t = Instant::now();
        enc_gather(m2, &profs[k], Mode::Ch2, true, &mut out);
        tc.push(t.elapsed().as_secs_f64() * 1e3);
        sink += out[0];

        let o = engine.forward(&[], None);
        sink += o.logits_row(0)[0];
        let t = Instant::now();
        enc_gather_replica(m2, &profs[k], &mut out);
        td.push(t.elapsed().as_secs_f64() * 1e3);
        sink += out[0];

        let o = engine.forward(&[], None);
        sink += o.logits_row(0)[0];
        let t = Instant::now();
        enc_gather_tf(m2, &profs[k], &mut out);
        te.push(t.elapsed().as_secs_f64() * 1e3);
        sink += out[0];
    }
    std::hint::black_box(sink);

    // equivalence check: const-N vs replica must agree bitwise-ish
    let mut o1 = vec![0.0f32; HIDDEN];
    let mut o2 = vec![0.0f32; HIDDEN];
    enc_gather(m2, &profs[0], Mode::Ch2, true, &mut o1);
    enc_gather_replica(m2, &profs[0], &mut o2);
    let maxdiff = o1.iter().zip(&o2).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

    let n = profs[0].idx.len() as f64;
    let (a, b, c, d, e) = (
        pct(&mut ta, 0.5),
        pct(&mut tb, 0.5),
        pct(&mut tc, 0.5),
        pct(&mut td, 0.5),
        pct(&mut te, 0.5),
    );
    println!("\n=== DIAG n={} (max |constN - replica| = {maxdiff:.2e}) ===", n as usize);
    println!("  A forward(items,None)          {a:8.3} ms");
    println!("  B forward(&[],None)            {b:8.3} ms");
    println!("  A-B implied engine gather      {:8.3} ms  ({:.3} us/item)", a - b, (a - b) * 1e3 / n);
    println!("  C scratch gather (const-N ptr) {c:8.3} ms  ({:.3} us/item)", c * 1e3 / n);
    println!("  D scratch gather (replica)     {d:8.3} ms  ({:.3} us/item)", d * 1e3 / n);
    println!("  E replica + target_feature     {e:8.3} ms  ({:.3} us/item)", e * 1e3 / n);
}

fn main() {
    let iters: usize = std::env::var("ITERS").map_or(400, |v| v.parse().unwrap());
    let threads: usize = std::env::var("INFER_THREADS").map_or(8, |v| v.parse().unwrap());
    let sizes: Vec<usize> = std::env::var("SIZES")
        .unwrap_or_else(|_| "50,250,800".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();

    let (cdf, r_rate, d_rate, w_rate) = load_empirical();
    println!("empirical flag rates: rated={r_rate:.4} dropped={d_rate:.4} watching={w_rate:.4}");
    println!("=> rows/item: 2ch=2.000 3ch={:.3} 5ch={:.3}", 2.0 + r_rate, 2.0 + r_rate + d_rate + w_rate);

    // Real 2-channel engine, prod config (bf16, 8 threads pinned to cores 0-7).
    let pins: Vec<usize> = (0..threads).collect();
    let params = Params::load(Path::new(&format!("{DATA}/jax_model_dec2025.msgpack")));
    let engine = Engine::new(&params, threads, DEFAULT_CFG, Some(&pins[..]), Precision::Bf16);
    drop(params);

    println!("\nloading Dense_0 matrices...");
    let m2 = load_dense0(&format!("{DATA}/jax_model_dec2025.msgpack"));
    let m3 = load_dense0(&format!("{DATA}/jax_model_maskchan.msgpack"));
    let m5 = load_dense0(&format!("{DATA}/jax_model_statuschan.msgpack"));
    for m in [&m2, &m3, &m5] {
        println!("  k={:5} ({} MB)", m.k, m.w.len() * 4 / 1_000_000);
    }
    println!("THP={:?} process AnonHugePages={} MB", std::env::var("THP"), anon_huge_kb() / 1024);

    let nprof = 48;
    if std::env::var("DIAG").is_ok() {
        for &n in &sizes {
            let profs = gen_profiles(&cdf, (r_rate, d_rate, w_rate), n, nprof, n as u64);
            diag(&engine, &m2, &profs, iters.min(200));
        }
        return;
    }
    bench_decoder_fixed(&engine, 32);
    let (dfix, dfix99) = bench_decoder_fixed(&engine, iters);
    println!("\nfixed rows=1 cost after the gather (empty-profile forward): p50={dfix:.3} p99={dfix99:.3} ms");

    let combos = [
        (&m2, Mode::Ch2, "2ch  (today)"),
        (&m3, Mode::Ch3Sparse, "3ch  sparse-flag"),
        (&m3, Mode::Ch3Dense, "3ch  always-3-rows"),
        (&m5, Mode::Ch5Sparse, "5ch  sparse-flag"),
        (&m5, Mode::Ch5Dense, "5ch  always-5-rows"),
    ];

    for (fast, regime) in [(false, "CURRENT codegen (axpy2 style, no target_feature)"), (true, "FIXED codegen (+#[target_feature])")] {
        println!("\n################ {regime} ################");
        for &n in &sizes {
            let profs = gen_profiles(&cdf, (r_rate, d_rate, w_rate), n, nprof, n as u64);
            // measured real 2ch request latencies for this n
            bench_fwd(&engine, &profs, false, 8);
            let (f1, f1_99) = bench_fwd(&engine, &profs, false, iters);
            bench_fwd(&engine, &profs, true, 4);
            let (fh, fh_99) = bench_fwd(&engine, &profs, true, iters.min(80));
            bench_enc(&m2, &profs, Mode::Ch2, fast, 32, Some(&engine));
            let (base, _) = bench_enc(&m2, &profs, Mode::Ch2, fast, iters, Some(&engine));

            println!(
                "\n n={n}  measured 2ch forward: rows=1 {f1:.3}/{f1_99:.3} ms   rows=n+1 {fh:.3}/{fh_99:.3} ms (p50/p99)\n\
                 {:>20} {:>9} {:>9} {:>10} {:>9} {:>11} {:>9}",
                "layout", "gather us", "us/item", "rows=1 ms", "d vs 2ch", "rows=n+1 ms", "d vs 2ch"
            );
            for (m, mode, label) in combos {
                bench_enc(m, &profs, mode, fast, 32, Some(&engine));
                let (g, _) = bench_enc(m, &profs, mode, fast, iters, Some(&engine));
                let r1 = f1 - base + g;
                let rh = fh - base + g;
                println!(
                    "{label:>20} {:>9.1} {:>9.3} {:>10.3} {:>8.1}% {:>11.3} {:>8.1}%",
                    g * 1e3,
                    g * 1e3 / n as f64,
                    r1,
                    100.0 * (r1 - f1) / f1,
                    rh,
                    100.0 * (rh - fh) / fh
                );
            }
        }
    }
    let _ = dfix;
}
