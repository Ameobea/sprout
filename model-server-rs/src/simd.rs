//! AVX-512 vector math: exp/log2/swish/softmax/pow. ~1e-7 relative accuracy,
//! plenty vs. the f32 model itself.
//!
//! Every function containing intrinsics carries `#[target_feature]`. Without it LLVM
//! refuses to inline the `_mm512_*` intrinsics and emits out-of-line calls that shuffle
//! each __m512 through the stack in 128-bit pieces (~4x slower). `-C target-cpu=znver4`
//! also fixes that, but only when the build actually picks up .cargo/config.toml; the
//! attribute makes the codegen independent of build configuration.

#![allow(clippy::excessive_precision)]

use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn exp_ps(x: __m512) -> __m512 {
    // Cephes expf: e^x = 2^n * e^r with r = x - n*ln2 (Cody-Waite)
    let x = _mm512_max_ps(_mm512_set1_ps(-87.336), _mm512_min_ps(_mm512_set1_ps(88.722), x));
    let n = _mm512_roundscale_ps::<0x08>(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341)));
    let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693359375), x);
    let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.12194440e-4), r);
    let z = _mm512_mul_ps(r, r);
    let mut y = _mm512_set1_ps(1.9875691500e-4);
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.3981999507e-3));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(8.3334519073e-3));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(4.1665795894e-2));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.6666665459e-1));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(5.0000001201e-1));
    y = _mm512_fmadd_ps(y, z, _mm512_add_ps(r, _mm512_set1_ps(1.0)));
    let pow2n = _mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_add_epi32(
        _mm512_cvtps_epi32(n),
        _mm512_set1_epi32(0x7f),
    )));
    _mm512_mul_ps(y, pow2n)
}

#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn log2_ps(x: __m512) -> __m512 {
    // mant in [0.75, 1.5) so the poly argument straddles 0; e = getexp(x * 4/3)
    let m = _mm512_getmant_ps::<_MM_MANT_NORM_P75_1P5, _MM_MANT_SIGN_ZERO>(x);
    let e = _mm512_getexp_ps(_mm512_mul_ps(x, _mm512_set1_ps(4.0 / 3.0)));
    let r = _mm512_sub_ps(m, _mm512_set1_ps(1.0));
    // minimax for log2(1+r), r in [-0.25, 0.5)
    let mut y = _mm512_set1_ps(-0.096163972);
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(0.204127339));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(-0.250652196));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(0.290073565));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(-0.360335294));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(0.480847510));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(-0.721350693));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.442695250));
    _mm512_fmadd_ps(y, r, e)
}

#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn swish_ps(x: __m512) -> __m512 {
    let sig = _mm512_div_ps(
        _mm512_set1_ps(1.0),
        _mm512_add_ps(_mm512_set1_ps(1.0), exp_ps(_mm512_sub_ps(_mm512_setzero_ps(), x))),
    );
    _mm512_mul_ps(x, sig)
}

/// out = x^w elementwise (x >= 0; x == 0 maps to ~0 for w > 0)
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn pow_ps(x: __m512, w: __m512) -> __m512 {
    const LN2: f32 = 0.6931471805599453;
    exp_ps(_mm512_mul_ps(_mm512_mul_ps(w, _mm512_set1_ps(LN2)), log2_ps(x)))
}

pub fn swish_slice(xs: &mut [f32]) {
    unsafe { swish_slice_avx(xs) }
}

#[target_feature(enable = "avx512f")]
unsafe fn swish_slice_avx(xs: &mut [f32]) {
    let n = xs.len();
    let mut i = 0;
    while i + 16 <= n {
        let v = _mm512_loadu_ps(xs.as_ptr().add(i));
        _mm512_storeu_ps(xs.as_mut_ptr().add(i), swish_ps(v));
        i += 16;
    }
    if i < n {
        let mask = (1u16 << (n - i)) - 1;
        let v = _mm512_maskz_loadu_ps(mask, xs.as_ptr().add(i));
        _mm512_mask_storeu_ps(xs.as_mut_ptr().add(i), mask, swish_ps(v));
    }
}

/// Returns (rowmax, sum of exp(x - rowmax)); does not write probabilities.
pub fn softmax_stats(xs: &[f32]) -> (f32, f32) {
    unsafe { softmax_stats_avx(xs) }
}

#[target_feature(enable = "avx512f")]
unsafe fn softmax_stats_avx(xs: &[f32]) -> (f32, f32) {
    let n = xs.len();
    debug_assert_eq!(n % 16, 0);
    let mut vmax = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;
    while i < n {
        vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(xs.as_ptr().add(i)));
        i += 16;
    }
    let max = _mm512_reduce_max_ps(vmax);
    let maxv = _mm512_set1_ps(max);
    let mut vsum = _mm512_setzero_ps();
    i = 0;
    while i < n {
        let e = exp_ps(_mm512_sub_ps(_mm512_loadu_ps(xs.as_ptr().add(i)), maxv));
        vsum = _mm512_add_ps(vsum, e);
        i += 16;
    }
    (max, _mm512_reduce_add_ps(vsum))
}

/// probs[i] = exp(x[i] - max) / sum
pub fn softmax_into(xs: &[f32], max: f32, sum: f32, out: &mut [f32]) {
    unsafe { softmax_into_avx(xs, max, sum, out) }
}

#[target_feature(enable = "avx512f")]
unsafe fn softmax_into_avx(xs: &[f32], max: f32, sum: f32, out: &mut [f32]) {
    let inv = _mm512_set1_ps(1.0 / sum);
    let maxv = _mm512_set1_ps(max);
    let mut i = 0;
    while i < xs.len() {
        let e = exp_ps(_mm512_sub_ps(_mm512_loadu_ps(xs.as_ptr().add(i)), maxv));
        _mm512_storeu_ps(out.as_mut_ptr().add(i), _mm512_mul_ps(e, inv));
        i += 16;
    }
}

/// Default ranking score for a full row: probs^w * max(ratings+1, 0.001)^(1-w).
/// `probs` input holds softmax probabilities.
pub fn combined_score_into(probs: &[f32], ratings: &[f32], w: f32, out: &mut [f32]) {
    unsafe { combined_score_into_avx(probs, ratings, w, out) }
}

#[target_feature(enable = "avx512f")]
unsafe fn combined_score_into_avx(probs: &[f32], ratings: &[f32], w: f32, out: &mut [f32]) {
    let wv = _mm512_set1_ps(w);
    let iw = _mm512_set1_ps(1.0 - w);
    let one = _mm512_set1_ps(1.0);
    let floor = _mm512_set1_ps(0.001);
    let mut i = 0;
    while i < probs.len() {
        let p = _mm512_loadu_ps(probs.as_ptr().add(i));
        let r = _mm512_loadu_ps(ratings.as_ptr().add(i));
        let rb = _mm512_max_ps(_mm512_add_ps(r, one), floor);
        let s = _mm512_mul_ps(pow_ps(p, wv), pow_ps(rb, iw));
        _mm512_storeu_ps(out.as_mut_ptr().add(i), s);
        i += 16;
    }
}

pub fn mean_std_512(xs: &[f32]) -> (f32, f32) {
    unsafe { mean_std_512_avx(xs) }
}

#[target_feature(enable = "avx512f")]
unsafe fn mean_std_512_avx(xs: &[f32]) -> (f32, f32) {
    let mut s = _mm512_setzero_ps();
    let mut i = 0;
    while i < xs.len() {
        s = _mm512_add_ps(s, _mm512_loadu_ps(xs.as_ptr().add(i)));
        i += 16;
    }
    let mean = _mm512_reduce_add_ps(s) / xs.len() as f32;
    let mv = _mm512_set1_ps(mean);
    let mut v = _mm512_setzero_ps();
    i = 0;
    while i < xs.len() {
        let d = _mm512_sub_ps(_mm512_loadu_ps(xs.as_ptr().add(i)), mv);
        v = _mm512_fmadd_ps(d, d, v);
        i += 16;
    }
    (mean, (_mm512_reduce_add_ps(v) / xs.len() as f32).sqrt())
}

/// Scores + softmax probs at a gathered subset of columns for one output row.
/// Avoids materializing full 6000-wide prob/score arrays for holdout rows.
pub fn row_scores_gathered(
    logits: &[f32],
    ratings: &[f32],
    idxs: &[u32],
    w: f32,
    alt: bool,
) -> (Vec<f32>, Vec<f32>) {
    let g = idxs.len();
    let padded = g.div_ceil(16) * 16;
    let (max, sum) = softmax_stats(logits);
    let mut lg = vec![max; padded];
    let mut rg = vec![0.0f32; padded];
    for (i, &idx) in idxs.iter().enumerate() {
        lg[i] = logits[idx as usize];
        rg[i] = ratings[idx as usize];
    }
    let mut probs = vec![0.0f32; padded];
    softmax_into(&lg, max, sum, &mut probs);
    let mut scores = vec![0.0f32; padded];
    if alt {
        let (lm, ls) = mean_std_512(logits);
        let (rm, rs) = mean_std_512(ratings);
        let (a, b) = (w / (ls + 1e-6), (1.0 - w) / (rs + 1e-6));
        for i in 0..g {
            scores[i] = a * (lg[i] - lm) + b * (rg[i] - rm);
        }
    } else {
        combined_score_into(&probs, &rg, w, &mut scores);
    }
    probs.truncate(g);
    scores.truncate(g);
    (scores, probs)
}

/// Alt ranking score: w * zscore(logits) + (1-w) * zscore(ratings)
pub fn alt_score_into(logits: &[f32], ratings: &[f32], w: f32, out: &mut [f32]) {
    let (lm, ls) = mean_std_512(logits);
    let (rm, rs) = mean_std_512(ratings);
    unsafe { alt_score_into_avx(logits, ratings, w, out, lm, ls, rm, rs) }
}

#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn alt_score_into_avx(
    logits: &[f32],
    ratings: &[f32],
    w: f32,
    out: &mut [f32],
    lm: f32,
    ls: f32,
    rm: f32,
    rs: f32,
) {
    let a = _mm512_set1_ps(w / (ls + 1e-6));
    let b = _mm512_set1_ps((1.0 - w) / (rs + 1e-6));
    let lmv = _mm512_set1_ps(lm);
    let rmv = _mm512_set1_ps(rm);
    let mut i = 0;
    while i < logits.len() {
        let l = _mm512_sub_ps(_mm512_loadu_ps(logits.as_ptr().add(i)), lmv);
        let r = _mm512_sub_ps(_mm512_loadu_ps(ratings.as_ptr().add(i)), rmv);
        _mm512_storeu_ps(out.as_mut_ptr().add(i), _mm512_fmadd_ps(l, a, _mm512_mul_ps(r, b)));
        i += 16;
    }
}
