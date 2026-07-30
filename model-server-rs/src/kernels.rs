//! Pre-packed-weight GEMM for the fixed model shapes.
//!
//! Weights are packed once at load into panel-major layout: panel p holds columns
//! [p*NR, (p+1)*NR) for all K rows contiguously, so the microkernel streams B
//! linearly and C accumulators live in registers for the entire K loop (no K
//! blocking, C written exactly once, bias+swish fused into the store).

use crate::pool::Pool;
use crate::simd::swish_ps;
use crate::weights::Mat;
use std::arch::x86_64::*;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

#[repr(align(64))]
#[derive(Clone, Copy)]
struct CacheLine([f32; 16]);

pub struct AVec {
    buf: Vec<CacheLine>,
    len: usize,
}

impl AVec {
    pub fn zeroed(len: usize) -> AVec {
        let blocks = len.div_ceil(16);
        let buf = vec![CacheLine([0.0; 16]); blocks];
        unsafe {
            libc::madvise(
                buf.as_ptr() as *mut libc::c_void,
                blocks * 64,
                libc::MADV_HUGEPAGE,
            );
        }
        AVec { buf, len }
    }
    #[inline(always)]
    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.buf.as_ptr() as *const f32, self.len) }
    }
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.buf.as_mut_ptr() as *mut f32, self.len) }
    }
}

pub struct PackedMat {
    pub k: usize,
    pub n: usize,
    pub nr: usize,
    pub npanels: usize,
    data: AVec,
    bias: AVec, // padded to npanels * nr
}

pub fn pack(w: &Mat, bias: &[f32], nr: usize) -> PackedMat {
    assert_eq!(nr % 16, 0);
    let (k, n) = (w.k, w.n);
    let npanels = n.div_ceil(nr);
    let mut data = AVec::zeroed(npanels * k * nr);
    let mut biasp = AVec::zeroed(npanels * nr);
    {
        let d = data.as_mut_slice();
        for p in 0..npanels {
            let base = p * k * nr;
            let c0 = p * nr;
            let cols = nr.min(n - c0);
            for kk in 0..k {
                let src = &w.w[kk * n + c0..kk * n + c0 + cols];
                d[base + kk * nr..base + kk * nr + cols].copy_from_slice(src);
            }
        }
        biasp.as_mut_slice()[..n].copy_from_slice(bias);
    }
    PackedMat { k, n, nr, npanels, data, bias: biasp }
}

pub enum AnyPacked {
    F32(PackedMat),
    Bf16(PackedMatBf16),
}

impl AnyPacked {
    pub fn npanels(&self) -> usize {
        match self {
            AnyPacked::F32(b) => b.npanels,
            AnyPacked::Bf16(b) => b.npanels,
        }
    }
}

/// One GEMM in a fused parallel batch: C[m x n] = act(A[m x k] * B + bias)
pub struct GemmOp<'a> {
    pub a: *const f32,
    pub lda: usize,
    pub m: usize,
    pub b: &'a AnyPacked,
    pub c: *mut f32,
    pub ldc: usize,
    pub act: bool,
}
unsafe impl Send for GemmOp<'_> {}
unsafe impl Sync for GemmOp<'_> {}

macro_rules! kern {
    ($name:ident, $MR:literal, $NV:literal) => {
        #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
        unsafe fn $name(
            a: *const f32,
            lda: usize,
            bp: *const f32,
            kc: usize,
            bias: *const f32,
            c: *mut f32,
            ldc: usize,
            act: bool,
        ) {
            let mut acc = [[_mm512_setzero_ps(); $NV]; $MR];
            for v in 0..$NV {
                let b = _mm512_load_ps(bias.add(v * 16));
                for m in 0..$MR {
                    acc[m][v] = b;
                }
            }
            let mut bptr = bp;
            for kk in 0..kc {
                let mut bv = [_mm512_setzero_ps(); $NV];
                for v in 0..$NV {
                    bv[v] = _mm512_load_ps(bptr.add(v * 16));
                }
                for m in 0..$MR {
                    let av = _mm512_set1_ps(*a.add(m * lda + kk));
                    for v in 0..$NV {
                        acc[m][v] = _mm512_fmadd_ps(av, bv[v], acc[m][v]);
                    }
                }
                bptr = bptr.add($NV * 16);
            }
            for m in 0..$MR {
                for v in 0..$NV {
                    let mut r = acc[m][v];
                    if act {
                        r = swish_ps(r);
                    }
                    _mm512_storeu_ps(c.add(m * ldc + v * 16), r);
                }
            }
        }
    };
}

kern!(kern_1x64, 1, 4);
kern!(kern_2x64, 2, 4);
kern!(kern_3x64, 3, 4);
kern!(kern_4x64, 4, 4);
kern!(kern_5x64, 5, 4);
kern!(kern_6x64, 6, 4);
kern!(kern_1x32, 1, 2);
kern!(kern_2x32, 2, 2);
kern!(kern_3x32, 3, 2);
kern!(kern_4x32, 4, 2);
kern!(kern_5x32, 5, 2);
kern!(kern_6x32, 6, 2);
kern!(kern_7x32, 7, 2);
kern!(kern_8x32, 8, 2);
kern!(kern_9x32, 9, 2);
kern!(kern_10x32, 10, 2);
kern!(kern_11x32, 11, 2);
kern!(kern_12x32, 12, 2);

type KernFn = unsafe fn(*const f32, usize, *const f32, usize, *const f32, *mut f32, usize, bool);

static KERNS_64: [KernFn; 6] = [kern_1x64, kern_2x64, kern_3x64, kern_4x64, kern_5x64, kern_6x64];
static KERNS_32: [KernFn; 12] = [
    kern_1x32, kern_2x32, kern_3x32, kern_4x32, kern_5x32, kern_6x32, kern_7x32, kern_8x32,
    kern_9x32, kern_10x32, kern_11x32, kern_12x32,
];

/// bf16 panels: same column layout as f32, but K consumed in pairs — each 32-bit
/// lane of a panel row holds {B[k][c], B[k+1][c]} for VDPBF16PS.
pub struct PackedMatBf16 {
    pub k: usize,
    pub n: usize,
    pub nr: usize,
    pub npanels: usize,
    kpairs: usize,
    data: AVec, // reinterpreted as u32 lanes of 2xbf16
    bias: AVec,
}

#[inline(always)]
fn f32_to_bf16(x: f32) -> u16 {
    // round-to-nearest-even, matching VCVTNEPS2BF16
    let bits = x.to_bits();
    let round = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(round)) >> 16) as u16
}

pub fn pack_bf16(w: &Mat, bias: &[f32], nr: usize) -> PackedMatBf16 {
    assert_eq!(nr % 16, 0);
    let (k, n) = (w.k, w.n);
    let npanels = n.div_ceil(nr);
    let kpairs = k.div_ceil(2);
    let mut data = AVec::zeroed(npanels * kpairs * nr);
    let mut biasp = AVec::zeroed(npanels * nr);
    {
        let d = data.as_mut_slice();
        for p in 0..npanels {
            let base = p * kpairs * nr;
            let c0 = p * nr;
            let cols = nr.min(n - c0);
            for kp in 0..kpairs {
                for c in 0..cols {
                    let lo = f32_to_bf16(w.w[(2 * kp) * n + c0 + c]);
                    let hi = if 2 * kp + 1 < k { f32_to_bf16(w.w[(2 * kp + 1) * n + c0 + c]) } else { 0 };
                    d[base + kp * nr + c] = f32::from_bits((lo as u32) | ((hi as u32) << 16));
                }
            }
        }
        biasp.as_mut_slice()[..n].copy_from_slice(bias);
    }
    PackedMatBf16 { k, n, nr, npanels, kpairs, data, bias: biasp }
}

macro_rules! kern_bf16 {
    ($name:ident, $MR:literal, $NV:literal) => {
        #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512bf16")]
        unsafe fn $name(
            a: *const u16, // bf16 activations, row stride lda (elements)
            lda: usize,
            bp: *const f32,
            kpairs: usize,
            bias: *const f32,
            c: *mut f32,
            ldc: usize,
            act: bool,
        ) {
            use std::mem::transmute;
            let mut acc = [[_mm512_setzero_ps(); $NV]; $MR];
            for v in 0..$NV {
                let b = _mm512_load_ps(bias.add(v * 16));
                for m in 0..$MR {
                    acc[m][v] = b;
                }
            }
            let mut bptr = bp as *const i32;
            for kp in 0..kpairs {
                let mut bv = [_mm512_setzero_si512(); $NV];
                for v in 0..$NV {
                    bv[v] = _mm512_load_si512(bptr.add(v * 16) as *const __m512i);
                }
                for m in 0..$MR {
                    let pair = *(a.add(m * lda + kp * 2) as *const i32);
                    let av: __m512bh = transmute(_mm512_set1_epi32(pair));
                    for v in 0..$NV {
                        acc[m][v] = _mm512_dpbf16_ps(acc[m][v], av, transmute(bv[v]));
                    }
                }
                bptr = bptr.add($NV * 16);
            }
            for m in 0..$MR {
                for v in 0..$NV {
                    let mut r = acc[m][v];
                    if act {
                        r = swish_ps(r);
                    }
                    _mm512_storeu_ps(c.add(m * ldc + v * 16), r);
                }
            }
        }
    };
}

kern_bf16!(kb_1x32, 1, 2);
kern_bf16!(kb_2x32, 2, 2);
kern_bf16!(kb_3x32, 3, 2);
kern_bf16!(kb_4x32, 4, 2);
kern_bf16!(kb_5x32, 5, 2);
kern_bf16!(kb_6x32, 6, 2);
kern_bf16!(kb_7x32, 7, 2);
kern_bf16!(kb_8x32, 8, 2);

type KernBf16Fn = unsafe fn(*const u16, usize, *const f32, usize, *const f32, *mut f32, usize, bool);
static KERNS_BF16_32: [KernBf16Fn; 8] =
    [kb_1x32, kb_2x32, kb_3x32, kb_4x32, kb_5x32, kb_6x32, kb_7x32, kb_8x32];

/// Converts f32 activations (m x lda) to a bf16 buffer with the same row stride.
pub fn a_to_bf16(a: *const f32, m: usize, lda: usize, out: &mut Vec<u16>) {
    out.clear();
    out.resize(m * lda + 2, 0); // +2: kernel k-pair tail may read one element past odd K
    unsafe {
        let n = m * lda;
        let mut i = 0;
        while i + 32 <= n {
            let lo = _mm512_loadu_ps(a.add(i));
            let hi = _mm512_loadu_ps(a.add(i + 16));
            let packed = _mm512_cvtne2ps_pbh(hi, lo);
            _mm512_storeu_si512(out.as_mut_ptr().add(i) as *mut __m512i, std::mem::transmute(packed));
            i += 32;
        }
        while i < n {
            out[i] = f32_to_bf16(*a.add(i));
            i += 1;
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct KernCfg {
    pub nr: usize,
    pub mr: usize,
}

// Winner of the on-target sweep: 1071 GFLOP/s @ 401 rows on the EPYC 4344P.
// mr=12 spills accumulators; SMT (16 threads) hurts both p50 and p95.
pub const DEFAULT_CFG: KernCfg = KernCfg { nr: 32, mr: 8 };

fn panel_body(op: &GemmOp, a16: Option<&[u16]>, panel: usize, cfg: KernCfg) {
    match op.b {
        AnyPacked::F32(b) => {
            let (kerns, mr_max): (&[KernFn], usize) = match b.nr {
                64 => (&KERNS_64[..], cfg.mr.min(6)),
                32 => (&KERNS_32[..], cfg.mr.min(12)),
                _ => unreachable!(),
            };
            let bp = unsafe { b.data.as_slice().as_ptr().add(panel * b.k * b.nr) };
            let bias = unsafe { b.bias.as_slice().as_ptr().add(panel * b.nr) };
            let c0 = panel * b.nr;
            let mut m0 = 0;
            while m0 < op.m {
                let mr = mr_max.min(op.m - m0);
                unsafe {
                    kerns[mr - 1](
                        op.a.add(m0 * op.lda),
                        op.lda,
                        bp,
                        b.k,
                        bias,
                        op.c.add(m0 * op.ldc + c0),
                        op.ldc,
                        op.act,
                    );
                }
                m0 += mr;
            }
        }
        AnyPacked::Bf16(b) => {
            assert_eq!(b.nr, 32);
            let a16 = a16.unwrap();
            let mr_max = cfg.mr.min(8);
            let bp = unsafe { b.data.as_slice().as_ptr().add(panel * b.kpairs * b.nr) };
            let bias = unsafe { b.bias.as_slice().as_ptr().add(panel * b.nr) };
            let c0 = panel * b.nr;
            let mut m0 = 0;
            while m0 < op.m {
                let mr = mr_max.min(op.m - m0);
                unsafe {
                    KERNS_BF16_32[mr - 1](
                        a16.as_ptr().add(m0 * op.lda),
                        op.lda,
                        bp,
                        b.kpairs,
                        bias,
                        op.c.add(m0 * op.ldc + c0),
                        op.ldc,
                        op.act,
                    );
                }
                m0 += mr;
            }
        }
    }
}

/// Runs a set of GEMMs as one fork-join region; work unit = (op, panel),
/// distributed dynamically via an atomic counter.
pub fn gemm_batch(pool: &Pool, ops: &[GemmOp], cfg: KernCfg) {
    let a16s: Vec<Option<Vec<u16>>> = ops
        .iter()
        .map(|op| match op.b {
            AnyPacked::Bf16(_) => {
                let mut v = Vec::new();
                a_to_bf16(op.a, op.m, op.lda, &mut v);
                Some(v)
            }
            AnyPacked::F32(_) => None,
        })
        .collect();
    let bounds: Vec<usize> = ops
        .iter()
        .scan(0, |acc, op| {
            *acc += op.b.npanels();
            Some(*acc)
        })
        .collect();
    let total = *bounds.last().unwrap();
    let counter = AtomicUsize::new(0);
    pool.run(&|_tid| loop {
        let w = counter.fetch_add(1, Relaxed);
        if w >= total {
            break;
        }
        let oi = bounds.partition_point(|&b| b <= w);
        let panel = w - if oi == 0 { 0 } else { bounds[oi - 1] };
        panel_body(&ops[oi], a16s[oi].as_deref(), panel, cfg);
    });
}
