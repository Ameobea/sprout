use crate::kernels::{self, AVec, AnyPacked, GemmOp, KernCfg};
use crate::pool::Pool;
use crate::simd::swish_slice;
use crate::weights::{Mat, Params};
use crate::{BOTTLENECK, CORPUS, DEC_MID, EASE_PROJ, HIDDEN};
use std::arch::x86_64::*;

pub const OUT_LD: usize = 6016; // CORPUS padded to panel multiple

/// Per-holdout-row delta vs. the full profile: removing entry i drops the presence
/// bit unless a duplicate corpus_idx remains, and shifts the rating input by dval.
#[derive(Clone, Copy, Debug)]
pub struct HoldoutDelta {
    pub idx: u32,
    pub presence_removed: bool,
    pub dval: f32,
}

pub struct ForwardOut {
    pub logits: AVec,
    pub ratings: AVec,
    pub rows: usize,
    /// Graft models only: the full-profile row's raw EASE score vector (pre-norm),
    /// for serve-side stack scoring (ease_lift = e - mu).
    pub ease_full_raw: Option<Vec<f32>>,
}

impl ForwardOut {
    #[inline]
    pub fn logits_row(&self, r: usize) -> &[f32] {
        &self.logits.as_slice()[r * OUT_LD..r * OUT_LD + CORPUS]
    }
    #[inline]
    pub fn ratings_row(&self, r: usize) -> &[f32] {
        &self.ratings.as_slice()[r * OUT_LD..r * OUT_LD + CORPUS]
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Precision {
    F32,
    Bf16,
}

pub struct Engine {
    enc_w: Mat,
    enc_b: Vec<f32>,
    bott: AnyPacked,
    item_up1: AnyPacked,
    item_up2: AnyPacked,
    item_out: AnyPacked,
    rat_up1: AnyPacked,
    rat_up2: AnyPacked,
    rat_out: AnyPacked,
    /// Graft (concat) models: packed 6000x256 projection + row-major EASE B (6000x6000).
    ease_proj: Option<AnyPacked>,
    ease_b: Option<Vec<f32>>,
    pub pool: Pool,
    pub cfg: KernCfg,
}

#[target_feature(enable = "avx512f")]
unsafe fn axpy2(out: &mut [f32], row_p: &[f32], row_v: &[f32], pc: f32, vc: f32) {
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

#[target_feature(enable = "avx512f")]
unsafe fn axpy1(out: &mut [f32], row: &[f32], c: f32) {
    let cv = _mm512_set1_ps(c);
    let mut i = 0;
    let n16 = out.len() & !15;
    while i < n16 {
        let o = _mm512_loadu_ps(out.as_ptr().add(i));
        _mm512_storeu_ps(out.as_mut_ptr().add(i), _mm512_fmadd_ps(cv, _mm512_loadu_ps(row.as_ptr().add(i)), o));
        i += 16;
    }
    while i < out.len() {
        out[i] += c * row[i];
        i += 1;
    }
}

/// Per-row z-norm matching the training-side ease_channel: (e - mean) / (std + 1e-6).
fn znorm_row(x: &mut [f32]) {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x.iter().map(|&v| (v as f64 - mean) * (v as f64 - mean)).sum::<f64>() / n;
    let inv = 1.0 / (var.sqrt() + 1e-6);
    let (m32, i32_) = (mean as f32, inv as f32);
    for v in x.iter_mut() {
        *v = (*v - m32) * i32_;
    }
}

impl Engine {
    pub fn new(p: &Params, nthreads: usize, cfg: KernCfg, pin: Option<&[usize]>, prec: Precision) -> Engine {
        Self::new_with_ease(p, None, nthreads, cfg, pin, prec)
    }

    pub fn new_with_ease(
        p: &Params,
        ease_b: Option<Vec<f32>>,
        nthreads: usize,
        cfg: KernCfg,
        pin: Option<&[usize]>,
        prec: Precision,
    ) -> Engine {
        let nr = cfg.nr;
        let pk = |l: &crate::weights::Layer| match prec {
            Precision::F32 => AnyPacked::F32(kernels::pack(&l.w, &l.b, nr)),
            Precision::Bf16 => AnyPacked::Bf16(kernels::pack_bf16(&l.w, &l.b, 32)),
        };
        if p.ease_proj.is_some() {
            let b = ease_b.as_ref().expect("graft checkpoint requires an EASE B matrix");
            assert_eq!(b.len(), CORPUS * CORPUS, "EASE B shape");
        }
        Engine {
            bott: pk(&p.bott),
            item_up1: pk(&p.item_up1),
            item_up2: pk(&p.item_up2),
            item_out: pk(&p.item_out),
            rat_up1: pk(&p.rat_up1),
            rat_up2: pk(&p.rat_up2),
            rat_out: pk(&p.rat_out),
            ease_proj: p.ease_proj.as_ref().map(&pk),
            ease_b: if p.ease_proj.is_some() { ease_b } else { None },
            enc_w: Mat { k: p.enc1.w.k, n: p.enc1.w.n, w: p.enc1.w.w.clone() },
            enc_b: p.enc1.b.clone(),
            pool: Pool::new(nthreads, pin),
            cfg,
        }
    }

    pub fn is_graft(&self) -> bool {
        self.ease_proj.is_some()
    }

    fn enc_row(&self, idx: usize) -> (&[f32], &[f32]) {
        let n = HIDDEN;
        (
            &self.enc_w.w[idx * n..(idx + 1) * n],
            &self.enc_w.w[(CORPUS + idx) * n..(CORPUS + idx + 1) * n],
        )
    }

    /// Forward pass over the full profile plus optional holdout rows.
    /// Row layout: rows 0..h = holdouts, last row = full profile.
    /// `items` must have unique corpus indices with last-write-wins values already applied.
    pub fn forward(&self, items: &[(u32, f32)], holdout: Option<&[HoldoutDelta]>) -> ForwardOut {
        let h = holdout.map_or(0, |d| d.len());
        let rows = h + 1;

        let mut u_full = vec![0.0f32; HIDDEN];
        u_full.copy_from_slice(&self.enc_b);
        for &(idx, val) in items {
            let (rp, rv) = self.enc_row(idx as usize);
            unsafe { axpy2(&mut u_full, rp, rv, 1.0, val) };
        }

        let mut a_enc = AVec::zeroed(rows * HIDDEN);
        {
            let a_addr = a_enc.as_mut_slice().as_mut_ptr() as usize;
            let u_full = &u_full;
            self.pool.run(&move |tid| {
                let nt = self.pool.n_threads();
                let a = unsafe { std::slice::from_raw_parts_mut(a_addr as *mut f32, rows * HIDDEN) };
                let mut r = tid;
                while r < rows {
                    let dst = unsafe { &mut *(a[r * HIDDEN..(r + 1) * HIDDEN].as_mut_ptr() as *mut [f32; HIDDEN]) };
                    dst.copy_from_slice(u_full);
                    if r < h {
                        let d = &holdout.unwrap()[r];
                        let (rp, rv) = self.enc_row(d.idx as usize);
                        let pc = if d.presence_removed { -1.0 } else { 0.0 };
                        unsafe { axpy2(dst, rp, rv, pc, -d.dval) };
                    }
                    swish_slice(dst);
                    r += nt;
                }
            });
        }

        let mut z = AVec::zeroed(rows * BOTTLENECK);
        self.gemm1(&a_enc, HIDDEN, rows, &self.bott, &mut z, BOTTLENECK, false);

        let mut d1 = AVec::zeroed(rows * DEC_MID);
        let mut d2 = AVec::zeroed(rows * DEC_MID);
        let mut ease_full_raw = None;
        if let Some(ease_proj) = &self.ease_proj {
            let b = self.ease_b.as_ref().unwrap();
            let mut e_full = vec![0.0f32; CORPUS];
            for &(idx, _) in items {
                unsafe { axpy1(&mut e_full, &b[idx as usize * CORPUS..(idx as usize + 1) * CORPUS], 1.0) };
            }
            ease_full_raw = Some(e_full.clone());

            let mut a_ease = AVec::zeroed(rows * CORPUS);
            {
                let a_addr = a_ease.as_mut_slice().as_mut_ptr() as usize;
                let e_full = &e_full;
                self.pool.run(&move |tid| {
                    let nt = self.pool.n_threads();
                    let a = unsafe { std::slice::from_raw_parts_mut(a_addr as *mut f32, rows * CORPUS) };
                    let mut r = tid;
                    while r < rows {
                        let dst = &mut a[r * CORPUS..(r + 1) * CORPUS];
                        dst.copy_from_slice(e_full);
                        if r < h {
                            let d = &holdout.unwrap()[r];
                            if d.presence_removed {
                                let bb = self.ease_b.as_ref().unwrap();
                                unsafe {
                                    axpy1(dst, &bb[d.idx as usize * CORPUS..(d.idx as usize + 1) * CORPUS], -1.0)
                                };
                            }
                        }
                        znorm_row(dst);
                        r += nt;
                    }
                });
            }

            let mut ep = AVec::zeroed(rows * EASE_PROJ);
            self.gemm1(&a_ease, CORPUS, rows, ease_proj, &mut ep, EASE_PROJ, true);

            let zc_ld = BOTTLENECK + EASE_PROJ;
            let mut zc = AVec::zeroed(rows * zc_ld);
            {
                let zs = z.as_slice();
                let eps = ep.as_slice();
                let zcs = zc.as_mut_slice();
                for r in 0..rows {
                    zcs[r * zc_ld..r * zc_ld + BOTTLENECK].copy_from_slice(&zs[r * BOTTLENECK..(r + 1) * BOTTLENECK]);
                    zcs[r * zc_ld + BOTTLENECK..(r + 1) * zc_ld]
                        .copy_from_slice(&eps[r * EASE_PROJ..(r + 1) * EASE_PROJ]);
                }
            }
            self.gemm_pair(
                (&zc, zc_ld, &self.item_up1, &mut d1, DEC_MID),
                (&z, BOTTLENECK, &self.rat_up1, &mut d2, DEC_MID),
                rows,
                true,
            );
        } else {
            self.gemm2(&z, BOTTLENECK, rows, &self.item_up1, &mut d1, &self.rat_up1, &mut d2, DEC_MID, true);
        }

        let mut d1b = AVec::zeroed(rows * HIDDEN);
        let mut d2b = AVec::zeroed(rows * HIDDEN);
        self.gemm_pair(
            (&d1, DEC_MID, &self.item_up2, &mut d1b, HIDDEN),
            (&d2, DEC_MID, &self.rat_up2, &mut d2b, HIDDEN),
            rows,
            true,
        );

        let mut logits = AVec::zeroed(rows * OUT_LD);
        let mut ratings = AVec::zeroed(rows * OUT_LD);
        self.gemm_pair(
            (&d1b, HIDDEN, &self.item_out, &mut logits, OUT_LD),
            (&d2b, HIDDEN, &self.rat_out, &mut ratings, OUT_LD),
            rows,
            false,
        );

        ForwardOut { logits, ratings, rows, ease_full_raw }
    }

    fn gemm1(&self, a: &AVec, lda: usize, m: usize, b: &AnyPacked, c: &mut AVec, ldc: usize, act: bool) {
        let ops = [GemmOp { a: a.as_slice().as_ptr(), lda, m, b, c: c.as_mut_slice().as_mut_ptr(), ldc, act }];
        kernels::gemm_batch(&self.pool, &ops, self.cfg);
    }

    // two heads reading the same A
    fn gemm2(
        &self,
        a: &AVec,
        lda: usize,
        m: usize,
        b1: &AnyPacked,
        c1: &mut AVec,
        b2: &AnyPacked,
        c2: &mut AVec,
        ldc: usize,
        act: bool,
    ) {
        let ap = a.as_slice().as_ptr();
        let ops = [
            GemmOp { a: ap, lda, m, b: b1, c: c1.as_mut_slice().as_mut_ptr(), ldc, act },
            GemmOp { a: ap, lda, m, b: b2, c: c2.as_mut_slice().as_mut_ptr(), ldc, act },
        ];
        kernels::gemm_batch(&self.pool, &ops, self.cfg);
    }

    fn gemm_pair(
        &self,
        (a1, lda1, b1, c1, ldc1): (&AVec, usize, &AnyPacked, &mut AVec, usize),
        (a2, lda2, b2, c2, ldc2): (&AVec, usize, &AnyPacked, &mut AVec, usize),
        m: usize,
        act: bool,
    ) {
        let ops = [
            GemmOp { a: a1.as_slice().as_ptr(), lda: lda1, m, b: b1, c: c1.as_mut_slice().as_mut_ptr(), ldc: ldc1, act },
            GemmOp { a: a2.as_slice().as_ptr(), lda: lda2, m, b: b2, c: c2.as_mut_slice().as_mut_ptr(), ldc: ldc2, act },
        ];
        kernels::gemm_batch(&self.pool, &ops, self.cfg);
    }
}
