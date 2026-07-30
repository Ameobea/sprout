use crate::kernels::{self, AVec, AnyPacked, GemmOp, KernCfg};
use crate::pool::Pool;
use crate::simd::swish_slice;
use crate::weights::{Mat, Params};
use crate::{BOTTLENECK, CORPUS, DEC_MID, HIDDEN};
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
    pub pool: Pool,
    pub cfg: KernCfg,
}

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

impl Engine {
    pub fn new(p: &Params, nthreads: usize, cfg: KernCfg, pin: Option<&[usize]>, prec: Precision) -> Engine {
        let nr = cfg.nr;
        let pk = |l: &crate::weights::Layer| match prec {
            Precision::F32 => AnyPacked::F32(kernels::pack(&l.w, &l.b, nr)),
            Precision::Bf16 => AnyPacked::Bf16(kernels::pack_bf16(&l.w, &l.b, 32)),
        };
        Engine {
            bott: pk(&p.bott),
            item_up1: pk(&p.item_up1),
            item_up2: pk(&p.item_up2),
            item_out: pk(&p.item_out),
            rat_up1: pk(&p.rat_up1),
            rat_up2: pk(&p.rat_up2),
            rat_out: pk(&p.rat_out),
            enc_w: Mat { k: p.enc1.w.k, n: p.enc1.w.n, w: p.enc1.w.w.clone() },
            enc_b: p.enc1.b.clone(),
            pool: Pool::new(nthreads, pin),
            cfg,
        }
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
        self.gemm2(&z, BOTTLENECK, rows, &self.item_up1, &mut d1, &self.rat_up1, &mut d2, DEC_MID, true);

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

        ForwardOut { logits, ratings, rows }
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
