//! Scalar reference implementation used for correctness testing of the optimized kernels.

use crate::weights::{Layer, Params};
use crate::{CORPUS, IN_DIM};

fn swish(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn dense(layer: &Layer, x: &[f32], out: &mut [f32], activate: bool) {
    let Layer { w, b } = layer;
    out.copy_from_slice(b);
    for (i, &xi) in x.iter().enumerate() {
        if xi == 0.0 {
            continue;
        }
        let row = &w.w[i * w.n..(i + 1) * w.n];
        for (o, &wv) in out.iter_mut().zip(row) {
            *o += xi * wv;
        }
    }
    if activate {
        for o in out.iter_mut() {
            *o = swish(*o);
        }
    }
}

pub fn make_dense_profile(idxs: &[u32], vals: &[f32]) -> Vec<f32> {
    let mut x = vec![0.0f32; IN_DIM];
    for (&i, &v) in idxs.iter().zip(vals) {
        x[i as usize] = 1.0;
        x[CORPUS + i as usize] = v;
    }
    x
}

pub fn forward_ref(p: &Params, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut h = vec![0.0; p.enc1.w.n];
    dense(&p.enc1, x, &mut h, true);
    let mut z = vec![0.0; p.bott.w.n];
    dense(&p.bott, &h, &mut z, false);

    let mut d1 = vec![0.0; p.item_up1.w.n];
    dense(&p.item_up1, &z, &mut d1, true);
    let mut d1b = vec![0.0; p.item_up2.w.n];
    dense(&p.item_up2, &d1, &mut d1b, true);
    let mut logits = vec![0.0; p.item_out.w.n];
    dense(&p.item_out, &d1b, &mut logits, false);

    let mut d2 = vec![0.0; p.rat_up1.w.n];
    dense(&p.rat_up1, &z, &mut d2, true);
    let mut d2b = vec![0.0; p.rat_up2.w.n];
    dense(&p.rat_up2, &d2, &mut d2b, true);
    let mut ratings = vec![0.0; p.rat_out.w.n];
    dense(&p.rat_out, &d2b, &mut ratings, false);

    (logits, ratings)
}
