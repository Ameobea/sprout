pub fn softmax(xs: &[f32], out: &mut [f32]) {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (o, &x) in out.iter_mut().zip(xs) {
        let e = (x - max).exp();
        *o = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for o in out.iter_mut() {
        *o *= inv;
    }
}

fn mean_std(xs: &[f32]) -> (f32, f32) {
    let n = xs.len() as f64;
    let mu = xs.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = xs.iter().map(|&x| (x as f64 - mu).powi(2)).sum::<f64>() / n;
    (mu as f32, var.sqrt() as f32)
}

/// Combined recommendation score. Returns (scores, probs).
pub fn ranking_scores(logits: &[f32], ratings: &[f32], logit_weight: f32, alt: bool) -> (Vec<f32>, Vec<f32>) {
    let mut probs = vec![0.0; logits.len()];
    softmax(logits, &mut probs);

    let scores = if alt {
        let (pm, ps) = mean_std(logits);
        let (rm, rs) = mean_std(ratings);
        let (ps, rs) = (ps + 1e-6, rs + 1e-6);
        logits
            .iter()
            .zip(ratings)
            .map(|(&l, &r)| logit_weight * (l - pm) / ps + (1.0 - logit_weight) * (r - rm) / rs)
            .collect()
    } else {
        probs
            .iter()
            .zip(ratings)
            .map(|(&p, &r)| p.powf(logit_weight) * (r + 1.0).max(0.001).powf(1.0 - logit_weight))
            .collect()
    };
    (scores, probs)
}

/// Top-k by score, excluding already-rated items. Descending order.
pub fn topk_excluding(scores: &[f32], rated_mask: &[bool], k: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..scores.len() as u32).collect();
    let key = |i: u32| {
        if rated_mask[i as usize] { f32::NEG_INFINITY } else { scores[i as usize] }
    };
    let k = k.min(idx.len());
    idx.select_nth_unstable_by(k.saturating_sub(1), |&a, &b| key(b).total_cmp(&key(a)));
    idx.truncate(k);
    idx.sort_unstable_by(|&a, &b| key(b).total_cmp(&key(a)));
    idx
}
