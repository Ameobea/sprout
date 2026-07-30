#[derive(Clone, Debug)]
pub struct NormStats {
    pub mu: f32,
    pub sigma: f32,
    pub alpha: f32,
    pub zscore: Vec<f32>,
    pub absolute: Vec<f32>,
}

fn mean(xs: &[f32]) -> f32 {
    (xs.iter().map(|&x| x as f64).sum::<f64>() / xs.len() as f64) as f32
}

fn pop_std(xs: &[f32], mu: f32) -> f32 {
    let var = xs.iter().map(|&x| ((x - mu) as f64).powi(2)).sum::<f64>() / xs.len() as f64;
    var.sqrt() as f32
}

/// Port of notebooks/normalize_ratings.py. 0 = unrated, -2 = unrated dropped.
pub fn normalize_ratings(scores_in: &[f32]) -> (Vec<f32>, NormStats) {
    let n = scores_in.len();
    if n <= 1 {
        return (
            vec![0.0; n],
            NormStats { mu: 0.0, sigma: 0.0, alpha: 0.0, zscore: vec![0.0; n], absolute: vec![0.0; n] },
        );
    }

    let mut scores = scores_in.to_vec();
    let rated: Vec<f32> = scores.iter().copied().filter(|&s| s > 0.0).collect();
    let (mu, sigma) = if rated.is_empty() {
        (5.0, 2.0)
    } else {
        let mu = mean(&rated);
        (mu, pop_std(&rated, mu) + 1e-6)
    };
    for s in &mut scores {
        if *s == 0.0 {
            *s = mu;
        }
    }
    for s in &mut scores {
        if *s == -2.0 {
            *s = mu - 1.5 * sigma;
        }
    }

    let zscore: Vec<f32> = scores.iter().map(|&s| ((s - mu) / sigma).clamp(-3.0, 3.0)).collect();
    let absolute: Vec<f32> = scores.iter().map(|&s| ((s - 5.5) / 2.5).clamp(-2.5, 2.0)).collect();
    let alpha = (sigma / 2.6).clamp(0.3, 0.8);

    let normed: Vec<f32> = zscore
        .iter()
        .zip(&absolute)
        .map(|(&z, &a)| (alpha * z + (1.0 - alpha) * a).clamp(-2.5, 2.5))
        .collect();

    (normed, NormStats { mu, sigma, alpha, zscore, absolute })
}
