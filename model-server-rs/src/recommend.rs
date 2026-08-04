//! Request-level logic mirroring notebooks/model_server.py.

use crate::engine::{Engine, HoldoutDelta};
use crate::norm::{normalize_ratings, NormStats};
use crate::post::topk_excluding;
use crate::simd::{alt_score_into, combined_score_into, row_scores_gathered, softmax_into, softmax_stats};
use crate::{CORPUS, DEFAULT_LOGIT_WEIGHT};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Deserialize, Serialize, Clone, PartialEq)]
pub struct ProfileEntry {
    pub anime_id: i64,
    #[serde(default)]
    pub rating: f32,
    #[serde(default)]
    pub watch_status: String,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct RecommendRequest {
    pub profile: Vec<ProfileEntry>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default = "d_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub logit_weight: Option<f32>,
    #[serde(default)]
    pub include_profile_holdout: bool,
    #[serde(default)]
    pub include_contribution_analysis: bool,
    #[serde(default = "d_top_contributors")]
    pub top_contributors: usize,
    #[serde(default)]
    pub use_alt_ranking: bool,
    #[serde(default)]
    pub niche_boost_factor: f32,
    #[serde(default)]
    pub include_raw_logits: bool,
    /// Dev flag, graft models only: blend w of z-normed EASE-lift into the z-normed
    /// presence lift before the (α,k) transform. Applies to the base row only —
    /// holdout analysis stays unstacked.
    #[serde(default)]
    pub stack_weight: Option<f32>,
}

fn d_top_k() -> usize {
    50
}
fn d_top_contributors() -> usize {
    3
}

#[derive(Serialize, Clone)]
pub struct Contributor {
    pub anime_id: i64,
    pub corpus_idx: u32,
    pub score_contribution: f32,
}

#[derive(Serialize, Clone)]
pub struct Recommendation {
    pub anime_id: i64,
    pub corpus_idx: u32,
    pub score: f32,
    pub probability: f32,
    pub predicted_rating: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_logit: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_contributors: Option<Vec<Contributor>>,
}

#[derive(Serialize)]
pub struct HoldoutItem {
    pub anime_id: i64,
    pub corpus_idx: u32,
    pub true_rating: f32,
    pub true_normalized_rating: f32,
    pub predicted_rating: f32,
    pub rating_error: f32,
    pub presence_probability: f32,
    pub recommendation_score: f32,
    pub impact_score: f32,
}

#[derive(Serialize)]
pub struct ProfileHoldout {
    pub items: Vec<HoldoutItem>,
    pub mean_rating_error: f32,
    pub std_rating_error: f32,
    pub mean_presence_prob: f32,
    pub std_presence_prob: f32,
}

#[derive(Serialize)]
pub struct NormStatsOut {
    pub mu: f32,
    pub sigma: f32,
    pub alpha: f32,
    pub zscore_norm: Vec<f32>,
    pub absolute_norm: Vec<f32>,
}

#[derive(Serialize)]
pub struct RecommendResponse {
    pub recommendations: Vec<Recommendation>,
    pub profile_holdout: Option<ProfileHoldout>,
    pub normalization_stats: NormStatsOut,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ServingFamily {
    /// dec2025-era: raw logits ranked, optional post-hoc surprise boost.
    Legacy,
    /// lift-trained model: serve_logits = lam·lift + α·log_pop, (α,k) from the knob remap.
    Logq,
}

pub struct ModelData {
    pub name: &'static str,
    pub serving: ServingFamily,
    pub engine: Engine,
    pub corpus_ids: Vec<i64>,
    pub id_to_idx: HashMap<i64, u32>,
    /// Metadata-derived popularity distribution; legacy niche_boost only.
    pub popularity: Option<Vec<f32>>,
    /// Training-set item counts (must byte-match the deployed weights' training data).
    pub train_counts: Option<Vec<f32>>,
    /// ln(max(count, 1)) over train_counts.
    pub log_pop: Option<Vec<f32>>,
    /// Graft models: per-item mean full-profile EASE score over reference users (for stack scoring).
    pub ease_mu: Option<Vec<f32>>,
}

/// Niche slider t → (α, k): α piecewise-linear, k log-interpolated through the
/// product anchors locked in logq-presence-prior.md. t=0 is clean-mainstream
/// (α=1 reproduces the unboosted model), t=1 is full niche (α floor 0.25).
const PATH_ANCHORS: [(f32, f32, f32); 3] = [(0.0, 1.0, 10_000.0), (0.35, 0.7, 2_750.0), (1.0, 0.25, 2_250.0)];

pub fn alpha_k(t: f32) -> (f32, f32) {
    let t = t.clamp(0.0, 1.0);
    let (lo, hi) = if t <= PATH_ANCHORS[1].0 {
        (PATH_ANCHORS[0], PATH_ANCHORS[1])
    } else {
        (PATH_ANCHORS[1], PATH_ANCHORS[2])
    };
    let f = (t - lo.0) / (hi.0 - lo.0);
    let alpha = lo.1 + f * (hi.1 - lo.1);
    let k = (lo.2.ln() + f * (hi.2.ln() - lo.2.ln())).exp();
    (alpha, k)
}

fn logq_transform(lift: &[f32], counts: &[f32], log_pop: &[f32], alpha: f32, k: f32, out: &mut [f32]) {
    for i in 0..lift.len() {
        let lam = counts[i] / (counts[i] + k);
        out[i] = lam * lift[i] + alpha * log_pop[i];
    }
}

fn znorm_into(x: &[f32], out: &mut [f32]) {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x.iter().map(|&v| (v as f64 - mean) * (v as f64 - mean)).sum::<f64>() / n;
    let inv = (1.0 / (var.sqrt() + 1e-9)) as f32;
    let m = mean as f32;
    for (o, &v) in out.iter_mut().zip(x) {
        *o = (v - m) * inv;
    }
}

/// (1-w)·z(lift_NN) + w·z(e - mu). NOTE: z-normed units — the (α,k) anchors were fit
/// on raw lift scale, so knob feel shifts under stacking; dev-flag territory until the
/// remap re-anchor (decision artifact §7 phase 2).
fn stack_lift(raw_logits: &[f32], e: &[f32], mu: &[f32], w: f32, out: &mut [f32]) {
    let mut zl = vec![0.0f32; raw_logits.len()];
    znorm_into(raw_logits, &mut zl);
    let lift_e: Vec<f32> = e.iter().zip(mu).map(|(&ev, &mv)| ev - mv).collect();
    let mut ze = vec![0.0f32; lift_e.len()];
    znorm_into(&lift_e, &mut ze);
    for i in 0..out.len() {
        out[i] = (1.0 - w) * zl[i] + w * ze[i];
    }
}

pub struct Prepped {
    pub anime_ids: Vec<i64>,
    pub corpus_indices: Vec<u32>,
    pub original: Vec<f32>,
    pub normalized: Vec<f32>,
    pub stats: NormStats,
    pub enc_items: Vec<(u32, f32)>,
    pub deltas: Vec<HoldoutDelta>,
    pub rated_mask: Vec<bool>,
}

pub fn preprocess(md: &ModelData, profile: &[ProfileEntry]) -> Result<Prepped, String> {
    let mut valid: Vec<(u32, i64, f32, bool)> = profile
        .iter()
        .filter_map(|e| {
            let &idx = md.id_to_idx.get(&e.anime_id)?;
            let dropped = e.watch_status == "dropped";
            if e.rating > 0.0 || matches!(e.watch_status.as_str(), "completed" | "watching" | "dropped") {
                Some((idx, e.anime_id, e.rating, dropped))
            } else {
                None
            }
        })
        .collect();
    if valid.is_empty() {
        return Err("No valid entries in user profile".into());
    }
    valid.sort_by_key(|v| v.0);

    let corpus_indices: Vec<u32> = valid.iter().map(|v| v.0).collect();
    let anime_ids: Vec<i64> = valid.iter().map(|v| v.1).collect();
    let original: Vec<f32> = valid
        .iter()
        .map(|&(_, _, rating, dropped)| if dropped && rating == 0.0 { -2.0 } else { rating })
        .collect();
    let (normalized, stats) = normalize_ratings(&original);

    // Dense-input set semantics: presence set once per index; last write wins for values.
    let mut enc_items: Vec<(u32, f32)> = Vec::with_capacity(valid.len());
    for (i, &idx) in corpus_indices.iter().enumerate() {
        match enc_items.last_mut() {
            Some(last) if last.0 == idx => last.1 = normalized[i],
            _ => enc_items.push((idx, normalized[i])),
        }
    }

    // Holdout of entry i: presence drops only if no duplicate index remains; the
    // rating input shifts to the last remaining duplicate's value.
    let n = valid.len();
    let deltas: Vec<HoldoutDelta> = (0..n)
        .map(|i| {
            let idx = corpus_indices[i];
            let dup_before = (0..i).rev().take_while(|&j| corpus_indices[j] == idx).count();
            let dup_after = (i + 1..n).take_while(|&j| corpus_indices[j] == idx).count();
            let full_val = enc_items.iter().find(|e| e.0 == idx).unwrap().1;
            if dup_after > 0 {
                HoldoutDelta { idx, presence_removed: false, dval: 0.0 }
            } else if dup_before > 0 {
                HoldoutDelta { idx, presence_removed: false, dval: full_val - normalized[i - 1] }
            } else {
                HoldoutDelta { idx, presence_removed: true, dval: full_val }
            }
        })
        .collect();

    let mut rated_mask = vec![false; CORPUS];
    for &(idx, _) in &enc_items {
        rated_mask[idx as usize] = true;
    }

    Ok(Prepped { anime_ids, corpus_indices, original, normalized, stats, enc_items, deltas, rated_mask })
}

fn effective_boost(f: f32) -> f32 {
    let f = f.clamp(0.0, 1.0);
    if f <= 0.5 { f } else { 0.5 + (f - 0.5) * (4.62 * (f - 0.5)).exp() }
}

pub fn run_inference(md: &ModelData, prep: &Prepped, req: &RecommendRequest) -> (Vec<Recommendation>, Option<ProfileHoldout>) {
    let w = req.logit_weight.unwrap_or(DEFAULT_LOGIT_WEIGHT);
    let alt = req.use_alt_ranking;
    let logq = md.serving == ServingFamily::Logq && md.train_counts.is_some();
    let (lq_alpha, lq_k) = if logq { alpha_k(req.niche_boost_factor) } else { (1.0, 0.0) };
    let boost_active = !logq && req.niche_boost_factor > 0.0 && md.popularity.is_some();
    let expanded_k = if boost_active { (req.top_k * 3).min(500) } else { req.top_k };
    let need_holdout = req.include_profile_holdout || req.include_contribution_analysis;

    let out = md.engine.forward(&prep.enc_items, need_holdout.then_some(&prep.deltas[..]));
    let base_row = out.rows - 1;
    let raw_logits = out.logits_row(base_row);
    let base_ratings = out.ratings_row(base_row);

    let mut stack_buf = vec![0.0f32; CORPUS];
    let lift: &[f32] = match (req.stack_weight, out.ease_full_raw.as_ref(), md.ease_mu.as_ref()) {
        (Some(w), Some(e), Some(mu)) if w > 0.0 => {
            stack_lift(raw_logits, e, mu, w.clamp(0.0, 1.0), &mut stack_buf);
            &stack_buf
        }
        _ => raw_logits,
    };
    let mut serve_buf = vec![0.0f32; CORPUS];
    let base_logits: &[f32] = if logq {
        logq_transform(
            lift,
            md.train_counts.as_ref().unwrap(),
            md.log_pop.as_ref().unwrap(),
            lq_alpha,
            lq_k,
            &mut serve_buf,
        );
        &serve_buf
    } else {
        lift
    };

    // Full score/prob rows for the baseline
    let mut base_probs = vec![0.0f32; CORPUS];
    let (bmax, bsum) = softmax_stats(base_logits);
    softmax_into(base_logits, bmax, bsum, &mut base_probs);
    let mut base_scores = vec![0.0f32; CORPUS];
    if alt {
        alt_score_into(base_logits, base_ratings, w, &mut base_scores);
    } else {
        combined_score_into(&base_probs, base_ratings, w, &mut base_scores);
    }

    let topk = topk_excluding(&base_scores, &prep.rated_mask, expanded_k.min(CORPUS));
    let mut recommendations: Vec<Recommendation> = topk
        .iter()
        .map(|&ci| Recommendation {
            anime_id: md.corpus_ids[ci as usize],
            corpus_idx: ci,
            score: base_scores[ci as usize],
            probability: base_probs[ci as usize],
            predicted_rating: base_ratings[ci as usize],
            raw_logit: req.include_raw_logits.then(|| raw_logits[ci as usize]),
            top_contributors: None,
        })
        .collect();

    if boost_active {
        let pop = md.popularity.as_ref().unwrap();
        let eff = effective_boost(req.niche_boost_factor);
        for rec in &mut recommendations {
            let surprise = rec.probability / (pop[rec.corpus_idx as usize] + 1e-9);
            rec.score *= 1.0 + eff * (1.0 + surprise).ln();
        }
        recommendations.sort_by(|a, b| b.score.total_cmp(&a.score));
    }
    recommendations.truncate(req.top_k);

    let mut profile_holdout = None;
    if need_holdout {
        let n = prep.deltas.len();
        let rec_idxs: Vec<u32> = recommendations.iter().map(|r| r.corpus_idx).collect();
        let num_impact = 50.min(recommendations.len());

        // Gather layout per holdout row: [held_idx, top50-for-impact..., recs-for-contrib...]
        let mut gather: Vec<u32> = Vec::with_capacity(1 + num_impact + rec_idxs.len());
        let impact_base = 1;
        gather.push(0); // placeholder for held idx, set per row
        gather.extend_from_slice(&rec_idxs[..num_impact]);
        let contrib_base = gather.len();
        if req.include_contribution_analysis {
            gather.extend_from_slice(&rec_idxs);
        }

        struct RowRes {
            pred_rating: f32,
            presence_prob: f32,
            rec_score: f32,
            impact: f32,
            contrib_scores: Vec<f32>,
        }
        let impact_baseline: Vec<f32> = recommendations[..num_impact].iter().map(|r| r.score).collect();
        let raw_baseline_at_recs: Vec<f32> = rec_idxs.iter().map(|&ci| base_scores[ci as usize]).collect();

        let mut rows: Vec<Option<RowRes>> = (0..n).map(|_| None).collect();
        {
            let rows_addr = rows.as_mut_ptr() as usize;
            let gather_ref = &gather;
            let out_ref = &out;
            let prep_ref = &prep;
            let counts_ref = md.train_counts.as_deref();
            let log_pop_ref = md.log_pop.as_deref();
            md.engine.pool.run(&move |tid| {
                let nt = md.engine.pool.n_threads();
                let rows = unsafe { std::slice::from_raw_parts_mut(rows_addr as *mut Option<RowRes>, n) };
                let mut g = gather_ref.clone();
                let mut serve_row = vec![0.0f32; CORPUS];
                let mut r = tid;
                while r < n {
                    g[0] = prep_ref.corpus_indices[r];
                    let raw_row = out_ref.logits_row(r);
                    let logits: &[f32] = if logq {
                        logq_transform(raw_row, counts_ref.unwrap(), log_pop_ref.unwrap(), lq_alpha, lq_k, &mut serve_row);
                        &serve_row
                    } else {
                        raw_row
                    };
                    let ratings = out_ref.ratings_row(r);
                    // Python quirk kept for parity: compute_profile_holdout_analysis never
                    // forwards use_alt_ranking, so holdout scores/impact always use the
                    // default ranking; contributions honor the flag.
                    let (scores, probs) = row_scores_gathered(logits, ratings, &g, w, false);
                    let contrib_scores = if req.include_contribution_analysis {
                        if alt {
                            row_scores_gathered(logits, ratings, &g, w, true).0[contrib_base..].to_vec()
                        } else {
                            scores[contrib_base..].to_vec()
                        }
                    } else {
                        Vec::new()
                    };
                    let impact = (0..num_impact)
                        .map(|j| (impact_baseline[j] - scores[impact_base + j]).abs())
                        .sum();
                    rows[r] = Some(RowRes {
                        pred_rating: ratings[g[0] as usize],
                        presence_prob: probs[0],
                        rec_score: scores[0],
                        impact,
                        contrib_scores,
                    });
                    r += nt;
                }
            });
        }
        let rows: Vec<RowRes> = rows.into_iter().map(Option::unwrap).collect();

        if req.include_contribution_analysis {
            for (j, rec) in recommendations.iter_mut().enumerate() {
                let mut drops: Vec<(f32, usize)> = (0..n)
                    .map(|i| (raw_baseline_at_recs[j] - rows[i].contrib_scores[j], i))
                    .collect();
                drops.sort_by(|a, b| b.0.total_cmp(&a.0));
                rec.top_contributors = Some(
                    drops
                        .iter()
                        .take(req.top_contributors)
                        .filter(|(d, _)| *d > 0.0)
                        .map(|&(d, i)| Contributor {
                            anime_id: prep.anime_ids[i],
                            corpus_idx: prep.corpus_indices[i],
                            score_contribution: d,
                        })
                        .collect(),
                );
            }
        }

        if req.include_profile_holdout {
            let items: Vec<HoldoutItem> = (0..n)
                .map(|i| {
                    let rr = &rows[i];
                    HoldoutItem {
                        anime_id: prep.anime_ids[i],
                        corpus_idx: prep.corpus_indices[i],
                        true_rating: prep.original[i],
                        true_normalized_rating: prep.normalized[i],
                        predicted_rating: rr.pred_rating,
                        rating_error: (rr.pred_rating - prep.normalized[i]).abs(),
                        presence_probability: rr.presence_prob,
                        recommendation_score: rr.rec_score,
                        impact_score: rr.impact,
                    }
                })
                .collect();
            let mean = |xs: &[f32]| xs.iter().sum::<f32>() / xs.len() as f32;
            let std = |xs: &[f32], m: f32| {
                (xs.iter().map(|&x| (x - m) * (x - m)).sum::<f32>() / xs.len() as f32).sqrt()
            };
            let errs: Vec<f32> = items.iter().map(|it| it.rating_error).collect();
            let probs: Vec<f32> = items.iter().map(|it| it.presence_probability).collect();
            let (me, mp) = (mean(&errs), mean(&probs));
            profile_holdout = Some(ProfileHoldout {
                items,
                mean_rating_error: me,
                std_rating_error: std(&errs, me),
                mean_presence_prob: mp,
                std_presence_prob: std(&probs, mp),
            });
        }
    }

    (recommendations, profile_holdout)
}

pub fn norm_stats_out(stats: &NormStats) -> NormStatsOut {
    NormStatsOut {
        mu: stats.mu,
        sigma: stats.sigma,
        alpha: stats.alpha,
        zscore_norm: stats.zscore.clone(),
        absolute_norm: stats.absolute.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_k_anchors() {
        for &(t, a, k) in &PATH_ANCHORS {
            let (ra, rk) = alpha_k(t);
            assert!((ra - a).abs() < 1e-6 && (rk - k).abs() < 0.5, "anchor t={t}: got ({ra}, {rk})");
        }
        let (a0, _) = alpha_k(-1.0);
        let (a1, _) = alpha_k(2.0);
        assert_eq!(a0, 1.0);
        assert_eq!(a1, 0.25);
        // monotone in t on both segments
        let mut prev_a = f32::INFINITY;
        let mut prev_k = f32::INFINITY;
        for i in 0..=20 {
            let (a, k) = alpha_k(i as f32 / 20.0);
            assert!(a <= prev_a + 1e-6, "alpha not monotone at t={}", i as f32 / 20.0);
            assert!(k <= prev_k + 1e-3, "k not monotone at t={}", i as f32 / 20.0);
            prev_a = a;
            prev_k = k;
        }
    }

    #[test]
    fn logq_transform_math() {
        let lift = [2.0f32, -1.0, 0.5];
        let counts = [30_000.0f32, 100.0, 0.0];
        let log_pop: Vec<f32> = counts.iter().map(|&c| c.max(1.0).ln()).collect();
        let mut out = [0.0f32; 3];
        logq_transform(&lift, &counts, &log_pop, 0.7, 2750.0, &mut out);
        for i in 0..3 {
            let lam = counts[i] / (counts[i] + 2750.0);
            assert!((out[i] - (lam * lift[i] + 0.7 * log_pop[i])).abs() < 1e-6);
        }
        // zero-count item: no lift contribution, log_pop clamped to ln(1)=0
        assert_eq!(out[2], 0.0);
        // alpha=1, huge count ≈ raw logits + log_pop (standard serving)
        logq_transform(&lift, &[1e9, 1e9, 1e9], &log_pop, 1.0, 2750.0, &mut out);
        assert!((out[0] - (lift[0] + log_pop[0])).abs() < 0.02);
    }
}
