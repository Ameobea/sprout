# logQ presence prior — decision record & migration reference

**Status: LOCKED (Casey, 2026-08-01).** The next model iteration trains with the
logQ popularity prior, and the prod `niche_boost` algorithm is torn out and replaced
by the α_add serving knob, powered by the existing `niche_boost_factor` request
param through a linearizing remap. This doc is the canonical reference for the
design, the evidence, and the migration.

Evidence detail: `archive/experiment-log-dec2025-era.md` ("Presence-head
investigation") · report JSONs in `notebooks/eval/reports/archive-dec2025/` ·
dashboard https://claude.ai/code/artifact/e9ff6c06-f6cb-4c77-9f7f-d122d916ab18

## The mechanism

**Training** (`train.py --presence-prior-alpha 1.0`): the presence loss becomes
`log_softmax(item_logits + log_pop_frac)` where `log_pop_frac` is a *fixed* vector,
`ln(count_i / Σcount)` with counts clipped ≥1, computed from the training npz itself
(train.py does this automatically — counts must come from the same vectors the model
trains on). The softmax still fits the true watch distribution, but the fixed offset
absorbs global popularity, so `item_logits` learn **lift over popularity** — per-user
taste deviation. Rating head, architecture, and all other hyperparams unchanged.

**Serving**: rank with `item_logits + α_add · log_pop` (any constant shift cancels
in softmax, so unnormalized `ln(count)` is fine).

- `α_add = 1.0` → standard mode: reproduces the old model's behavior (verified —
  LOO ranking/MAE at baseline, temporal parity-to-better).
- `α_add < 1.0` → progressively niche. The useful product range is **[0.5, 1.0]**;
  below ~0.3 franchise fragments (specials/recaps of watched shows) crowd the list.
- `α_add = 0` (raw lift) is a diagnostic mode, never a product mode.

## Why (one paragraph of evidence)

Every dec2025-era model's softmax correlated with global popularity at ρ≈0.65;
45% of users' actual future watches sit at popularity rank 1k+ where recall@50 was
0.01–0.06; tail items were near-chance NLL even when present in the model's own
input. ~1.2 of the ~2.6 nats every model "learned" was the popularity prior itself —
which is why presence loss (6.05–6.20) never responded to filters, channels, or 2×
params. The logq model, confirmed over two runs with both LOO guardrails green,
dominates post-hoc reranking of the old model at every matched overall-recall point
(+20–35% relative deep-tail recall) and strictly dominates the old `niche_boost`
formula everywhere. Under logq the tier-NLL gradient *inverts*: niche items become
the best-encoded, because they carry the taste signal.

## Serving-side migration (tear out niche_boost)

Rust (`model-server-rs`):
- `recommend.rs`: delete the `boost_active` / `effective_boost` / expanded-`top_k`
  path. Replace with: add `α_add · log_pop` to the logits row before
  `softmax_stats`/scoring (one SIMD axpy over 6000 floats, ~free; candidates:
  `post.rs` alongside `ranking_scores`).
- Popularity vector: ship the **training-set counts** the model was trained with as
  a model artifact (e.g. `log_pop.npy` next to the weights; loaded into `ModelData`).
  `md.popularity` (metadata popularity, used by old niche_boost) is NOT the same
  vector — the prior must byte-match training or serve/train skew is reintroduced.
- Request API: keep `niche_boost_factor: f32` (0 = off) for compatibility; map it
  through the linearization remap to α_add (0 → 1.0, max → the floor value). Old
  clients keep working; semantics improve silently.
- Regenerate golden tests against the new weights + serve path.

TS (`recommendation.ts`):
- Remove `effective_boost`-era plumbing; the knob UI value passes through unchanged
  (the remap lives server-side so all clients share it).
- The extra-season/related filters stay — they're the main mitigation for franchise
  fragments at low α_add and become *more* load-bearing after this change.

Python (`model.py` / harnesses): serve-mode eval via
`eval_harness.py --serve-prior-alpha`; frontier via `pop_correction_sweep.py`
(negative α = re-add). `compute_recommendation_ranking_score` itself is unchanged —
the prior-add happens on logits before it.

## Serving family: (α, k), exposed as ONE user knob (Casey, 2026-08-01)

The serving family gained a second parameter (validated 2026-08-01 on both logq
checkpoints, `alpha-k-{logq-run1,logq-run2}.json` + explorer artifact
https://claude.ai/code/artifact/3f9c7464-4ec5-4c19-9b84-216c1374903b):

    serve_logits = lam * lift + alpha * log_pop,   lam_i = count_i/(count_i + k)

**k is an evidence bar** — appearances needed for a show's lift to count at half
volume. It mutes thin-evidence lift spikes (the franchise-fragment/ONA failure
mode) without touching well-evidenced shows. Measured effects, reproduced on both
runs: moderate k *raises* overall AND mid-band (rank 50–3k) recall above today's
serving (e.g. α=0.4, k=30k: 0.242/0.248 overall, 0.181/0.185 mid vs 0.237/0.238,
0.172/0.172), restores the MRR that low α costs, and zeroes deep-tail (3k+) recall
— those hits ride on thin-evidence lift; accepted trade per the product judgment
that the 50–3k band is where recommendation value lives. Also measured: today's
serving is *more head-concentrated than the audience's actual watch distribution*
(TVD 0.184; closest alignment near α=0.6, k≈10k).

**Product design: a single unified niche-boost slider (Casey, 2026-08-01/02).**
α and k are NOT exposed separately; the slider t ∈ [0, 1] traces a path through
(α, k) space. Important distinction learned here: the *metric-optimal* ridge has k
rising as α falls (k=30k at α≤0.4), but that path quietly re-mainstreams the list
(α=0.3/k=30k → mean top-10 pop rank ≈ today's) — metric-optimal ≠ product-right.
The slider must be monotone in *felt nicheness*, so k stays a light junk-guard on
the niche side. Working anchors (validated on both checkpoints, 10×10 grid in
`alpha-k-logq-run{1,2}.json`):

| t | (α, k) | behavior |
|---|---|---|
| 0 | (1.0, ~10k) | clean mainstream — nichest lists OFF, best MRR (0.415–0.432), overall = today |
| ~0.35 **(new default — slider no longer defaults to 0)** | (0.7, ~2750) | overall+mid recall ABOVE today on both runs (0.239–0.240 / 0.176), nicher lists (mean top-10 rank ~600 vs ~420–450), deep tail dimmed not dead |
| 1 | (0.25, ~2250) | full niche (mean rank ~1370–1394, deep-tail recall 0.05–0.07), overall −0.015–0.020 — accepted; at max boost, niche slant IS the product intent |

k > 7.5k is out of the niche-side range entirely (only the t=0 endpoint uses ~10k).
α floor 0.25: below α≈0.3 top-10s become hypersensitive to small α changes
(lift-dominated regime).

**Status 2026-08-02: baseline path accepted as the starting point; further tuning
DEFERRED** to the fresh-data model + live-site experimentation (Casey). The working
path (α piecewise-linear, k log-interpolated through the anchors) is encoded in the
explorer's `PATH_ANCHORS` and behaves well: niche-ness ≈ linear in t, default sits
on the overall-recall plateau. Known knob-space wrinkle for the eventual re-fit:
k's motion is crammed into t∈[0, 0.35] while the deep-tail response lives in
t∈[0.5, 1] — stretching the k descent rightward would align them. Also deliberately
NOT pushed nicher than α=0.25: raw-model top-10s at high niche are dominated by
franchise fragments that prod's extra-season filter would remove, so offline
eyeballs can't judge that region (see backlog: filter-aware eval lists). Re-fit on
fresh data with `alpha_k_sweep.py`; ship as a lookup behind `niche_boost_factor`.
Explorer: https://claude.ai/code/artifact/3f9c7464-4ec5-4c19-9b84-216c1374903b

## Risks / gotchas carried forward

- **Prior/serve consistency**: the serve-time `log_pop` must be the training-set
  counts of the deployed model's training run. Recompute per data pull; version it
  with the weights.
- **Zero-count items** (61 in the dec2025 corpus): unlearnable by CF, logits stay
  near init. Never surfaced in any eyeball/eval top-50 during validation, but
  masking them at serve is cheap insurance if they ever do. Long-term fix is
  content features (backlog).
- **Recency ≈ niche**: static counts make newly-airing shows read as high-lift
  (e.g. "City The Animation" topped lift lists). Acceptable — arguably desirable —
  and self-correcting with fresh data pulls.
- **LOO ranking on raw lift reads low by construction** — always evaluate
  serve-mode (`--serve-prior-alpha 1.0`) for guardrail comparisons.
- **bf16 gate**: before deploy, run the bf16-simulated eval (backlog item) to check
  the win survives prod quantization.

## Interactions to re-test (see WORKSTREAMS §3)

The prior changes what the encoder/decoder spend capacity on, so dec2025-era
conclusions about *other* axes may shift — notably the statuschan/maskchan
channels (their MAE-vs-recall trade may look different when the presence head isn't
relearning popularity) and, less likely, the filter set. Dropout 0.4 and lw 0.3 are
assumed transferable until evidence says otherwise.
