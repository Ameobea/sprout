# Eval harnesses

Two frozen-fixture evals for comparing model versions, data pulls, filtering schemes,
normalization, and input layouts. Preprocessing is shared with the python model server
via `notebooks/profile_preprocessing.py` (the Rust server mirrors it — keep in sync).

**Read `WORKSTREAMS.md` §0 before optimizing against these.** The product goal is
surfacing hidden gems; both evals reward predicting what a user *did* engage with, so
they undervalue that goal. Use them as guardrails, not as the objective.

- `eval_harness.py` — deterministic leave-one-out over every in-corpus item of every
  fixture profile (zero RNG). Reports rating MAE, recall@{10,50,100}, median rank, per
  bucket and overall. ~40 min on CPU.
- `temporal_eval.py` — future-watch eval: input is the profile as of 2025-06-24,
  targets are items watched after. Includes a global-popularity baseline and sweeps the
  ranking blend (`--logit-weights`) without re-running the model. ~3 min on CPU.
- Both take `--input-channels {2,3,5}`, which **must match how the model was trained**.
- `fixtures/` — frozen, never regenerate; comparability across all reports depends on it.
  `reports/` — one JSON per run.

## Fixtures

| file | what |
|---|---|
| `ameo___.json`, `snapsauce.json` | sentinel profiles, MAL snapshots 2026-07-30 |
| `sampled_profiles.json` | v1: 160 Dec-2025 profiles stratified by rating count (seed 1234) |
| `sampled_profiles_v2.json` | v2: 300 profiles, 4 size buckets × {long, mid, burst} history class, 25/cell (seed 20260730) |
| `temporal_v3.json` | 300 profiles split at a 2025-06-24 cutoff into inputs + future targets (seed 33) |

v1 exists because it predates v2; harness reports `overall_v1`, `overall_v2` and
combined so older reports stay comparable. v2 was added because v1's 10–29 bucket is
32.6% burst profiles, which confounded filtering experiments.

## Noise floors — read before believing any delta

From identical-config retrains: **LOO overall MAE ~0.003**, per 25-profile bucket
~0.02. The temporal eval at n=300 is noisier — two identical statuschan runs differed
by 0.005 recall@10 and 29 median-rank positions. **Confirm every winner with a second
training run.**

## Baselines (LOO, v1+v2 combined, n=454)

| model | MAE | recall@50 | median rank |
|---|---|---|---|
| dec2025 deployed (`dec2025-baseline-fixv2`) | 0.4364 | 0.596 | 55 |
| replication, same data + code (`replication-fixv2`) | 0.4337 | 0.597 | 54 |
| best: statuschan × cleanup_notrust ×2 | 0.4230 / 0.4234 | 0.600 / 0.599 | 53 |

v1-only slice of the deployed model (comparable to pre-2026-07-31 reports): MAE 0.4118,
recall@50 0.590, median rank 59; buckets 10-29 0.4631, 30-99 0.3936, 100-299 0.3994,
300+ 0.4023; ameo___ 0.4514, snapsauce 0.4549.

## Baselines (temporal, lw=0.3)

| model | recall@10 | recall@50 | median target rank | MRR |
|---|---|---|---|---|
| global popularity | 0.040 | 0.124 | 673 | 0.238 |
| dec2025 deployed | 0.091 | 0.243 | 366 | 0.405 |
| best: statuschan × cleanup_notrust | 0.095 | 0.243 | 342 | 0.432 |

Context for reading these: the median fixture user has 16 future targets, so the
arithmetic ceiling is recall@10 ≈ 0.616 and recall@50 ≈ 0.951 — the model reaches ~15%
and ~26% of achievable. Only 5.2% of future watches fall outside the 6000-item corpus.
The blend sweep (0.1–0.9) peaks at 0.3, confirming the production value; presence-
dominated ranking (0.9) is uniformly worse than rating-dominated.

## Notes

- The in-training debug-profile MAE printed by `train.py` uses a drifted ad-hoc
  protocol and does not match these numbers. Trust the harness.
- Cross-version comparisons where corpora differ need `--restrict-corpus` pointed at
  the other model's `corpus_ids.json` (scores both on the shared item set only).
