# Deterministic eval harness

Frozen-fixture, leave-one-out eval for comparing model versions, training-data pulls,
filtering schemes, and normalization changes. See `eval_harness.py` docstring for usage;
preprocessing is shared with the python model server via `notebooks/profile_preprocessing.py`.

- `fixtures/` — frozen profiles: `ameo___` + `snapsauce` sentinels (MAL snapshots 2026-07-30)
  plus 160 December-2025 profiles stratified by rating count (seed 1234). Do not regenerate
  these without versioning; comparability depends on them staying fixed.
- `reports/` — one JSON per eval run.

## Baseline: dec2025 deployed weights (`dec2025-baseline.json`)

| slice | MAE | recall@50 | median rank |
|---|---|---|---|
| overall (n=154) | 0.4118 | 0.590 | 59 |
| 10-29 ratings | 0.4631 | 0.765 | 23 |
| 30-99 | 0.3936 | 0.699 | 24 |
| 100-299 | 0.3994 | 0.576 | 42 |
| 300+ | 0.4023 | 0.363 | 136 |
| ameo___ | 0.4514 | 0.596 | 31 |
| snapsauce | 0.4549 | 0.276 | 156 |

Cross-check: December's in-training eval (different protocol: 200 training-set profiles,
50-300 ratings) recorded MAE 0.3959 — consistent with the 30-299 buckets here.

## Replication check (2026-07-30, `replication-decdata-2026-07-30.json`)

Retrained from scratch with current train.py/model.py on the December vectors:
overall MAE 0.4092 / recall@50 0.594 / median rank 58 vs baseline 0.4118 / 0.590 / 59 —
all buckets within run-to-run noise. Current training code reproduces the deployed model.
(The in-training debug-profile MAE printed by train.py runs ~0.57 vs 0.47 in tricks.txt —
protocol drift in that ad-hoc eval; disregard it, trust the harness.)
