# Data filtering experiments — log

Branch: `data-filtering-experiments`. Tracker artifact mirrors this; raw analysis in
`dec2025-data-quality-report.md`. All trainings on December 2025 data (same corpus as
deployed model → no --restrict-corpus needed for comparisons).

## Protocol

- Fixtures: v1 (frozen 2026-07-30, 162 profiles) + v2 (`sampled_profiles_v2.json`,
  seed 20260730, 300 profiles: 4 size buckets × {long, mid, burst} by rated_span,
  25/cell, items embedded from the December dump — static snapshots, immune to future
  pulls). v2 requires ≥10 rated + stage-1 pass. Harness reports overall / overall_v1 /
  overall_v2 + per-bucket.
- Noise floor: identical-config retrains differ ~0.003 MAE / ~0.005 recall@50
  (deployed vs replication). Winner needs a confirmation re-run.
- Variants (vectorize_variants.py, one pass produces all npz files):
  - `baseline`: replica of notebook logic — must exactly match original
    user_input_vectors.npz (1,489,525 users / 355,949,403 entries) before anything
    else is trusted.
  - `trustmask`: rated_flag=False (gradient masked, scores still in input channel)
    for one-sitting raters (rated_span≤7d, n_rated≥10) + degenerate raters
    (n_distinct==1 or mode_frac≥0.9, n_rated≥10). ~134.6k users flagged.
  - `harddrop`: flagged users dropped entirely.
  - `cleanup`: trustmask + drop >2000-entry lists (~10.4k users) + keep rated PTW
    (train/serve alignment) + gate ≥20 model-input entries (replaces ≥30 raw rows).
- Training: train.py parameterized (--vectors/--out/--steps), 50k steps, rocm_jax
  container, sequential runs.

## Vectorization (2026-07-30)

`baseline` reproduces the original npz EXACTLY (1,489,525 users / 355,949,403
entries) — replica verified, variants trusted.

| variant | users | entries | rated |
|---|---|---|---|
| baseline | 1,489,525 | 355,949,403 | 258,526,047 (72.6%) |
| trustmask | 1,489,525 | 355,949,403 | 248,344,682 (69.8%) — 3.9% of ratings masked |
| harddrop | 1,388,346 (−6.8%) | 339,947,450 (−4.5%) | 248,344,682 (73.1%) |
| cleanup | 1,486,176 | 341,712,762 (−4.0%) | 238,752,232 (69.9%) |

## Runs

| run | vectors | status | notes |
|---|---|---|---|
| trustmask | user_input_vectors_trustmask.npz | trained 2026-07-31, weights `data/jax_model_trustmask.msgpack`; eval running | 50k steps, log `data/train_trustmask.log` |
| harddrop | user_input_vectors_harddrop.npz | training (relaunched post-reboot 2026-07-31 21:20) | first launch died at init — container was killed pre-reboot |
| cleanup | user_input_vectors_cleanup.npz | queued next on GPU | |

## Resume checklist (post-reboot)

1. `just launch-jax` + container setup per notebooks/README_training.md (libdw1 +
   flax constraint install), repo at /jax_dir.
2. Verify `data/jax_model_harddrop.msgpack` exists and `train_harddrop.log` ends with
   "Model parameters saved" — if not, relaunch that training.
3. Evals still owed (CPU, in container):
   `JAX_PLATFORMS=cpu python eval_harness.py --weights ../../data/jax_model_trustmask.msgpack --corpus ../../data/corpus_ids.json --name trustmask-fixv2`
   and same for harddrop (`--name harddrop-fixv2`).
4. Launch cleanup training:
   `python train.py --vectors ../data/user_input_vectors_cleanup.npz --out ../data/jax_model_cleanup.msgpack`
5. Compare all reports vs `dec2025-baseline-fixv2` + `replication-fixv2`; winner gets
   a confirmation re-run (different seed run-to-run noise check).

## Results

### dec2025 deployed weights on v1+v2 (`dec2025-baseline-fixv2.json`)

overall_v1 0.4118/0.590/59 — exact match with the original dec2025-baseline report,
harness extension verified. overall_v2 0.4477/0.599/52.

v2 highlights (MAE / recall@50 / med rank): burst-10-29 0.5122/0.754/23 vs
long-10-29 0.4597/0.683/29 — small burst profiles' ratings are markedly harder to
predict; at 100-299 it flips (burst 0.4217 vs long 0.4391 — mainstream taste is
easier). mid-10-29 is the worst rating slice overall (0.5268).

### trustmask (`trustmask-fixv2.json`) — REGRESSION

overall 0.4416/0.598/55 vs replication 0.4337/0.597/54 → +0.008 MAE (2.6x noise
floor), ranking flat. Damage concentrated in burst cells (burst-300+ +0.039,
burst-100-299 +0.030 — the masked users' own eval slices) but v1 also +0.007.
Reading: those 10.2M "untrusted" ratings still carry usable signal — or
gradient-only masking creates input/target mismatch (scores remain in the input
channel but the loss never anchors them). harddrop disambiguates: if dropping the
users entirely is also negative → the users are net-informative; if neutral →
the masking mechanism itself is the problem.

### FULL COMPARISON (overall on v1+v2, n=454: MAE / recall@50 / median rank)

| model | MAE | recall@50 | med rank | verdict |
|---|---|---|---|---|
| dec2025 deployed | 0.4364 | 0.596 | 55 | baseline |
| replication | 0.4337 | 0.597 | 54 | baseline |
| trustmask | 0.4416 | 0.598 | 55 | regression (+0.008) |
| harddrop | 0.4388 | 0.595 | 55 | regression (+0.005) |
| cleanup | 0.4391 | 0.600 | 53 | confounded by trustmask |
| **cleanup_notrust** | **0.4359** | **0.599** | **53** | **MAE-neutral, ranking best — winner** |

### cleanup_notrust (`cleanup-notrust-fixv2.json`) — the keeper

+0.002 MAE vs replication (inside the ~0.003 noise floor); recall@50 0.599 and
median rank 53 — both cleanup runs (which share the gate/huge-drop/PTW parts)
independently show the ranking edge (0.599-0.600 / 53 vs 0.595-0.598 / 54-55
everywhere else). v1 10-29 bucket improved in both cleanup runs (0.4506 / 0.4515
vs 0.4601-0.4636 all others) → the ≥20-model-input gate genuinely helps small
profiles. burst-300+ hit its best MAE anywhere (0.3939). Structural wins on top:
rated-PTW train/serve alignment, 4% fewer entries (bot lists gone).

Confirmation re-run (`cleanup-notrust-run2-fixv2.json`): overall 0.4324/0.599/54 —
best MAE of all seven models. Two cleanup_notrust runs (0.4359, 0.4324) bracket the
replication baseline (0.4337) → MAE-neutral confirmed; the ranking edge held in
both (recall@50 0.599 vs 0.596-0.597 baselines). VERDICT CONFIRMED.

**Adopt for the fresh-data run: drop >2000-entry lists, keep rated PTW, gate on
≥20 model-input entries, NO trust-based rating filtering.**

### harddrop (`harddrop-fixv2.json`) — no benefit, mild regression

overall 0.4388/0.595/55 (+0.005 MAE vs replication, ~1.7x noise floor; ranking flat).
v1 0.4115 (within noise of 0.4092), v2 0.4514 vs 0.4451 (+0.006). Long-history
cells show NO improvement (long-100-299 −0.011, others +0.003..+0.007 — mixed,
within cell noise); burst cells degrade (+0.02..0.03) — dropping burst-like
training users hurts burst-like eval users, the population-shift effect in reverse.

**Axis verdict (trustmask + harddrop together):** one-sitting/degenerate raters'
ratings are net-informative; removing them harms slightly (harddrop +0.005) and
gradient-masking them harms more (trustmask +0.008, input/target mismatch adds
on top). Filtering this population does not help real users. Dead end as a filter;
their data stays.

Follow-up queued: `cleanup_notrust` (cleanup bundle minus trustmask) to deconfound
the cleanup run, since cleanup includes the now-known-harmful masking component.

### replication weights on v1+v2 (`replication-fixv2.json`)

overall_v1 0.4092/0.594/58 (matches prior report), overall_v2 0.4451/0.598/52,
overall 0.4337/0.597/54 (deployed: 0.4364/0.596/55). Per-cell deltas between these
two identical-config models reach ±0.015 MAE (n=25 cells) → treat cell-level variant
deltas < ~0.02 MAE as noise; overall-level floor ~0.003.

## Training-speed investigation (2026-07-30)

Measured on the live harddrop run: 26.8 ms/step steady-state (batch 512) → ~22 min
of stepping per 50k run. Host batch build measured at 6.0 ms/batch (vectorized
rewrite only gets 5.2 → not the fix); H2D ~1.5 ms; rest is GPU compute + dispatch.

Applied to train.py (numerics-identical, verified by CPU smoke run):
- `prefetch()` generator wrapper — batch construction now overlaps GPU compute
- `donate_argnums=(0,1,2)` on train_step — in-place buffer reuse for params/opt state

Live A/B verdict: patched loop measured 29.3 ms/step (concurrent load) and 30.0
(clean) vs 26.8 unpatched — the patch was ~3 ms SLOWER (prefetch-thread GIL
contention; async dispatch already overlapped host work). **Reverted**; train.py
keeps only the argparse additions. Conclusion: the loop was already at its floor;
step time is GPU-dominated. Real speedups = bf16 matmuls / larger batch — both
change numerics, park until after the pilot. (cleanup_notrust trained on the
patched loop — numerics-identical, just slower; results unaffected.)

Bigger levers, NOT applied (change numerics → would need re-baselining mid-pilot):
- bf16 matmul precision (prod Rust inference is already bf16; plausible ~2x on GPU time)
- batch size 512 → 1024+ (utilization vs dynamics tradeoff)
Eval harness is the other slow leg (~40 min/CPU eval, per-item unjitted rank loop —
vectorizable to minutes if eval count grows).

## Notes / gotchas

- normalize_ratings emits a cosmetic NaN warning for zero-rated profiles (sigma of
  empty slice) — pre-existing, values are guarded afterwards.
- Known train/eval inconsistency found during review: training drops ALL PTW while
  prod + harness keep rated PTW (`filter_profile_entries`, Rust recommend.rs). Only
  `cleanup` changes this on the training side.
