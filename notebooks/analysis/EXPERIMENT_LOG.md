# Model experiments — log

Branches: `data-filtering-experiments` (filtering) → `input-channels` (input layouts,
temporal eval, dropout sweep, znver4 work). Raw analysis in
`dec2025-data-quality-report.md`; plan of record in `/WORKSTREAMS.md`. All trainings on
December 2025 data (same corpus as deployed model → no --restrict-corpus needed).
Tracker artifacts: filtering https://claude.ai/code/artifact/c3f7fe3b-b22f-463a-8cf4-87d906b7d591
· input channels https://claude.ai/code/artifact/b9b3c7fa-9633-4274-9390-b672f0d0c86b

## Summary of both sessions (2026-07-30 → 08-01)

17 trainings, ~25 evals. What changed and what didn't:

- **Best model**: `statuschan × cleanup_notrust` (5-channel input, validated filters) —
  LOO MAE 0.4230/0.4234 vs 0.4337 replication baseline, recall@50 0.599–0.600.
  **Decided not to deploy** (§ WORKSTREAMS 3): win is under the prod bf16 quantization
  noise per prediction, ranking unmoved, ~1 day Rust migration, 2× params.
- **Adopted**: the `cleanup_notrust` filter set (data-side only, free).
- **Closed axes** (do not redo): trust-based rating filtering (harmful), dropout rate
  (0.4 optimal), ranking blend weight (0.3 optimal), corpus size (5.2% out-of-corpus),
  host-side training loop speed (GPU-bound at floor).
- **The one thing nothing moved**: presence loss, 6.05–6.20 across all eleven models
  including a 55M→92M parameter doubling. Ranking metrics correspondingly flat
  (recall@50 0.591–0.604 for everything not deliberately broken). This is now the
  focus of the next session.
- **Built**: `temporal_eval.py` + fixtures v3 (future-watch eval, popularity baseline,
  sweepable ranking blend). Note the §0 caveat in WORKSTREAMS — it is a guardrail, not
  the product objective.
- **Incidental**: znver4 cargo-config bug found and fixed (local model-server-rs builds
  were silently ~2× slow); SIMD helpers hardened against the same class of accident.

Two calibration facts worth keeping in mind: the model beats a popularity baseline by
2–3× on the temporal eval (so it genuinely personalizes), and it reaches only ~15% of
the arithmetic recall@10 ceiling (so the *task* is far from saturated even though our
*tuning* of it is).

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

## Input-channel experiments (2026-07-31, branch `input-channels`)

Setup: `CONF["input_channels"]` flag (2=prod, 3=+rated-mask) through model/train/
harness; rated flags already in npz → no re-vectorization for maskchan. 2-channel
regression check reproduced `replication-fixv2` byte-identically after the refactor.
maskchan: 67.75M params (+12.3M first layer), 40.6 ms/step (+51%), no OOM
(rocm-smi showed 97% VRAM — mostly allocator preallocation).

### maskchan run 1 (`maskchan-fixv2.json`) — promising, mixed

overall 0.4298/0.593/54 vs replication 0.4337/0.597/54. Best MAE of any model;
gains concentrated in unrated/burst-heavy cells (burst-100-299 −0.022,
burst-300+ 0.3913 best anywhere, v1 300+ −0.007, v1 30-99 −0.007). BUT recall@50
−0.004 (v1 30-99: 0.682 vs 0.697) — possible uncertainty-weighting rebalance
toward the easier rating task at slight presence-head cost. Both deltas at the
noise floor from one run → confirmation run 2 in flight.

### maskchan run 2 (`maskchan-run2-fixv2.json`) — trade-off CONFIRMED

overall 0.4314/0.595/55. Two maskchan runs (0.4298, 0.4314) both beat every
2-channel model on MAE (replication 0.4337, cleanup_notrust 0.4359/0.4324) AND
both show the small recall dip (0.593/0.595 vs 0.597 repl, v1 recall 0.587/0.588
vs 0.594). Reproducible pattern: rated-mask channel buys ~−0.003 MAE (concentrated
in unrated/burst-heavy cells) for ~−0.003 recall@50. Median rank flat.
Open: does statuschan shift the trade, and does stacking with cleanup_notrust
filters (ranking-positive) recover the recall? cleanup_notrust npz regenerating
with statuses for the combination run.

### statuschan run 1 (`statuschan-fixv2.json`) — best MAE ever, ranking recovered

overall 0.4283/0.596/54 — best MAE of all 10 models AND recall back at baseline
(0.596 vs repl 0.597; maskchan had dipped to 0.593/0.595). v1 30-99 MAE 0.3798
and burst-300+ 0.3806 best anywhere; ameo___ recall recovered (0.579 vs maskchan
0.474). Status channels boost the rating head further without the presence-head
cost. Single run — confirmation queued after mask_cleanup. If confirmed, the
deploy-candidate combo is statuschan + cleanup_notrust filters (npz ready).

### mask_cleanup combo (`mask-cleanup-fixv2.json`)

overall 0.4287/0.595/54. MAE holds the mask-channel gain (v1 0.4030 = best v1
anywhere); filters recover part of the recall dip (0.595, v1 0.591 vs maskchan
0.587/0.588) but not to the cleanup family's 0.599. statuschan still leads
overall. If statuschan run2 confirms → final candidate run: statuschan +
cleanup_notrust filters.

### statuschan run 2 (`statuschan-run2-fixv2.json`) — CONFIRMED

overall 0.4269/0.595/54 (run 1: 0.4283/0.596/54). The two statuschan runs are the
best two MAE results of the whole effort (~−0.006 vs replication, 2x noise floor,
reproduced) with ranking at baseline. Status channels are the phase winner.
Final piece: status_cleanup combo (training) — statuschan + validated filters.

### status_cleanup combo (`status-cleanup-fixv2.json`) — BEST OF EFFORT

overall 0.4230/0.600/53 — best MAE by 3.5x noise floor (−0.011 vs replication)
AND best-tier ranking. v1 0.3974 (first sub-0.40), v2 0.4348 (best). Temporal:
0.095/0.243/342, MRR 0.432 (highest). The statuschan MAE gain and cleanup ranking
edge stack fully. Deploy candidate; confirmation run queued after dropout sweep.

### statuschan design (phase 2, queued)

5 channels: [presence | ratings | rated | dropped | in-progress] (completed =
implicit default; avoids 4-way one-hot blowup). Input 30k dims → first layer
61.4M, total ~92M params — watch VRAM. Keep the −2 dropped sentinel initially
(one variable at a time); ablate it after if statuschan wins. Needs statuses in
the npz (new vectorization) + status-aware harness preprocessing.

## Inference-perf assessment for wider inputs (Opus subagent, 2026-07-31)

Encoder gather in model-server-rs is genuinely sparse — cost scales with weight
row-touches/item (2.0 today; 2.73 at 3ch, 2.86 at 5ch from real fixture status
rates), NOT input width. 5ch ≈ 3ch (dropped+in-progress add 0.13 rows/item).

**Codegen-bug claim RETRACTED after re-verification — prod is fine.** The
degraded binary was a LOCAL build artifact: `~/.cargo/config.toml` defines
`[target.x86_64-unknown-linux-gnu].rustflags` (lld + uuid cfg), and cargo's
rustflags sources are mutually exclusive — the global target-triple entry
silently discards the project's `build.rustflags` incl. `-C target-cpu=znver4`
(`cargo config get` still shows the merged value — deceptive). The prod docker
build (`Dockerfile.model_server_rs`, clean CARGO_HOME) applies znver4 correctly;
verified experimentally. Consequences: (1) local benches/tests of model-server-rs
run ~2× slower than prod — fix by moving znver4 to a target-triple entry in the
project config or exporting RUSTFLAGS; (2) `#[target_feature]` on the simd
helpers would harden against this accident class (zero cost) — neither applied.

Corrected wider-input costs (znver4 build, p50, rows=1 recommender path):
n=250: 1.100 ms (2ch) → 1.139 (+3.5%) 3ch → 1.169 (+6.3%) 5ch;
n=800: 1.621 → 1.762 (+8.7%) → 1.817 (+12.1%). Holdout path: <1% at all sizes.
Negligibility hypothesis HOLDS. Implementation note: the wider encoder must
skip zero-valued flag rows — naive emit-all-channels costs +19-40%.

Migration for wider inputs ~1 day: IN_DIM derivation, weights.rs shape assert
(currently PANICS on the new checkpoints), enc_row/axpy widening + per-item
flags through Prepped/HoldoutDelta, reconcile the -2 dropped sentinel vs explicit
dropped channel, regen goldens. Decoder kernels unaffected. RSS +49 MB (3ch) /
+147 MB (5ch). Scratch bench: `model-server-rs/src/bin/scratch_bench_channels.rs`
(untracked; REAL_PROFILES/ITERS/SIZES env vars; DIAG=1 shows the codegen issue).

## Temporal (future-watch) eval — first results (2026-07-31 night)

Fixtures v3: 300 users (t-30-99/t-100-299/t-300+ ×100, seed 33), input = profile
before 2025-06-24, targets = in-corpus completed/watching added after (≥5).
`temporal_eval.py`: one forward pass/user, logit-weight sweepable, popularity
baseline built in. Eligible pools huge (60k/203k/219k) — can scale fixtures.

Key findings (recall@10 / recall@50 / med target rank, lw=0.3):
| model | r@10 | r@50 | med rank |
|---|---|---|---|
| popularity baseline | 0.040 | 0.124 | 673 |
| dec2025 deployed | 0.091 | 0.243 | 366 |
| replication | 0.090 | 0.227 | 363 |
| cleanup_notrust | 0.090 | 0.244 | 343 |
| maskchan | 0.085 | 0.243 | 346 |
| mask_cleanup | 0.088 | 0.233 | 352 |
| statuschan | 0.093 | 0.249 | 356 |
| statuschan run2 | 0.098 | 0.241 | 327 |

1. Model >> popularity (2-3x on recall, med rank 330-400 vs 673) — real
   personalization signal, not near the trivial floor.
2. statuschan family best here too (top-2 recall@10, best med ranks) —
   directionally consistent with LOO, but deltas are near the temporal noise
   floor (statuschan run1 vs run2 spread: r@10 0.093→0.098, med rank 356→327 —
   substantial run noise at n=300 fixtures).
3. **logit_weight=0.3 is validated**: the sweep peaks at 0.3 (recall@10, med
   rank) or 0.1-0.3 (recall@50/100); higher presence-weighting uniformly worse.
   The vibes-tuned prod value was already right — no free win, but now measured.
4. Deployed dec2025 performs as well as any experiment on the product task —
   the honest "diminishing returns on this architecture/objective" datapoint.

## znver4 hardening — APPLIED (working tree, uncommitted, 2026-07-31 night)

Files: model-server-rs/.cargo/config.toml (znver4 moved to a
`[target.x86_64-unknown-linux-gnu]` entry), engine.rs (+1 attr), simd.rs (11
attrs incl. safe-wrapper splits), kernels.rs (a_to_bf16 — beyond the requested
list, drop that hunk if unwanted). Agent deliberately did NOT re-add lld/uuid to
the project config: cargo MERGES same-key arrays across config files (tested), so
locally znver4+lld+uuid all apply now, while the prod docker build's flags stay
byte-identical (re-adding would have newly injected lld into the prod image,
which likely lacks the linker — potential build break avoided).

Verified: all tests pass with outputs bitwise-identical to pristine HEAD;
disassembly under a deliberately-lost config shows full vectorization (e.g.
combined_score_into 0 zmm/124 calls → 220 zmm/0 calls); no perf regression
(n=800 forward 1.472 ms). Local builds are now ~2× faster (znver4 finally applies).

**Pre-existing issue flagged (not fixed)**: golden tests default to
`data/jax_model.msgpack`, which every training run overwrites — 3/4 tests fail
against it since the replication run; they pass against `jax_model_dec2025.msgpack`
(what the goldens were generated from). Repoint the test default to the stable
checkpoint or regenerate goldens.

## Dropout sweep (overnight 2026-07-31→08-01, 2ch baseline vectors)

| rate | LOO MAE | LOO r@50 | temporal r@10 | temporal med rank | MRR |
|---|---|---|---|---|---|
| 0.2 | 0.4514 | 0.594 | 0.093 | 382 | 0.428 |
| 0.3 | 0.4418 | 0.598 | 0.087 | 369 | 0.408 |
| 0.4 (baseline) | 0.4337 | 0.597 | 0.090 | 363 | 0.389 |
| 0.5 | 0.4348 | 0.591 | 0.088 | 355 | 0.409 |
| 0.65 | 0.4421 | 0.574 | 0.085 | 333 | 0.390 |

Verdict: 0.4 already optimal. Low dropout notably WORSE on MAE
(under-regularization); 0.65 craters LOO ranking. dropout_variation=±40% means
base 0.4 already samples 0.24-0.56 per batch → flat middle. Interesting trend:
temporal median rank improves monotonically with rate (tail robustness) while
MRR/top-list sharpness degrades — corruption-scheme design hint, not actionable
now. Knob closed.

## Notes / gotchas

- normalize_ratings emits a cosmetic NaN warning for zero-rated profiles (sigma of
  empty slice) — pre-existing, values are guarded afterwards.
- Known train/eval inconsistency found during review: training drops ALL PTW while
  prod + harness keep rated PTW (`filter_profile_entries`, Rust recommend.rs). Only
  `cleanup` changes this on the training side.
