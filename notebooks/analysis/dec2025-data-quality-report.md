# Dec 2025 training data — quality analysis

Source: full pass over `data/mal-user-animelists.csv.xz` (191GB raw, 1,898,159 rows:
1,784,860 non-empty profiles + 113,299 empty/private lists). Per-user metrics in
`data/dec2025-profile-metrics.csv` (`extract_profile_metrics.py`); aggregates from
`analyze_profile_metrics.py`. "Training-passing" = recomputed filter funnel
(1,487,586 vs actual 1,489,525 — 0.13% off due to stage-1 recency approximation).

## Current filter funnel (verified from code)

1. `process-collected-profiles.ipynb`: keep users with ≥10 list entries (ANY status,
   including plan_to_watch and unrated) and ≥1 entry updated after 2020-08-10.
2. `vectorize_training_data.ipynb`: require ≥30 CSV rows (again any status incl. PTW),
   then per entry: drop PTW (even rated ones), drop out-of-corpus, drop on_hold+unrated;
   dropped+unrated → -2 sentinel; unrated watched → mean-fill with rated_flag=False
   (rating gradient masked in train.py).
3. Corpus: top 6000 by rating count minus rx-rated.

Funnel: 1,784,860 non-empty → 1,592,485 (stage 1) → 1,488,589 (≥30 entries) →
1,489,525 training vectors, 355.9M vector entries.

## Global characteristics (training-passing users)

| metric | p5 | p25 | p50 | p75 | p95 | p99 | mean |
|---|---|---|---|---|---|---|---|
| n_entries | 46 | 116 | 231 | 432 | 983 | 1,760 | 342 |
| n_rated | 0 | 40 | 110 | 234 | 585 | 1,095 | 180 |
| n_model_input | 33 | 83 | 164 | 306 | 692 | 1,195 | 239 |
| n_model_rated | 0 | 39 | 109 | 229 | 564 | 1,015 | 174 |

Status mix over all 526M list entries: completed 63.5%, PTW 25.2%, watching 5.1%,
dropped 3.7%, on_hold 2.5%. 278M entries scored (score>0).
User mean_score: median 8, p95 = 10. 27.4% of model-input entries are unrated (mean-filled input, gradient-masked).

## History length ("how long has the user been rating")

| metric (days) | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|
| span (first→last non-PTW update) | 90 | 759 | 1,423 | 2,187 | 4,053 |
| span_robust (p10→p90) | 0 | 368 | 871 | 1,514 | 3,043 |
| rated_span | 0 | 139 | 686 | 1,395 | 2,777 |
| n distinct update days | 3 | 19 | 49 | 111 | 296 |
| account age at dump | 733 | 1,353 | 1,775 | 2,509 | 4,313 |
| recency (dump − last update) | 2 | 27 | 213 | 746 | 1,313 |

Median training user: ~3.9y list span, ~4.9y account age, active on 49 distinct days.

**New-account dumps are rare**: age ≤90d = 0.65% of training users (0.92% of all
collected — the recently-online scrape mostly surfaces established accounts).
Burst archetypes among training users:

| archetype | users | share of rated entries |
|---|---|---|
| span ≤7d | 2.4% | 0.7% |
| span ≤30d | 3.4% | 1.0% |
| span_robust ≤7d (mass-import+continue) | 7.0% | 3.2% |
| new_dump (span≤30d AND age≤90d) | 0.4% | 0.2% |
| old burst (span≤30d, age>90d) | 3.0% | 0.9% |
| ≤3 distinct update days | 5.1% | 1.4% |
| **rated_span ≤7d, ≥10 rated (all ratings in one sitting)** | **6.5% of raters** | **3.0%** |
| — of which old list (span>365d) mass-rated later | 3.6% of raters | 1.9% |

Burst/mass-raters ARE behaviorally different (hypothesis confirmed directionally):
higher scores (med 8.06–8.12 vs 7.81 long-history), fewer distinct scores, more
mainstream taste (med corpus rank 465–640 vs 759–772), and essentially never enter
start/finish dates (med frac_dates 0.00 vs 0.22).

Caveat: updated_at is bumped by ANY list edit, so mass-editors of old lists compress
into "burst". rated_span is the cleaner signal for rating trustworthiness.

## Flagged problems (ranked)

1. **27.4% of model-input entries unrated**; 22.8% of users have ≥50% unrated inputs;
   6.9% pure-presence (0 ratings). By design (masked gradient) but a large mean-fill
   share on the input channel.
2. **Huge lists**: >2000 entries = 0.7% of users but 4.3% of model entries (max
   29,439; 785 users >5000). Near-uniform presence targets, likely bots/list-collectors.
3. **One-sitting raters**: 6.5% of raters / 3.0% of rated entries (includes the
   new-account dump case, which alone is negligible).
4. **Train/serve skew — rated PTW**: prod (`filter_profile_entries` + Rust
   `recommend.rs`) keeps rated PTW at inference; training drops ALL PTW. 878k rated
   in-corpus PTW entries, 12% of users affected.
5. **Degenerate raters**: all-identical scores 0.6%, mode_frac≥0.9 1.4%, mean≥9.5
   2.7% of users. Z-score channel degenerate (alpha clips to 0.3).
6. **Stage gates count PTW**: the ≥10 and ≥30 thresholds count PTW + unrated entries,
   so "30 entries" can be mostly PTW. Gate on model-input entries instead.
7. **Eval fixture contamination** (matters for filtering experiments): fixture bucket
   10-29 samples stage-1-only users (below training threshold) where **32.6% are
   burst30** (med span 161d); bucket 30-99 is 10.4% burst30; 100-299 2.4%; 300+ 0.7%.
8. `min_latest_updated=2020-08-10` is a stale constant (5.5y back); parametrize for
   the new pull.
9. Median recency 213d; 25% of training users dormant 2+ years (mild; taste drift).

## Proposed experiments (pending review)

Mechanics: branch off main; vectorizer parameterized by a filter config joining
per-user flags from `dec2025-profile-metrics.csv` by username; each variant = new
`user_input_vectors` file → full retrain (December data) → harness eval vs
`dec2025-baseline` (same corpus, no --restrict needed).

0. **Fixtures v2 first** (prerequisite): larger frozen set (~300–400) sampled from the
   RAW dump, stratified size-bucket × history-class (long-history / burst / one-sitting
   raters), including sub-30-entry users. Keeps v1 untouched; per-class slices show
   where each filter helps/hurts real vs garbage users.
1. **Rating-trust mask** (surgical, preferred): keep presence, set rated_flag=False for
   users with rated_span≤7d & n_rated≥10, plus degenerate raters (n_distinct==1 or
   mode_frac≥0.9). Kills ~4% of rating gradient, keeps co-occurrence.
2. **Hard drop variant**: same predicate but drop users entirely (tests whether their
   presence signal was helping or hurting).
3. **Huge-list handling**: drop (or subsample entries of) profiles >2000 entries.
4. **Rated-PTW alignment**: include rated PTW in training vectors (match prod). Fold
   into all variants; fixes skew #4.
5. **Gate cleanup**: stage-2 threshold on n_model_input ≥20-30 instead of raw entries.
6. (Softer alternative to 1/2): per-user loss weight from history length — only if the
   mask variants show signal.

Expectation: aggregate deltas likely small (burst populations are 1–3% of rated
entries); per-class fixture slices are where effects will show. Combined "conservative
cleanup" (1+3+4+5) is the headline candidate for the new-data training run.
