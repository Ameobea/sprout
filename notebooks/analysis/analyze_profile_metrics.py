"""
Aggregate report over per-user metrics from extract_profile_metrics.py.
Usage: analyze_profile_metrics.py <metrics.csv> [dump_date YYYY-MM-DD, default 2025-12-21]
"""

import sys
from datetime import date

import numpy as np
import pandas as pd

MIN_RECENT = date(2020, 8, 10).toordinal()
DUMP_DAY = date.fromisoformat(sys.argv[2]).toordinal() if len(sys.argv) > 2 else date(2025, 12, 21).toordinal()
BUCKETS = [(10, 29), (30, 99), (100, 299), (300, 10_000_000)]

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")


def pctiles(s, qs=(5, 25, 50, 75, 95, 99)):
    v = np.percentile(s, qs)
    return "  ".join(f"p{q}={x:,.0f}" for q, x in zip(qs, v)) + f"  mean={s.mean():,.1f}"


def main():
    df = pd.read_csv(sys.argv[1])
    print(f"rows: {len(df):,}")

    df["span"] = (df.upd_max - df.upd_min).clip(lower=0)
    df["span_robust"] = (df.upd_p90 - df.upd_p10).clip(lower=0)
    df["rated_span"] = (df.upd_rated_max - df.upd_rated_min).clip(lower=0)
    df["age_at_dump"] = DUMP_DAY - df.upd_min
    df["recency"] = DUMP_DAY - df.upd_max
    df["frac_unrated_model"] = 1 - df.n_model_rated / df.n_model_input.clip(lower=1)

    has_upd = df.upd_max > 0

    # ---- filter funnel ----
    s1 = (df.n_entries >= 10) & (df.upd_max > MIN_RECENT)
    s2 = s1 & (df.n_entries >= 30)
    s3 = s2 & (df.n_model_input >= 1)
    print("\n=== FILTER FUNNEL (recomputed; stage1 recency approximated non-PTW) ===")
    print(f"all users:                     {len(df):>10,}")
    print(f"stage1 (>=10 entries+recent):  {s1.sum():>10,}")
    print(f"stage2 (>=30 entries):         {s2.sum():>10,}")
    print(f"stage3 (>=1 model input):      {s3.sum():>10,}  (actual training set: 1,489,525)")

    t = df[s3].copy()

    print("\n=== GLOBAL DISTRIBUTIONS (training-passing users) ===")
    for col in ["n_entries", "n_rated", "n_model_input", "n_model_rated", "n_ptw"]:
        print(f"{col:>15}: {pctiles(t[col])}")
    print(f"{'mean_score':>15}: {pctiles(t[t.n_rated>0].mean_score, (5,25,50,75,95))}")
    print(f"{'std_score':>15}: {pctiles(t[t.n_rated>0].std_score, (5,25,50,75,95))}")

    tot_entries = df.n_entries.sum()
    print(f"\ntotal list entries (all users): {tot_entries:,}")
    for c in ["n_completed", "n_watching", "n_onhold", "n_dropped", "n_ptw"]:
        print(f"  {c}: {df[c].sum():,} ({df[c].sum()/tot_entries:.1%})")
    print(f"  rated (score>0): {df.n_rated.sum():,}")

    # ---- history length ----
    print("\n=== HISTORY LENGTH (training-passing, days) ===")
    for col in ["span", "span_robust", "rated_span", "n_upd_days", "age_at_dump", "recency"]:
        print(f"{col:>15}: {pctiles(t[col])}")

    print("\nspan buckets (share of training users / of their model-input entries):")
    edges = [(0, 1), (2, 7), (8, 30), (31, 90), (91, 365), (366, 3650), (3651, 10**9)]
    for lo, hi in edges:
        m = (t.span >= lo) & (t.span <= hi)
        print(f"  span {lo:>5}-{hi:<7}: {m.mean():6.1%} users  {t[m].n_model_input.sum()/t.n_model_input.sum():6.1%} entries  {t[m].n_model_rated.sum()/t.n_model_rated.sum():6.1%} rated")

    # ---- burst archetypes ----
    t["burst7"] = t.span <= 7
    t["burst30"] = t.span <= 30
    t["burst_robust"] = t.span_robust <= 7
    t["new_dump"] = (t.span <= 30) & (t.age_at_dump <= 90)
    t["old_burst"] = (t.span <= 30) & (t.age_at_dump > 90)
    t["few_days"] = t.n_upd_days <= 3

    print("\n=== BURST / ACCOUNT-DUMP ARCHETYPES (training-passing) ===")
    for c in ["burst7", "burst30", "burst_robust", "new_dump", "old_burst", "few_days"]:
        m = t[c]
        print(f"{c:>13}: {m.mean():6.1%} users, {t[m].n_model_rated.sum()/t.n_model_rated.sum():6.1%} of rated entries")

    print("\nbehavior: burst30 vs long-history (span>365):")
    grp = {"burst30 (new_dump)": t[t.new_dump], "burst30 (old)": t[t.old_burst], "span 31-365": t[(t.span > 30) & (t.span <= 365)], "span>365": t[t.span > 365]}
    cols = ["n_entries", "n_rated", "mean_score", "std_score", "mode_frac", "n_distinct_scores", "mean_corpus_rank", "frac_dates", "frac_unrated_model", "n_ptw"]
    rows = []
    for name, g in grp.items():
        gr = g[g.n_rated > 0]
        rows.append([name, len(g)] + [gr[c].median() for c in cols])
    print(pd.DataFrame(rows, columns=["group", "n"] + [f"med_{c}" for c in cols]).to_string(index=False))

    # ---- rating-quality flags ----
    print("\n=== RATING-QUALITY FLAGS (training-passing) ===")
    r = t[t.n_model_rated > 0]
    flags = {
        "all ratings identical (n_distinct==1, >=10 rated)": (t.n_distinct_scores == 1) & (t.n_rated >= 10),
        "mode_frac>=0.9 (>=10 rated)": (t.mode_frac >= 0.9) & (t.n_rated >= 10),
        "mean_score>=9.5 (>=10 rated)": (t.mean_score >= 9.5) & (t.n_rated >= 10),
        "mean_score<=3 (>=10 rated)": (t.mean_score <= 3) & (t.n_rated >= 10) & (t.n_rated > 0),
        "zero rated (pure presence)": t.n_model_rated == 0,
        "frac_unrated_model>=0.5": t.frac_unrated_model >= 0.5,
        "huge lists (>2000 entries)": t.n_entries > 2000,
    }
    for name, m in flags.items():
        mm = m & s3.reindex(t.index, fill_value=True)
        print(f"  {name:<50}: {t[m].shape[0]:>8,} users ({m.mean():5.1%}), {t[m].n_model_input.sum()/t.n_model_input.sum():5.1%} of model entries")

    # ---- train/serve PTW skew ----
    rated_ptw_corpus = (df.n_rated_in_corpus - df.n_model_rated).clip(lower=0)
    print("\n=== TRAIN/SERVE SKEW: rated PTW (kept at inference, dropped in training) ===")
    print(f"  users with >=1 rated in-corpus PTW: {(rated_ptw_corpus>0).sum():,} ({(rated_ptw_corpus>0).mean():.1%})")
    print(f"  total rated in-corpus PTW entries:  {rated_ptw_corpus.sum():,}")

    # ---- eval-bucket contamination ----
    print("\n=== BURST SHARE BY FIXTURE BUCKET (bucketed on n_entries like sampler) ===")
    for lo, hi in BUCKETS:
        m = t[(t.n_entries >= lo) & (t.n_entries <= hi)]
        print(f"  {lo}-{hi if hi<10**6 else '+'}: n={len(m):,}  burst30={m.burst30.mean():.1%}  new_dump={m.new_dump.mean():.1%}  span_med={m.span.median():,.0f}d")

    t.to_pickle(sys.argv[1].replace(".csv", "_training.pkl"))
    print("\ntraining-subset frame pickled for follow-up analysis")


if __name__ == "__main__":
    main()
