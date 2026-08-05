"""One streaming pass over the collected CSV, cleanup-gated to match the training
npz user order exactly: per-user raw-score histogram over model-input entries
(bin 0 = unrated, 1..10 = raw score), dropped-unrated count, and trust flags
joined from the metrics CSV. Output aligns row i to npz user i."""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vectorize_variants import users


def main():
    src, metrics_path, corpus_path, out_prefix = sys.argv[1:5]

    df = pd.read_csv(
        metrics_path,
        usecols=["username", "n_entries", "n_rated", "n_distinct_scores", "mode_frac",
                 "upd_rated_min", "upd_rated_max"],
    )
    rated_span = (df.upd_rated_max - df.upd_rated_min).clip(lower=0)
    one_sitting = set(df.username[(rated_span <= 7) & (df.n_rated >= 10) & (df.upd_rated_max > 0)])
    degenerate = set(df.username[(df.n_rated >= 10) & ((df.n_distinct_scores == 1) | (df.mode_frac >= 0.9))])
    huge = set(df.username[df.n_entries > 2000])
    print(f"one_sitting {len(one_sitting):,}  degenerate {len(degenerate):,}  huge {len(huge):,}", flush=True)

    with open(corpus_path) as f:
        ix_by_id = {aid: i for i, aid in enumerate(json.load(f))}

    hists, drop_unrated, names = [], [], []
    for n_scanned, (username, rows) in enumerate(users(src), 1):
        if n_scanned % 200_000 == 0:
            print(f"{n_scanned:,} users scanned, {len(hists):,} kept", flush=True)
        if username in huge:
            continue
        h = np.zeros(11, dtype=np.uint16)
        n_du = 0
        for aid, score, status in rows:
            if aid not in ix_by_id:
                continue
            if status == "plan_to_watch" and not score > 0:
                continue
            if status == "on_hold" and score == 0:
                continue
            s = int(score)
            h[s] += 1
            if s == 0 and status == "dropped":
                n_du += 1
        if h.sum() >= 20:
            hists.append(h)
            drop_unrated.append(n_du)
            names.append(username)

    hists = np.stack(hists)
    is_os = np.fromiter((n in one_sitting for n in names), dtype=bool, count=len(names))
    is_dg = np.fromiter((n in degenerate for n in names), dtype=bool, count=len(names))
    np.savez(
        f"{out_prefix}.npz",
        hist=hists,
        n_dropped_unrated=np.array(drop_unrated, dtype=np.uint16),
        one_sitting=is_os,
        degenerate=is_dg,
    )
    with gzip.open(f"{out_prefix}_usernames.txt.gz", "wt") as f:
        f.write("\n".join(names))
    print(f"{out_prefix}.npz: {len(names):,} users  one_sitting {is_os.mean():.3%}  degenerate {is_dg.mean():.3%}", flush=True)


if __name__ == "__main__":
    main()
