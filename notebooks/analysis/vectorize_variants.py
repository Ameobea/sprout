"""
Script version of vectorize_training_data.ipynb with per-user quality filters.
One streaming pass over collected_animelists.csv.gz produces every requested
variant's user_input_vectors_{name}.npz (same npz format train.py loads).

Variants:
  baseline  - exact replica of the notebook logic (validation reference)
  trustmask - rating gradient masked (rated_flag=False) for one-sitting raters
              (rated_span<=7d, n_rated>=10) and degenerate raters
              (n_rated>=10 and (n_distinct_scores==1 or mode_frac>=0.9));
              scores still feed the input channel
  harddrop  - those users dropped entirely
  cleanup   - trustmask + drop lists >2000 entries + keep rated PTW entries
              + gate on >=20 model-input entries instead of >=30 raw rows

Usage: vectorize_variants.py <collected.csv.gz> <metrics.csv> <corpus_ids.json> <outdir> [variant ...]
"""

import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from normalize_ratings import normalize_ratings

MIN_RAW_ROWS = 30
VARIANTS = ["baseline", "trustmask", "harddrop", "cleanup", "cleanup_notrust"]


def load_flags(metrics_path):
    df = pd.read_csv(
        metrics_path,
        usecols=["username", "n_entries", "n_rated", "n_distinct_scores", "mode_frac",
                 "upd_rated_min", "upd_rated_max"],
    )
    rated_span = (df.upd_rated_max - df.upd_rated_min).clip(lower=0)
    one_sitting = (rated_span <= 7) & (df.n_rated >= 10) & (df.upd_rated_max > 0)
    degenerate = (df.n_rated >= 10) & ((df.n_distinct_scores == 1) | (df.mode_frac >= 0.9))
    untrusted = set(df.username[one_sitting | degenerate])
    huge = set(df.username[df.n_entries > 2000])
    print(f"untrusted raters: {len(untrusted):,}  huge lists: {len(huge):,}")
    return untrusted, huge


def process(user_ratings, ix_by_id, keep_rated_ptw):
    indices, scores, rated = [], [], []
    for aid, score, status in user_ratings:
        ix = ix_by_id.get(aid)
        if ix is None:
            continue
        if status == "plan_to_watch" and not (keep_rated_ptw and score > 0):
            continue
        if status == "on_hold" and score == 0:
            continue
        if status == "dropped" and score == 0:
            score = -2
            rated.append(False)
        elif score == 0:
            rated.append(False)
        else:
            rated.append(True)
        indices.append(ix)
        scores.append(score)
    if not indices:
        return None
    norm, _ = normalize_ratings(np.array(scores, dtype=np.float32))
    return (
        np.array(indices, dtype=np.int16),
        norm.astype(np.float32),
        np.array(rated, dtype=bool),
    )


def save(vectors, path):
    all_idx, all_val, all_rated, lengths = [], [], [], []
    for idxs, vals, rated in vectors:
        all_idx.append(idxs)
        all_val.append(vals)
        all_rated.append(rated)
        lengths.append(len(idxs))
    masks = np.concatenate(all_rated).astype(np.uint8)
    np.savez(
        path,
        indices=np.concatenate(all_idx).astype(np.int16),
        values=np.concatenate(all_val).astype(np.float32),
        lengths=np.array(lengths, dtype=np.int32),
        rated_masks=np.packbits(masks),
        total_mask_bits=np.array([len(masks)], dtype=np.int64),
    )
    print(f"{path}: {len(vectors):,} users, {len(masks):,} entries, "
          f"{int(masks.sum()):,} rated ({masks.mean():.1%})")


def users(path):
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        cur, rows = None, []
        for row in reader:
            if row[0] != cur:
                if cur is not None:
                    yield cur, rows
                cur, rows = row[0], []
            rows.append((int(row[1]), float(row[2]), row[3]))
        if cur is not None:
            yield cur, rows


def main():
    src, metrics_path, corpus_path, outdir = sys.argv[1:5]
    variants = sys.argv[5:] or VARIANTS

    with open(corpus_path) as f:
        ix_by_id = {aid: i for i, aid in enumerate(json.load(f))}
    untrusted, huge = load_flags(metrics_path)

    out = {v: [] for v in variants}
    for n_users, (username, rows) in enumerate(users(src), 1):
        if n_users % 200_000 == 0:
            print(f"{n_users:,} users scanned", flush=True)
        base = process(rows, ix_by_id, keep_rated_ptw=False) if len(rows) >= MIN_RAW_ROWS else None
        is_untrusted = username in untrusted

        if "baseline" in out and base is not None:
            out["baseline"].append(base)
        if "trustmask" in out and base is not None:
            if is_untrusted:
                out["trustmask"].append((base[0], base[1], np.zeros(len(base[0]), dtype=bool)))
            else:
                out["trustmask"].append(base)
        if "harddrop" in out and base is not None and not is_untrusted:
            out["harddrop"].append(base)
        if ("cleanup" in out or "cleanup_notrust" in out) and username not in huge:
            res = process(rows, ix_by_id, keep_rated_ptw=True)
            if res is not None and len(res[0]) >= 20:
                if "cleanup_notrust" in out:
                    out["cleanup_notrust"].append(res)
                if "cleanup" in out:
                    if is_untrusted:
                        res = (res[0], res[1], np.zeros(len(res[0]), dtype=bool))
                    out["cleanup"].append(res)

    print(f"done scanning {n_users:,} users")
    for v in variants:
        save(out[v], str(Path(outdir) / f"user_input_vectors_{v}.npz"))


if __name__ == "__main__":
    main()
