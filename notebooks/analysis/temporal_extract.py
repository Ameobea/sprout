"""Per-entry start_date extraction aligned to user_input_vectors_cleanup_notrust
entry order: replicates the cleanup vectorizer gating (huge-user drop, rated-PTW
keep, >=20 entries) on a stream of the collected CSV. Emits start_day int16
(days since 2000-01-01, -32768 = missing/garbage) plus lengths for verification
against the npz. Run on host venv."""

import csv
import gzip
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

EPOCH = date(2000, 1, 1).toordinal()
MISSING = -32768


def load_huge(metrics_path):
    df = pd.read_csv(metrics_path, usecols=["username", "n_entries"])
    return set(df.username[df.n_entries > 2000])


def make_parser():
    cache = {}

    def parse(s):
        v = cache.get(s)
        if v is not None:
            return v
        parts = s.split("-")
        try:
            y = int(parts[0])
            if not (1950 <= y <= 2035):
                raise ValueError
            m = int(parts[1]) if len(parts) > 1 and parts[1] else 1
            dd = int(parts[2]) if len(parts) > 2 and parts[2] else 1
            v = date(y, max(m, 1), max(dd, 1)).toordinal() - EPOCH
        except (ValueError, IndexError):
            v = MISSING
        cache[s] = v
        return v

    return parse


def users_full(path):
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        cur, rows = None, []
        for row in reader:
            if row[0] != cur:
                if cur is not None:
                    yield cur, rows
                cur, rows = row[0], []
            rows.append(row)
        if cur is not None:
            yield cur, rows


def main():
    src, metrics_path, corpus_path, npz_path, out_path = sys.argv[1:6]
    with open(corpus_path) as f:
        ix_by_id = {aid: i for i, aid in enumerate(json.load(f))}
    huge = load_huge(metrics_path)
    parse = make_parser()

    all_days, all_rated, lengths = [], [], []
    for n_users, (username, rows) in enumerate(users_full(src), 1):
        if n_users % 200_000 == 0:
            print(f"{n_users:,} users scanned, {len(lengths):,} kept", flush=True)
        if username in huge:
            continue
        days, rated = [], []
        for row in rows:
            aid = int(row[1])
            if aid not in ix_by_id:
                continue
            score = float(row[2])
            status = row[3]
            if status == "plan_to_watch" and not score > 0:
                continue
            if status == "on_hold" and score == 0:
                continue
            rated.append(score > 0)
            days.append(parse(row[4]) if row[4] else MISSING)
        if len(days) >= 20:
            all_days.append(np.array(days, dtype=np.int16))
            all_rated.append(np.array(rated, dtype=bool))
            lengths.append(len(days))

    days = np.concatenate(all_days)
    rated = np.concatenate(all_rated)
    lengths = np.array(lengths, dtype=np.int32)

    vec = np.load(npz_path)
    ref = vec["lengths"]
    ok = len(lengths) == len(ref) and bool((lengths == ref).all())
    print(f"alignment: users {len(lengths):,} vs npz {len(ref):,} -> {'OK' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        diff = np.nonzero(lengths[: len(ref)] != ref[: len(lengths)])[0][:5]
        print(f"first diffs at {diff}", flush=True)

    dated = days != MISSING
    print(f"entries {len(days):,}  dated {dated.mean():.4f}  dated|rated {dated[rated].mean():.4f}")
    per_user_frac = np.array([ (d != MISSING).mean() for d in all_days ])
    print("per-user dated frac quartiles:", np.percentile(per_user_frac, [25, 50, 75, 90, 95]).round(3))
    print(f"users with >=30% dated: {(per_user_frac >= 0.3).mean():.4f}")

    np.savez(out_path, start_day=days, lengths=lengths, aligned=np.array([ok]))
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
