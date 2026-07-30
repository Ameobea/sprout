"""
One-time seeded sampler: freezes a stratified set of December-2025 profiles from
collected_animelists.csv.gz into fixtures/sampled_profiles.json for the eval harness.
Relies on the CSV being grouped by username (it is written that way by
process-collected-profiles). Deterministic given the same input file + seed.
"""

import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

SEED = 1234
PER_BUCKET = 40
BUCKETS = [(10, 29), (30, 99), (100, 299), (300, 10_000)]
EXCLUDE = {"ameo___", "snapsauce"}

SRC = Path(__file__).parent / "../../data/collected_animelists.csv.gz"
OUT = Path(__file__).parent / "fixtures/sampled_profiles.json"


def bucket_of(n):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= n <= hi:
            return i
    return None


def profiles(path):
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        cur_user, items = None, []
        for row in reader:
            username, anime_id, score, status = row[0], int(row[1]), int(row[2]), row[3]
            if username != cur_user:
                if cur_user is not None:
                    yield cur_user, items
                cur_user, items = username, []
            items.append((anime_id, score, status))
        if cur_user is not None:
            yield cur_user, items


def main():
    rngs = [np.random.default_rng(SEED + i) for i in range(len(BUCKETS))]
    reservoirs = [[] for _ in BUCKETS]
    seen = [0] * len(BUCKETS)

    for i, (username, items) in enumerate(profiles(SRC)):
        if i % 100_000 == 0:
            print(f"{i} profiles scanned", file=sys.stderr)
        if username in EXCLUDE:
            continue
        b = bucket_of(len(items))
        if b is None:
            continue
        seen[b] += 1
        if len(reservoirs[b]) < PER_BUCKET:
            reservoirs[b].append((username, items))
        else:
            j = rngs[b].integers(0, seen[b])
            if j < PER_BUCKET:
                reservoirs[b][j] = (username, items)

    out = {}
    for (lo, hi), res in zip(BUCKETS, reservoirs):
        for username, items in res:
            out[username] = {"bucket": f"{lo}-{hi}", "items": items}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"wrote {len(out)} profiles to {OUT} (bucket counts seen: {seen})")


if __name__ == "__main__":
    main()
