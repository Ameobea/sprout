"""
One-time seeded sampler for temporal fixtures (v3): input = profile as of CUTOFF,
targets = in-corpus items the user watched after CUTOFF. Measures the real product
task (predict future watches) instead of LOO reconstruction.

Per user: input_items = entries updated before CUTOFF (any status; model-side
filtering applies as usual); targets = entries updated on/after CUTOFF with status
completed/watching, in corpus, not already an input item. Items that were PTW
before CUTOFF can be targets (they were never model inputs). Known limitation:
updated_at moves on ANY edit, so bulk re-editors leak old watches into targets.

Usage: sample_temporal_fixtures.py <dump.csv.xz> <corpus_ids.json>
"""

import csv
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import orjson

SEED = 33
CUTOFF = "2025-06-24"
PER_BUCKET = 100
BUCKETS = [(30, 99), (100, 299), (300, 10_000_000)]
MIN_TARGETS = 5
OUT = Path(__file__).parent / "fixtures/temporal_v3.json"

csv.field_size_limit(sys.maxsize)


def split_profile(parsed, corpus):
    inputs, targets = [], []
    input_ids = set()
    for node in parsed:
        ls = node.get("list_status") or {}
        n = node.get("node") or {}
        aid, status, ts = n.get("id"), ls.get("status"), ls.get("updated_at")
        if aid is None or status is None or not ts:
            continue
        score = ls.get("score") or 0
        if ts[:10] < CUTOFF:
            inputs.append((aid, score, status))
            if status != "plan_to_watch":
                input_ids.add(aid)
        elif status in ("completed", "watching") and aid in corpus:
            targets.append((aid, score))
    targets = [(a, s) for a, s in targets if a not in input_ids]
    return inputs, targets


def main():
    dump_path, corpus_path = sys.argv[1], sys.argv[2]
    with open(corpus_path) as f:
        corpus = set(json.load(f))

    rngs = [np.random.default_rng(SEED + i) for i in range(len(BUCKETS))]
    reservoirs = [[] for _ in BUCKETS]
    seen = [0] * len(BUCKETS)

    xz = subprocess.Popen(["xz", "-dc", "-T0", dump_path], stdout=subprocess.PIPE, bufsize=1 << 22)
    reader = csv.reader((l.decode("utf-8", "replace") for l in xz.stdout))
    next(reader)
    for i, row in enumerate(reader):
        if i % 200_000 == 0:
            print(f"{i:,} rows scanned, reservoirs {[len(r) for r in reservoirs]}", flush=True)
        try:
            parsed = orjson.loads(row[1])
        except orjson.JSONDecodeError:
            continue
        if not isinstance(parsed, list) or len(parsed) < 35:
            continue
        inputs, targets = split_profile(parsed, corpus)
        n_in = sum(1 for _, _, s in inputs if s != "plan_to_watch")
        if len(targets) < MIN_TARGETS:
            continue
        b = next((k for k, (lo, hi) in enumerate(BUCKETS) if lo <= n_in <= hi), None)
        if b is None:
            continue
        seen[b] += 1
        entry = (row[0], inputs, targets)
        if len(reservoirs[b]) < PER_BUCKET:
            reservoirs[b].append(entry)
        else:
            j = rngs[b].integers(0, seen[b])
            if j < PER_BUCKET:
                reservoirs[b][j] = entry

    out = {}
    for (lo, hi), res in zip(BUCKETS, reservoirs):
        label = f"t-{lo}-{hi}" if hi <= 10_000 else "t-300+"
        for username, inputs, targets in res:
            out[username] = {"bucket": label, "items": inputs, "targets": targets}
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"wrote {len(out)} profiles to {OUT} (eligible per bucket: {seen})")


if __name__ == "__main__":
    main()
