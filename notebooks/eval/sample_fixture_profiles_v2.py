"""
One-time seeded sampler for fixtures v2: size-bucket x history-class stratified
profiles selected from the per-user metrics table, with frozen item snapshots
extracted from the raw December dump. Buckets labeled v2-{class}-{lo}-{hi}.

Classes (rated_span = days between first and last scored-entry update):
  burst: rated_span <= 7   long: rated_span > 365   mid: 30 < rated_span <= 365

Usage: sample_fixture_profiles_v2.py <metrics.csv> <dump.csv.xz>
"""

import csv
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import orjson
import pandas as pd

SEED = 20260730
PER_CELL = 25
SIZE_BUCKETS = [(10, 29), (30, 99), (100, 299), (300, 10_000_000)]
MIN_RECENT = date(2020, 8, 10).toordinal()
FIXTURES_DIR = Path(__file__).parent / "fixtures"
OUT = FIXTURES_DIR / "sampled_profiles_v2.json"

csv.field_size_limit(sys.maxsize)


def main():
    metrics_path, dump_path = sys.argv[1], sys.argv[2]

    exclude = {"ameo___", "snapsauce"}
    with open(FIXTURES_DIR / "sampled_profiles.json") as f:
        exclude |= set(json.load(f).keys())

    df = pd.read_csv(metrics_path)
    df["rated_span"] = (df.upd_rated_max - df.upd_rated_min).clip(lower=0)
    df = df[
        (df.n_entries >= 10) & (df.upd_max > MIN_RECENT)
        & (df.n_rated >= 10) & (df.upd_rated_max > 0)
        & ~df.username.isin(exclude) & df.username.notna()
    ]

    rng = np.random.default_rng(SEED)
    selected = {}
    for lo, hi in SIZE_BUCKETS:
        sb = df[(df.n_entries >= lo) & (df.n_entries <= hi)]
        for cls, mask in [
            ("long", sb.rated_span > 365),
            ("mid", (sb.rated_span > 30) & (sb.rated_span <= 365)),
            ("burst", sb.rated_span <= 7),
        ]:
            pool = sb[mask]
            label = f"v2-{cls}-{lo}-{hi}" if hi <= 10_000 else f"v2-{cls}-300+"
            n = min(PER_CELL, len(pool))
            if n < PER_CELL:
                print(f"WARN cell {label}: only {len(pool)} candidates")
            picks = pool.iloc[rng.choice(len(pool), size=n, replace=False)]
            for u in picks.username:
                selected[u] = label
            print(f"{label}: {n} from pool of {len(pool):,}")

    print(f"total selected: {len(selected)}")

    wanted = set(selected)
    out = {}
    xz = subprocess.Popen(["xz", "-dc", "-T0", dump_path], stdout=subprocess.PIPE, bufsize=1 << 22)
    reader = csv.reader((l.decode("utf-8", "replace") for l in xz.stdout))
    next(reader)
    for i, row in enumerate(reader):
        if row[0] in wanted:
            items = []
            for node in orjson.loads(row[1]):
                ls = node.get("list_status") or {}
                n = node.get("node") or {}
                if n.get("id") is None or ls.get("status") is None:
                    continue
                items.append((n["id"], ls.get("score") or 0, ls["status"]))
            out[row[0]] = {"bucket": selected[row[0]], "items": items}
            if len(out) == len(wanted):
                break
        if i % 200_000 == 0:
            print(f"scanned {i:,} rows, found {len(out)}/{len(wanted)}", flush=True)
    xz.terminate()

    missing = wanted - set(out)
    if missing:
        print(f"WARN: {len(missing)} usernames not found in dump: {sorted(missing)[:5]}")
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"wrote {len(out)} profiles to {OUT}")


if __name__ == "__main__":
    main()
