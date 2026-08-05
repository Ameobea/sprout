"""Per-entry updated_at extraction from the raw MAL dump, aligned to
user_input_vectors_cleanup_notrust: same gating as the cleanup vectorizer
(corpus filter, rated-PTW keep, on_hold skip, huge drop, >=20 entries), user
set/order taken from the census usernames file (npz-row-aligned). Emits
upd_sec int32 (seconds since 2000-01-01 UTC, -1 = missing). Run on host venv."""

import gzip
import io
import json
import sys
from datetime import date

import numpy as np
import pandas as pd
import zstandard

EPOCH_ORD = date(2000, 1, 1).toordinal()
STATUS_OK = {"completed", "watching", "on_hold", "dropped", "plan_to_watch"}


def main():
    src, names_path, metrics_path, corpus_path, npz_path, out_path = sys.argv[1:7]

    with gzip.open(names_path, "rt") as f:
        names = f.read().split("\n")
    order = {n: i for i, n in enumerate(names)}
    print(f"{len(names):,} target users", flush=True)

    with open(corpus_path) as f:
        ids = set(json.load(f))
    df = pd.read_csv(metrics_path, usecols=["username", "n_entries"])
    huge = set(df.username[df.n_entries > 2000])

    date_cache = {}

    def to_sec(ts):
        d = date_cache.get(ts[:10])
        if d is None:
            d = (date(int(ts[:4]), int(ts[5:7]), int(ts[8:10])).toordinal() - EPOCH_ORD) * 86400
            date_cache[ts[:10]] = d
        return d + int(ts[11:13]) * 3600 + int(ts[14:16]) * 60 + int(ts[17:19])

    per_user = {}
    n_rows = 0
    dctx = zstandard.ZstdDecompressor()
    with open(src, "rb") as fh, dctx.stream_reader(fh) as zr:
        txt = io.TextIOWrapper(zr, encoding="utf-8", newline="")
        next(txt)
        for line in txt:
            n_rows += 1
            if n_rows % 200_000 == 0:
                print(f"{n_rows:,} rows scanned, {len(per_user):,} matched", flush=True)
            tab = line.index("\t")
            username = line[:tab]
            if username not in order or username in huge or username in per_user:
                continue
            tab2 = line.rindex("\t")
            try:
                entries = json.loads(line[tab + 1 : tab2])
            except json.JSONDecodeError:
                continue
            secs = []
            for e in entries:
                ls = e.get("list_status") or {}
                st = ls.get("status")
                aid = (e.get("node") or {}).get("id")
                if aid is None or st is None or aid not in ids:
                    continue
                score = ls.get("score") or 0
                if st == "plan_to_watch" and not score > 0:
                    continue
                if st == "on_hold" and score == 0:
                    continue
                ts = ls.get("updated_at")
                try:
                    secs.append(to_sec(ts) if ts else -1)
                except ValueError:
                    secs.append(-1)
            if len(secs) >= 20:
                per_user[username] = np.array(secs, dtype=np.int32)

    print(f"scan done: {n_rows:,} rows, matched {len(per_user):,}/{len(names):,}", flush=True)

    vec = np.load(npz_path)
    ref = vec["lengths"]
    missing = [n for n in names if n not in per_user]
    print(f"missing users: {len(missing):,} (e.g. {missing[:5]})", flush=True)

    chunks, ok_users, bad_len = [], 0, 0
    for i, n in enumerate(names):
        arr = per_user.get(n)
        if arr is None or len(arr) != ref[i]:
            if arr is not None:
                bad_len += 1
            chunks.append(np.full(ref[i], -1, dtype=np.int32))
        else:
            ok_users += 1
            chunks.append(arr)
    upd = np.concatenate(chunks)
    print(f"aligned: ok {ok_users:,}  len-mismatch {bad_len:,}  missing {len(missing):,}", flush=True)
    print(f"entries {len(upd):,}  covered {(upd >= 0).mean():.4f}", flush=True)

    np.savez(out_path, upd_sec=upd)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
