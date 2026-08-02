"""
Presence stats for specific shows in a raw animelists dump, with full popularity
distribution for percentile context.

Usage: show_presence.py <dump.tsv.zst> <out.json> <anime_id,anime_id,...>
"""

import subprocess
import sys
import multiprocessing as mp
from collections import Counter

import orjson

RECENT = "2026-07-01"

_targets = set()


def _init(targets):
    global _targets
    _targets = targets


def process_batch(rows):
    counts = Counter()
    rated = Counter()
    model_input = Counter()
    detail = {a: [0, 0, 0, 0, 0, 0, 0.0, 0] for a in _targets}
    n_users = 0
    for raw in rows:
        try:
            parsed = orjson.loads(raw)
        except orjson.JSONDecodeError:
            continue
        if not isinstance(parsed, list) or not parsed:
            continue
        n_users += 1
        for node in parsed:
            ls = node.get("list_status") or {}
            n = node.get("node") or {}
            aid = n.get("id")
            status = ls.get("status")
            if aid is None or status is None:
                continue
            score = ls.get("score") or 0
            counts[aid] += 1
            if score > 0:
                rated[aid] += 1
            if status != "plan_to_watch" and not (status == "on_hold" and score == 0):
                model_input[aid] += 1
            if aid in _targets:
                d = detail[aid]
                ix = ("completed", "watching", "on_hold", "dropped", "plan_to_watch").index(status)
                d[ix] += 1
                if score > 0:
                    d[5] += 1
                    d[6] += score
                if (ls.get("updated_at") or "") >= RECENT:
                    d[7] += 1
    return counts, rated, model_input, detail, n_users


def batches(path, size=1000):
    proc = subprocess.Popen(["zstd", "-dc", "-T0", path], stdout=subprocess.PIPE, bufsize=1 << 22)
    hdr = proc.stdout.readline().decode().rstrip("\n").split("\t")
    j_ix = hdr.index("animelist_json")
    batch = []
    for line in proc.stdout:
        batch.append(line.decode("utf-8", "replace").rstrip("\n").split("\t")[j_ix])
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
    proc.wait()


def main():
    dump_path, out_path = sys.argv[1], sys.argv[2]
    targets = frozenset(int(x) for x in sys.argv[3].split(","))

    counts, rated, model_input = Counter(), Counter(), Counter()
    detail = {a: [0, 0, 0, 0, 0, 0, 0.0, 0] for a in targets}
    n_users = 0
    with mp.Pool(16, initializer=_init, initargs=(targets,)) as pool:
        for c, r, m, d, nu in pool.imap_unordered(process_batch, batches(dump_path), chunksize=1):
            counts.update(c)
            rated.update(r)
            model_input.update(m)
            for a, row in d.items():
                cur = detail[a]
                for i in range(8):
                    cur[i] += row[i]
            n_users += nu
            if n_users % 100_000 < 1000:
                print(f"{n_users:,} users", flush=True)

    all_counts = sorted(counts.values(), reverse=True)
    out = {
        "n_users": n_users,
        "n_distinct_anime": len(counts),
        "total_entries": sum(all_counts),
        "top_counts": all_counts[:50],
        "count_percentiles": {p: all_counts[min(len(all_counts) - 1, len(all_counts) * p // 100)] for p in (1, 5, 10, 25, 50, 75, 90)},
        "targets": {
            a: {
                "total": counts.get(a, 0),
                "rank": 1 + sum(1 for v in counts.values() if v > counts.get(a, 0)),
                "rated": rated.get(a, 0),
                "model_input": model_input.get(a, 0),
                "completed": detail[a][0],
                "watching": detail[a][1],
                "on_hold": detail[a][2],
                "dropped": detail[a][3],
                "ptw": detail[a][4],
                "mean_score": round(detail[a][6] / detail[a][5], 3) if detail[a][5] else None,
                "updated_since_jul1": detail[a][7],
            }
            for a in sorted(targets)
        },
    }
    with open(out_path, "wb") as f:
        f.write(orjson.dumps(out, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS))
    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
