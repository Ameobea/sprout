"""
Streaming pass over the raw mal-user-animelists dump producing one row of
per-user metrics for data-quality analysis. Accepts the legacy dec2025
csv.xz archive or a dump-table.sh tsv.zst (header-mapped columns).

Usage: extract_profile_metrics.py <dump.{csv.xz,tsv.zst}> <out.csv> [corpus_ids.json]
"""

import csv
import subprocess
import sys
import multiprocessing as mp
from collections import Counter
from itertools import chain
from datetime import date
from math import sqrt

import orjson

csv.field_size_limit(sys.maxsize)

WATCHED = ("completed", "watching", "dropped")

_ix_by_id = {}
_date_memo = {}


def _init(corpus_path):
    global _ix_by_id
    with open(corpus_path, "rb") as f:
        _ix_by_id = {aid: i for i, aid in enumerate(orjson.loads(f.read()))}


def _day(ts):
    d = _date_memo.get(ts[:10])
    if d is None:
        try:
            d = date.fromisoformat(ts[:10]).toordinal()
        except ValueError:
            d = 0
        _date_memo[ts[:10]] = d
    return d


def process_row(username, raw_json):
    try:
        parsed = orjson.loads(raw_json)
    except orjson.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or not parsed:
        return None

    status_counts = Counter()
    scores = []
    upd_days = []
    upd_rated = []
    n_in_corpus = n_rated_in_corpus = n_model_input = n_model_rated = 0
    corpus_rank_sum = 0
    n_dates = n_nonptw = 0

    for node in parsed:
        ls = node.get("list_status") or {}
        n = node.get("node") or {}
        anime_id = n.get("id")
        status = ls.get("status")
        if anime_id is None or status is None:
            continue
        score = ls.get("score") or 0
        status_counts[status] += 1
        is_ptw = status == "plan_to_watch"

        if not is_ptw:
            n_nonptw += 1
            ts = ls.get("updated_at")
            if ts:
                d = _day(ts)
                if d:
                    upd_days.append(d)
                    if score > 0:
                        upd_rated.append(d)
            if ls.get("start_date") or ls.get("finish_date"):
                n_dates += 1

        if score > 0:
            scores.append(score)

        ix = _ix_by_id.get(anime_id)
        if ix is not None:
            n_in_corpus += 1
            corpus_rank_sum += ix
            if score > 0:
                n_rated_in_corpus += 1
            if not is_ptw and not (status == "on_hold" and score == 0):
                n_model_input += 1
                if score > 0:
                    n_model_rated += 1

    n_entries = sum(status_counts.values())
    if n_entries == 0:
        return None

    n_rated = len(scores)
    if n_rated:
        mean_s = sum(scores) / n_rated
        var = sum(s * s for s in scores) / n_rated - mean_s * mean_s
        std_s = sqrt(max(var, 0.0))
        mode_frac = Counter(scores).most_common(1)[0][1] / n_rated
        n_distinct = len(set(scores))
    else:
        mean_s = std_s = mode_frac = 0.0
        n_distinct = 0

    if upd_days:
        upd_days.sort()
        n_u = len(upd_days)
        p = lambda q: upd_days[int(q * (n_u - 1))]
        u_min, u_p10, u_p50, u_p90, u_max = upd_days[0], p(0.1), p(0.5), p(0.9), upd_days[-1]
        n_upd_days = 1 + sum(1 for a, b in zip(upd_days, upd_days[1:]) if b != a)
        frac_1d = sum(1 for d in upd_days if u_max - d <= 1) / n_u
        frac_7d = sum(1 for d in upd_days if u_max - d <= 7) / n_u
        frac_30d = sum(1 for d in upd_days if u_max - d <= 30) / n_u
    else:
        u_min = u_p10 = u_p50 = u_p90 = u_max = n_upd_days = 0
        frac_1d = frac_7d = frac_30d = 0.0

    return (
        username, n_entries,
        status_counts["completed"], status_counts["watching"], status_counts["on_hold"],
        status_counts["dropped"], status_counts["plan_to_watch"],
        n_rated, round(mean_s, 3), round(std_s, 3), n_distinct, round(mode_frac, 3),
        u_min, u_p10, u_p50, u_p90, u_max, n_upd_days,
        round(frac_1d, 4), round(frac_7d, 4), round(frac_30d, 4),
        upd_rated[0] if upd_rated else 0,
        max(upd_rated) if upd_rated else 0,
        n_in_corpus, n_rated_in_corpus, n_model_input, n_model_rated,
        round(corpus_rank_sum / n_in_corpus, 1) if n_in_corpus else -1,
        round(n_dates / n_nonptw, 4) if n_nonptw else 0.0,
    )


HEADER = [
    "username", "n_entries", "n_completed", "n_watching", "n_onhold", "n_dropped",
    "n_ptw", "n_rated", "mean_score", "std_score", "n_distinct_scores", "mode_frac",
    "upd_min", "upd_p10", "upd_p50", "upd_p90", "upd_max", "n_upd_days",
    "frac_1d", "frac_7d", "frac_30d", "upd_rated_min", "upd_rated_max",
    "n_in_corpus", "n_rated_in_corpus", "n_model_input", "n_model_rated",
    "mean_corpus_rank", "frac_dates",
]


def process_batch(rows):
    out = []
    bad = 0
    for username, raw in rows:
        r = process_row(username, raw)
        if r is None:
            bad += 1
        else:
            out.append(r)
    return out, bad


def batches(reader, size=100):
    next(reader)
    batch = []
    for row in reader:
        batch.append((row[0], row[1]))
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def open_dump(path):
    if path.endswith(".zst"):
        proc = subprocess.Popen(
            ["zstd", "-dc", "-T0", path], stdout=subprocess.PIPE, bufsize=1 << 22
        )
        lines = (l.decode("utf-8", "replace").rstrip("\n").split("\t") for l in proc.stdout)
        hdr = next(lines)
        ui, ji = hdr.index("username"), hdr.index("animelist_json")
        return proc, chain([hdr], ((r[ui], r[ji]) for r in lines))
    proc = subprocess.Popen(
        ["xz", "-dc", "-T0", path], stdout=subprocess.PIPE, bufsize=1 << 22
    )
    return proc, csv.reader((l.decode("utf-8", "replace") for l in proc.stdout))


def main():
    dump_path, out_path = sys.argv[1], sys.argv[2]
    corpus_path = sys.argv[3] if len(sys.argv) > 3 else "../../data/corpus_ids.json"

    xz, reader = open_dump(dump_path)

    n_users = n_bad = 0
    with open(out_path, "wt", newline="") as wf, mp.Pool(
        20, initializer=_init, initargs=(corpus_path,)
    ) as pool:
        writer = csv.writer(wf)
        writer.writerow(HEADER)
        for out, bad in pool.imap_unordered(process_batch, batches(reader), chunksize=1):
            writer.writerows(out)
            n_users += len(out)
            n_bad += bad
            if n_users % 100_000 < 100:
                print(f"{n_users} users processed ({n_bad} bad)", flush=True)

    xz.wait()
    print(f"done: {n_users} users, {n_bad} bad/empty rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
