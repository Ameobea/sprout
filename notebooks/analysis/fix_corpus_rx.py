"""
Rebuild corpus_ids_aug2026.json with proper rx exclusions.

The July metadata-table backfill stored JSON without the `rating` field, so the
metadata notebook's rx exclusion silently no-oped (18 exclusions vs dec's 1597).
Rating sources, in precedence order: fresh metadata dump (rows that have it),
dec processed-metadata.csv, live MAL API for corpus-margin ids covered by neither.
Also patches the rating column of processed-metadata_aug2026.csv in place.
"""

import csv
import gzip
import json
import subprocess
import sys
import time
import urllib.request
from collections import Counter

csv.field_size_limit(sys.maxsize)

DATA = "../../data"
MARGIN = 7000


def mal_rating(aid, cid):
    req = urllib.request.Request(
        f"https://api.myanimelist.net/v2/anime/{aid}?nsfw=true&fields=rating",
        headers={"X-MAL-CLIENT-ID": cid},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r).get("rating") or "Unknown"
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "Unknown"
        raise


def main():
    counts = Counter()
    with gzip.open(f"{DATA}/collected_animelists_aug2026.csv.gz", "rt", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            counts[int(row[1])] += 1
    print(f"counted {len(counts):,} distinct anime", flush=True)

    rating_by_id = {}
    with open(f"{DATA}/processed-metadata.csv") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[9] and row[9] != "Unknown":
                rating_by_id[int(row[0])] = row[9]
    n_dec = len(rating_by_id)

    p = subprocess.Popen(["zstd", "-dc", f"{DATA}/anime-metadata-aug2026.tsv.zst"], stdout=subprocess.PIPE)
    hdr = p.stdout.readline().decode().rstrip("\n").split("\t")
    id_ix, m_ix = hdr.index("id"), hdr.index("metadata")
    for line in p.stdout:
        row = line.decode("utf-8", "replace").rstrip("\n").split("\t")
        try:
            meta = json.loads(row[m_ix])
        except Exception:
            continue
        if isinstance(meta, dict) and meta.get("rating"):
            rating_by_id[int(row[id_ix])] = meta["rating"]
    p.wait()
    print(f"ratings known: dec={n_dec:,} +fresh -> {len(rating_by_id):,}", flush=True)

    candidates = [a for a, _ in counts.most_common(MARGIN)]
    uncovered = [a for a in candidates if a not in rating_by_id]
    print(f"corpus-margin ids missing rating: {len(uncovered)}", flush=True)

    with open("../../fleet/fleet.env") as f:
        cid = next(l for l in f if l.startswith("MAL_COLLECTOR_CLIENT_ID")).strip().split("=", 1)[1]
    for i, aid in enumerate(uncovered, 1):
        rating_by_id[aid] = mal_rating(aid, cid)
        if i % 50 == 0:
            print(f"fetched {i}/{len(uncovered)}", flush=True)
        time.sleep(1.1)

    rx = {a for a, r in rating_by_id.items() if r.lower() == "rx"}
    corpus = [a for a, _ in counts.most_common(len(counts)) if a not in rx][:6000]
    with open(f"{DATA}/corpus_ids_aug2026.json", "wt") as f:
        json.dump(corpus, f)
    excluded_from_margin = sum(1 for a in candidates if a in rx)
    print(f"rx known total: {len(rx)}  rx in top-{MARGIN}: {excluded_from_margin}", flush=True)

    rows = []
    with open(f"{DATA}/processed-metadata_aug2026.csv") as f:
        reader = csv.reader(f)
        rows.append(next(reader))
        for row in reader:
            known = rating_by_id.get(int(row[0]))
            if known and (not row[9] or row[9] == "Unknown"):
                row[9] = known
            rows.append(row)
    with open(f"{DATA}/processed-metadata_aug2026.csv", "wt", newline="") as f:
        csv.writer(f).writerows(rows)
    print("done: corpus_ids_aug2026.json rebuilt, processed-metadata_aug2026.csv ratings patched", flush=True)


if __name__ == "__main__":
    main()
