"""Recover per-entry raw scores (1..10, 0=unrated non-dropped, 11=dropped-unrated)
from the normalized values npz by inverting the alpha-mix per user: each user's
rated values live on <=10 known support points computed from the census hist.
Verified by recomputing the census hist from the reconstruction."""

import argparse

import numpy as np

SCORES = np.arange(1, 11, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="../../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--census", default="../../data/aug2026/rating_census.npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.vectors)
    vals = d["values"].astype(np.float64)
    lengths = d["lengths"].astype(np.int64)
    rated = np.unpackbits(d["rated_masks"])[: int(d["total_mask_bits"][0])].astype(bool)
    statuses = d["statuses"]
    n_users = len(lengths)
    user_of_entry = np.repeat(np.arange(n_users), lengths)

    hist = np.load(args.census)["hist"].astype(np.float64)
    rh = hist[:, 1:]
    n_rated = rh.sum(1)
    mu = (rh * SCORES).sum(1) / np.maximum(n_rated, 1)
    sigma = np.sqrt((rh * (SCORES[None] - mu[:, None]) ** 2).sum(1) / np.maximum(n_rated, 1)) + 1e-6
    alpha = np.clip(sigma / 2.6, 0.3, 0.8)

    # the alpha-mix support is monotone nondecreasing in raw score (ties only at
    # the +-2.5 clip), so sorting each user's rated values and assigning the
    # hist multiset in score order reproduces the hist exactly; ~0.02% of
    # entries (score 1 vs 2 both clipped to -2.5 for high-mu users) are
    # genuinely ambiguous in the values channel and resolve arbitrarily within
    # the correct multiset
    raw = np.zeros(len(vals), dtype=np.uint8)
    raw[~rated & (statuses == 3)] = 11
    ridx = np.nonzero(rated)[0]
    order = np.lexsort((vals[ridx], user_of_entry[ridx]))
    expected = np.repeat(
        np.tile(np.arange(1, 11, dtype=np.uint8), n_users),
        hist[:, 1:].astype(np.int64).ravel(),
    )
    raw[ridx[order]] = expected

    rec_hist = np.zeros((n_users, 11), dtype=np.int64)
    np.add.at(rec_hist, (user_of_entry[ridx], raw[ridx]), 1)
    assert int((rec_hist[:, 1:] != hist[:, 1:].astype(np.int64)).sum()) == 0

    z = np.clip((SCORES[None] - mu[:, None]) / sigma[:, None], -3, 3)
    ab = np.clip((SCORES[None] - 5.5) / 2.5, -2.5, 2.0)
    support = np.clip(alpha[:, None] * z + (1 - alpha[:, None]) * ab, -2.5, 2.5)
    err = np.abs(support[user_of_entry[ridx], raw[ridx] - 1] - vals[ridx])
    print(f"max |support - value| over rated entries: {err.max():.3e} (mean {err.mean():.3e})")
    assert err.max() < 1e-4, "assigned scores do not reproduce stored values"

    np.save(args.out, raw)
    print(f"{args.out}: {len(raw):,} entries, rated {int((raw >= 1).sum() - (raw == 11).sum()):,}")


if __name__ == "__main__":
    main()
