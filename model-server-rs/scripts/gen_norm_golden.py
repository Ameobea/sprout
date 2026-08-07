"""Regenerate testdata/norm_golden.json — the rating-normalization parity fixture.

Usage: gen_norm_golden.py <out.json>
Reference is notebooks/normalize_ratings.py; src/norm.rs must match it.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
from normalize_ratings import normalize_ratings

PROFILES = [
    [8, 7, 10, 3, 0, 0, 6, -2, 9, 5],
    [7, 7, 7, 7, 7],
    [0, 0, 0, -2, 0],
    [10, 1],
    [5],
    [],
    [0, 0, 8],
]


def main():
    cases = []
    for scores in PROFILES:
        normed, stats = normalize_ratings(np.array(scores, dtype=np.float32))
        cases.append({
            "scores": scores,
            "normed": np.asarray(normed).tolist(),
            "mu": stats["mu"],
            "sigma": stats["sigma"],
            "alpha": stats["alpha"],
            "zscore": np.asarray(stats["zscore_norm"]).tolist(),
            "absolute": np.asarray(stats["absolute_norm"]).tolist(),
        })
    with open(sys.argv[1], "wt") as f:
        json.dump(cases, f)
    print(f"wrote {len(cases)} cases -> {sys.argv[1]}")


if __name__ == "__main__":
    main()
