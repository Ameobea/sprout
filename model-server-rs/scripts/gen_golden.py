"""Regenerate forward-pass golden fixtures for a weights msgpack.

Usage: gen_golden.py <weights.msgpack> <out.json>
f64 numpy reference matching src/refimpl.rs; synthetic seeded profiles.
"""

import json
import sys

import msgpack
import numpy as np

CORPUS = 6000
LAYERS = [
    "Dense_0", "bottleneck", "dec_item_up1", "dec_item_up2", "item_logits",
    "dec_rating_up1", "dec_rating_up2", "rating_pred",
]


def load_params(path):
    with open(path, "rb") as f:
        root = msgpack.unpack(f, strict_map_key=False)

    def arr(v):
        shape, dtype, buf = msgpack.unpackb(v.data)
        assert dtype == "float32"
        return np.frombuffer(buf, dtype=np.float32).reshape(shape).astype(np.float64)

    return {name: (arr(root[name]["kernel"]), arr(root[name]["bias"])) for name in LAYERS}


def swish(x):
    return x / (1.0 + np.exp(-x))


def forward(p, x):
    h = swish(x @ p["Dense_0"][0] + p["Dense_0"][1])
    z = h @ p["bottleneck"][0] + p["bottleneck"][1]
    d1 = swish(z @ p["dec_item_up1"][0] + p["dec_item_up1"][1])
    d1b = swish(d1 @ p["dec_item_up2"][0] + p["dec_item_up2"][1])
    logits = d1b @ p["item_logits"][0] + p["item_logits"][1]
    d2 = swish(z @ p["dec_rating_up1"][0] + p["dec_rating_up1"][1])
    d2b = swish(d2 @ p["dec_rating_up2"][0] + p["dec_rating_up2"][1])
    ratings = d2b @ p["rating_pred"][0] + p["rating_pred"][1]
    return logits, ratings


def main():
    weights_path, out_path = sys.argv[1], sys.argv[2]
    p = load_params(weights_path)
    rng = np.random.default_rng(20260802)
    cases = []
    for n in (1, 3, 17, 120):
        idxs = np.sort(rng.choice(CORPUS, size=n, replace=False)).astype(np.int64)
        vals = np.round(rng.uniform(-2.0, 2.0, size=n), 3)
        x = np.zeros(CORPUS * 2)
        x[idxs] = 1.0
        x[CORPUS + idxs] = vals
        logits, ratings = forward(p, x)
        cases.append({
            "idxs": idxs.tolist(),
            "vals": vals.tolist(),
            "logits": np.round(logits, 6).tolist(),
            "ratings": np.round(ratings, 6).tolist(),
        })
    with open(out_path, "wt") as f:
        json.dump(cases, f)
    print(f"wrote {len(cases)} cases -> {out_path}")


if __name__ == "__main__":
    main()
