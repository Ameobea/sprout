"""Generates golden test vectors for the Rust engine using f64 numpy math.

Usage: python3 gen_golden.py ../data/jax_model.msgpack testdata/
"""
import json
import sys

import msgpack
import numpy as np


def load_params(path):
    with open(path, "rb") as f:
        obj = msgpack.unpackb(f.read(), raw=False, ext_hook=lambda c, d: (c, d))
    out = {}
    for name, node in obj.items():
        if not isinstance(node, dict):
            continue
        code, payload = node["kernel"]
        shape, dtype, data = msgpack.unpackb(payload, raw=False)
        kernel = np.frombuffer(data, dtype=dtype).reshape(shape).astype(np.float64)
        code, payload = node["bias"]
        shape, dtype, data = msgpack.unpackb(payload, raw=False)
        bias = np.frombuffer(data, dtype=dtype).reshape(shape).astype(np.float64)
        out[name] = (kernel, bias)
    return out


def swish(x):
    return x / (1.0 + np.exp(-x))


def forward(p, x):
    h = swish(x @ p["Dense_0"][0] + p["Dense_0"][1])
    z = h @ p["bottleneck"][0] + p["bottleneck"][1]
    d1 = swish(z @ p["dec_item_up1"][0] + p["dec_item_up1"][1])
    d1 = swish(d1 @ p["dec_item_up2"][0] + p["dec_item_up2"][1])
    logits = d1 @ p["item_logits"][0] + p["item_logits"][1]
    d2 = swish(z @ p["dec_rating_up1"][0] + p["dec_rating_up1"][1])
    d2 = swish(d2 @ p["dec_rating_up2"][0] + p["dec_rating_up2"][1])
    ratings = d2 @ p["rating_pred"][0] + p["rating_pred"][1]
    return logits, ratings


def main():
    model_path, out_dir = sys.argv[1], sys.argv[2]
    p = load_params(model_path)
    corpus = 6000
    rng = np.random.default_rng(1234)

    cases = []
    for n in [1, 3, 17, 120]:
        idxs = np.sort(rng.choice(corpus, size=n, replace=False))
        vals = np.round(rng.uniform(-2.0, 2.0, size=n), 3)
        x = np.zeros(corpus * 2)
        x[idxs] = 1.0
        x[corpus + idxs] = vals
        logits, ratings = forward(p, x)
        cases.append(
            {
                "idxs": idxs.tolist(),
                "vals": vals.tolist(),
                "logits": np.round(logits, 6).tolist(),
                "ratings": np.round(ratings, 6).tolist(),
            }
        )
    with open(f"{out_dir}/forward_golden.json", "w") as f:
        json.dump(cases, f)

    norm_cases = []
    profiles = [
        [8, 7, 10, 3, 0, 0, 6, -2, 9, 5],
        [7, 7, 7, 7, 7],
        [0, 0, 0, -2, 0],
        [10, 1],
        [5],
        [],
        [0, 0, 8],
    ]
    sys.path.insert(0, "../notebooks")
    from normalize_ratings import normalize_ratings

    for scores in profiles:
        normed, stats = normalize_ratings(np.array(scores, dtype=np.float32))
        norm_cases.append(
            {
                "scores": scores,
                "normed": np.asarray(normed).tolist(),
                "mu": stats["mu"],
                "sigma": stats["sigma"],
                "alpha": stats["alpha"],
                "zscore": np.asarray(stats["zscore_norm"]).tolist(),
                "absolute": np.asarray(stats["absolute_norm"]).tolist(),
            }
        )
    with open(f"{out_dir}/norm_golden.json", "w") as f:
        json.dump(norm_cases, f)

    print("golden files written")


if __name__ == "__main__":
    main()
