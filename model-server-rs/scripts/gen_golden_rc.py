"""Forward golden for 3-channel + concat-graft (RC) checkpoints: f64 numpy
reference over [presence | z-mix | abs] input with ease = znorm(presence @ B).

Usage: gen_golden_rc.py <weights.msgpack> <ease_B.npy-or-f32bin> <out.json>
"""

import json
import sys

import msgpack
import numpy as np

CORPUS = 6000
LAYERS = [
    "Dense_0", "bottleneck", "ease_proj", "dec_item_up1", "dec_item_up2", "item_logits",
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


def main():
    weights_path, b_path, out_path = sys.argv[1:4]
    p = load_params(weights_path)
    assert p["Dense_0"][0].shape[0] == CORPUS * 3, "expected 3-channel checkpoint"
    if b_path.endswith(".npy"):
        B = np.load(b_path).astype(np.float64)
    else:
        B = np.fromfile(b_path, dtype=np.float32).reshape(CORPUS, CORPUS).astype(np.float64)

    rng = np.random.default_rng(20260806)
    cases = []
    for n in (1, 3, 17, 120):
        idxs = np.sort(rng.choice(CORPUS, size=n, replace=False)).astype(np.int64)
        raw = rng.integers(1, 11, size=n).astype(np.float64)
        raw[rng.random(n) < 0.3] = 0.0
        vals = np.round(rng.uniform(-2.0, 2.0, size=n), 3)
        absv = np.where(raw > 0, (raw - 5.5) / 2.5, 0.0)

        x = np.zeros(CORPUS * 3)
        x[idxs] = 1.0
        x[CORPUS + idxs] = vals
        x[2 * CORPUS + idxs] = absv
        e = B[idxs].sum(axis=0)
        ez = (e - e.mean()) / (e.std() + 1e-6)

        h = swish(x @ p["Dense_0"][0] + p["Dense_0"][1])
        z = h @ p["bottleneck"][0] + p["bottleneck"][1]
        ep = swish(ez @ p["ease_proj"][0] + p["ease_proj"][1])
        zc = np.concatenate([z, ep])
        d1 = swish(zc @ p["dec_item_up1"][0] + p["dec_item_up1"][1])
        d1 = swish(d1 @ p["dec_item_up2"][0] + p["dec_item_up2"][1])
        logits = d1 @ p["item_logits"][0] + p["item_logits"][1]
        d2 = swish(z @ p["dec_rating_up1"][0] + p["dec_rating_up1"][1])
        d2 = swish(d2 @ p["dec_rating_up2"][0] + p["dec_rating_up2"][1])
        ratings = d2 @ p["rating_pred"][0] + p["rating_pred"][1]

        cases.append({
            "idxs": idxs.tolist(),
            "vals": vals.tolist(),
            "abs": absv.tolist(),
            "logits_head": logits[:64].tolist(),
            "ratings_head": ratings[:64].tolist(),
            "logits_sum": float(logits.sum()),
            "ratings_sum": float(ratings.sum()),
        })
    with open(out_path, "wt") as f:
        json.dump(cases, f)
    print(f"wrote {len(cases)} RC cases -> {out_path}")


if __name__ == "__main__":
    main()
