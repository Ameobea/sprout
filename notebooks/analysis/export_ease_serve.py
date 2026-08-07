"""Export EASE serve artifacts for the Rust server: B (row-major f32 LE) + mu vector,
plus a graft forward golden (f64 reference) for the Rust parity test.

Run inside rocm_jax: cd /jax_dir/notebooks && python analysis/export_ease_serve.py
"""

import json
import sys
from pathlib import Path

import msgpack
import numpy as np

A = Path("../data/aug2026")
OUT = A / "serve"
OUT.mkdir(exist_ok=True)

LAM = 200.0
b_path = A / "ease_B6k_lam200.npy"
if b_path.exists():
    B = np.load(b_path).astype(np.float32)
else:
    G = np.load(A / "gram6k_aug2026.npz")["G"].astype(np.float64)
    G[np.diag_indices(6000)] += LAM
    P = np.linalg.inv(G)
    B = (-P / np.diag(P)[None, :]).astype(np.float32)
    np.fill_diagonal(B, 0.0)
    np.save(b_path, B)
    del G, P
assert B.shape == (6000, 6000)
B.tofile(OUT / "ease_B6k_lam200.f32bin")

d = np.load(A / "user_input_vectors_cleanup_notrust.npz")
indices = d["indices"].astype(np.int64)
lengths = d["lengths"].astype(np.int64)
starts = np.zeros(len(lengths), dtype=np.int64)
np.cumsum(lengths[:-1], out=starts[1:])
rng_ref = np.random.default_rng(555)
rng_h = np.random.default_rng(999)
perm = rng_h.permutation(len(lengths))
ref_users = rng_ref.choice(perm[len(lengths) // 10:], size=20000, replace=False)
s_sum = np.zeros(6000, dtype=np.float64)
for u in ref_users:
    s0, l = starts[u], lengths[u]
    s_sum += B[indices[s0 : s0 + l]].sum(axis=0)
mu = (s_sum / len(ref_users)).astype(np.float32)
mu.tofile(OUT / "ease_mu6k.f32bin")

counts = np.bincount(indices, minlength=6000).astype(np.float64)
with open(OUT / "train_counts_aug2026.json", "w") as f:
    json.dump(counts.tolist(), f)

# ---- graft forward golden (f64 reference, mirrors gen_golden.py conventions) ----
def load_params(path):
    with open(path, "rb") as f:
        obj = msgpack.unpackb(f.read(), raw=False, ext_hook=lambda c, d: (c, d))
    out = {}
    for name, node in obj.items():
        if not isinstance(node, dict) or "kernel" not in node:
            continue
        _, payload = node["kernel"]
        shape, dtype, data = msgpack.unpackb(payload, raw=False)
        kernel = np.frombuffer(data, dtype=dtype).reshape(shape).astype(np.float64)
        _, payload = node["bias"]
        shape, dtype, data = msgpack.unpackb(payload, raw=False)
        bias = np.frombuffer(data, dtype=dtype).reshape(shape).astype(np.float64)
        out[name] = (kernel, bias)
    return out


def swish(x):
    return x / (1.0 + np.exp(-x))


p = load_params(sys.argv[1] if len(sys.argv) > 1 else A / "probe/probe_graft_concat.msgpack")
Bd = B.astype(np.float64)

rng = np.random.default_rng(1234)
idxs = np.sort(rng.choice(6000, size=120, replace=False)).astype(np.int64)
vals = rng.normal(0, 1, size=120).round(3)

x = np.zeros(12000)
x[idxs] = 1.0
x[6000 + idxs] = vals
e = Bd[idxs].sum(axis=0)
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

golden = {
    "idxs": idxs.tolist(),
    "vals": vals.tolist(),
    "logits_head": logits[:64].tolist(),
    "ratings_head": ratings[:64].tolist(),
    "logits_sum": float(logits.sum()),
    "ratings_sum": float(ratings.sum()),
    "ease_head": e[:16].tolist(),
}
with open(OUT / "forward_golden_graft_concat.json", "w") as f:
    json.dump(golden, f)
print("exported: B 144MB f32, mu, train_counts, graft golden", flush=True)
