"""Export rating-stack serve artifacts: residual-EASE B + shrunk item means as
raw f32 LE for the Rust server, plus a stack golden (f64 reference) covering
blend + era debias for the parity test.
Run inside rocm_jax: cd /jax_dir/notebooks && python analysis/export_rating_serve.py"""

import json
from pathlib import Path

import numpy as np

A = Path("../data/aug2026")
OUT = A / "serve"

B = np.load(A / "rating_resid_B6k.npy").astype(np.float32)
assert B.shape == (6000, 6000) and np.abs(np.diag(B)).max() == 0.0
B.tofile(OUT / "rating_resid_B6k.f32bin")
imean = np.load(A / "rating_item_prior_lam50.npy").astype(np.float32)
imean.tofile(OUT / "rating_imean6k.f32bin")

rng = np.random.default_rng(4321)
n = 40
idxs = np.sort(rng.choice(6000, size=n, replace=False))
raw = rng.integers(1, 11, size=n).astype(np.float64)
raw[5] = 0.0
upd = np.sort(rng.integers(1_400_000_000, 1_750_000_000, size=n))
upd[7] = 0
rng.shuffle(upd)

rated = raw > 0
mu = raw[rated].mean()
sigma = np.sqrt(((raw[rated] - mu) ** 2).mean()) + 1e-6
scores = np.where(raw == 0, mu, raw)
z = np.clip((scores - mu) / sigma, -3, 3)
absn = np.clip((scores - 5.5) / 2.5, -2.5, 2.0)
alpha = np.clip(sigma / 2.6, 0.3, 0.8)
normed = np.clip(alpha * z + (1 - alpha) * absn, -2.5, 2.5)

Bd = B.astype(np.float64)
r = np.where(rated, normed - imean[idxs].astype(np.float64), 0.0)
S = (r[None, :] @ Bd[idxs]).ravel()
w = 0.5 * min(sigma, 1.0)

base = rng.normal(0, 0.5, size=6000).round(4)
blended = base.copy()
blended += w * S

dated = rated & (upd > 0)
order = np.argsort(upd[dated], kind="stable")
ranks = np.empty(dated.sum())
ranks[order] = np.arange(1, dated.sum() + 1)
eras = ranks / dated.sum()
errs = blended[idxs[dated]] - normed[dated]
em = eras.mean()
ec = eras - em
slope = (ec * errs).sum() / (ec * ec).sum() * dated.sum() / (dated.sum() + 30.0)
corr_now = slope * (1.0 - em)
final = blended - corr_now

golden = {
    "idxs": idxs.tolist(),
    "raw": raw.tolist(),
    "updated_at": upd.tolist(),
    "base_ratings": base.tolist(),
    "sigma": float(sigma),
    "w": float(w),
    "era_slope": float(slope),
    "era_corr_now": float(corr_now),
    "scores_head": S[:16].tolist(),
    "final_head": final[:16].tolist(),
    "final_sum": float(final.sum()),
}
with open(OUT / "rating_stack_golden.json", "w") as f:
    json.dump(golden, f)
print(f"exported rating B f32bin ({B.nbytes // 2**20}MB), imean, stack golden (w={w:.4f}, slope={slope:.4f})")
