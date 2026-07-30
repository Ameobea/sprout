# model-server-rs

Bespoke Rust inference engine for the anime recommendation model. Drop-in
API-compatible replacement for `notebooks/model_server.py` (Jax/FastAPI):
same endpoints (`/recommend`, `/health`, `/corpus`, `/cache/stats`,
`/cache/clear`), same request/response JSON, same env vars, loads the same
`jax_model.msgpack` flax checkpoint.

Tuned specifically for the Zen 4 deployment target (EPYC 4344P) with
`-C target-cpu=znver4`; requires AVX-512.

## Design

- **Pre-packed weights**: decoder weight matrices are packed once at load into
  panel-major layout (NR=32 columns per panel), so GEMM streams B linearly with
  zero per-call packing. C accumulators live in registers across the full K loop
  (no K blocking) with bias+swish fused into the store epilogue.
- **Sparse encoder**: model input is 12000-dim but only 2n entries are nonzero
  for an n-item profile; the input GEMV collapses to a gather-sum of weight rows.
- **Rank-1 holdout**: each ablation row = full profile minus one item, so holdout
  encoder pre-activations are two vector subtractions from the full-profile
  pre-activation. The (n+1)-row batch (n holdouts + baseline) runs in one pass.
- **8×32 AVX-512 f32 microkernel** (16 zmm accumulators), 8 threads pinned to
  physical cores, spin-then-park fork-join pool (~µs barriers). ~1.07 TFLOP/s
  sustained on the EPYC 4344P (≈85% of its 8-core f32 peak).
- **bf16 mode** (`PRECISION=bf16`): weights stored bf16, `VDPBF16PS` kernels,
  ~2.0 TFLOP/s. ~2x faster at low batch; top-10 recommendations 99.2% identical
  to f32, predicted ratings within ±0.005. Contributor/impact orderings can flip
  among near-ties (~21% of contributor lists differ somewhere).
- Vectorized softmax/pow/swish (AVX-512 polynomial exp/log2), gathered scoring
  for holdout rows to avoid materializing full 6000-wide score arrays.

Parity vs. the live Jax server: recommendation lists byte-identical in order on
all tested profiles (real + synthetic + edge cases), scores within ~1e-6.
One deliberate quirk kept: the Python server ignores `use_alt_ranking` inside
profile-holdout scoring (it never forwards the flag); this is replicated.

## Env

`MODEL_PATH`, `CORPUS_PATH`, `METADATA_PATH`, `PORT`, `METRICS_PORT` (5709),
`INFER_THREADS` (8), `PIN_CORES` (1), `PRECISION` (`f32`|`bf16`).

Prometheus metrics (via `foundations`) are served on `METRICS_PORT` at
`/metrics`: request rates/latencies, inference + queue-wait timings, profile
size and response size distributions, cache utilization, errors.

## Bench / test

```
cargo test --release                 # golden tests vs numpy f64 + holdout consistency
cargo run --release --bin bench -- --threads 8 --nr 32 --mr 8 --pin [--bf16]
cargo build --release --features bench-compare   # adds faer comparison to bench
python3 gen_golden.py ../data/jax_model.msgpack testdata/   # regenerate goldens
```

Deploy: `just deploy-model-server` (phost-managed; builds the Docker image,
ships it, and health-check-swaps the `anime-model-server-rs` container on 5708
with automatic rollback to the previous image).
