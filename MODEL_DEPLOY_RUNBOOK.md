# Model deploy runbook

Checklist for shipping a new model version. Data pipeline upstream of this:
`notebooks/README.md` (dump → process → metadata → corpus → vectorize) and
`notebooks/README_training.md` (training). Written 2026-08-02 after the
`2026-logq` migration; update it when the process changes.

## Artifact naming convention

Canonical names always mean "current model"; the webapp Dockerfile never changes:
- `data/processed-metadata.csv`, `data/projected_model_embedding.json`

Every version keeps tagged copies (`processed-metadata-dec2025.csv`,
`corpus_ids_aug2026.json`, ...). The model-server Dockerfile + `models.json` pin
tagged files per model entry — that's where per-release churn belongs. A legacy
(dec2025-era) entry MUST pin its own metadata: its niche_boost popularity vector
derives from metadata `rating_count`, so rolling metadata would silently change
its serving behavior.

Pipeline writes tagged outputs; promotion = `cp` over the canonical name.

## Checklist

1. **Eval gates**: head-to-head vs prior model (LOO both-directions
   `--restrict-corpus`, temporal), run2 confirmation, `--bf16-sim` LOO
   (prod numerics). Judge against WORKSTREAMS §4 noise floors.
2. **Serve-prior artifact** (logq models): training-set counts from the model's
   own training npz → `train_counts_<name>.json` (see RUN_STATE; must byte-match
   training — never metadata rating_count).
3. **Model registry**: add entry to `model-server-rs/deploy/models.json` +
   Dockerfile.model_server_rs COPY lines + dockerignore allowlist. `serving:
   "logq"` for lift-trained models (α,k knob path), `"legacy"` for dec2025-era.
4. **Goldens**: `model-server-rs/scripts/gen_golden.py <weights> <fixture>`, then
   `MODEL_PATH=... GOLDEN_PATH=... cargo test`. Old models' goldens must still pass.
5. **Atlas embedding** (two steps, container-bound):
   - `extract_embeddings.py <weights> <embeddings.npy>` (rocm container)
   - `gen_atlas_embedding.py <embeddings.npy> <corpus> <metadata> <out.json>`
     (jupyter container; needs umap-learn)
   **Schema trap**: the webapp needs the emblaze-era `{points: {idx: {x,y}},
   ids: [...], neighbors: {...}}` shape (`src/embedding.ts` RawEmbedding).
   `project_model_embedding.ipynb`'s export cell emits a flat array — feeding
   that to prod crashes the app at boot (learned 2026-08-02). Use
   `gen_atlas_embedding.py`, which encodes the correct schema. Historical note:
   emblaze produced this format originally; prod has been down to ONE embedding
   (`EmbeddingName.Model`) since the 2025 rewrite.
6. **Promote canonicals**: metadata CSV + embedding json (`cp` over canonical).
7. **Webapp**: new `ModelName` entry ("Mon. YYYY" display), `DEFAULT_MODEL_NAME`,
   per-model knob default (`getDefaultNicheBoostFactor`), `just deploy`.
8. **Deploy order**: model-server image first (old webapp keeps working — server
   default stays compatible), then webapp.
9. **Post-deploy**: Grafana anime-model-server dashboard — per-model panels
   populating; `unknown_model_total` flat; spot-check the site on the new default.
10. **Archive**: tar the training inputs + weights (see `data/aug2026/MANIFEST.md`
    for the shape), cloud-sync. The DB gets overwritten by the next collection
    run — the archive is the only copy.

## Metadata DB gotchas (2026-08-02 incident)

- The `anime-metadata` table is filled ONLY via the app's fetch path
  (`fetchAnimeFromMALAPI` — full fields list incl. `rating`, `nsfw=true`).
  Never seed it with ad-hoc scripts: the July 2026 backfill used a hand-rolled
  fields list without `rating`, which silently no-oped the corpus rx-exclusion
  (93 rx titles reached a corpus candidate before being caught).
- Corpus builds must sanity-check the rx-exclusion count against the prior
  cycle (~1.6k expected, not ~tens).
- In-place refresh paths (no NULL window, safe for prod):
  `POST /refreshMetadata` (4 stale rows per call, cron-driven) and
  `POST /refresh-all-metadata` (full-table walk @1.2s, ~10h for 30k rows; runs
  inside the app process — a redeploy kills it, re-POST to restart).
  Do NOT mass-NULL metadata rows: recommendation hydration fires parallel MAL
  fetches for missing rows (429 risk + latency) until they self-heal.

## Server metrics gotcha

foundations metrics bind the service name at first touch: never call any metric
before `telemetry::init` in serve.rs, or the whole registry exports as
`undefined_*` and every dashboard goes dark (shipped once, 2026-08-02).
