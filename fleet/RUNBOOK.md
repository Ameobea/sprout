# MAL Data Collection Run — Runbook

> Live status of the current run: `RUN_STATE.md` at the repo root.

## Architecture

- **Username discovery**: Cloud Run service `anime-atlas` (us-west1) scrapes MAL's recently-online users page via `POST /collect-usernames` and forwards results to `https://anime.ameo.dev/add-usernames`. Driven by a curl loop (~4.5s interval).
- **Profile collection**: fleet of OVH public cloud `d2-2` VPSes (US-WEST-OR-1), one dedicated IP each. Each runs a small node agent (`fleet/agent/`) under systemd that loops: pull username from `/next-username-to-collect` → fetch list from the MAL API using the VPS's IP → `POST /submit-collected-list`. Sleeps `PACE_MS` (default 1200ms) after each request completes — the rate-limit-safe baseline; idles 60s when the queue drains or on errors.
- **No direct DB access from any remote machine.** All DB work happens in the anime.ameo.dev SvelteKit app (same box as MariaDB). Do not punch 3306 holes in the firewall.
- **Queue**: `anime-atlas`.`usernames-to-collect`. `collected` / `collected-manga` columns hold 0 (pending) or the HTTP-ish result status (200 success, 403 private, 404 gone, ...).
- **Metrics**: `GET https://anime.ameo.dev/metrics?token=...` publishes `anime_atlas_usernames_by_collected_total{collected="..."}` gauges. Prometheus on nelebrie-2 scrapes it via the `mal-collector` job (formerly pointed at a long-dead exporter on localhost:4484). Grafana: https://ameo.dev/grafana/
  - Collection rate PromQL: `sum(rate(anime_atlas_usernames_by_collected_total{collected!="0"}[$__rate_interval]))`

## Prerequisites

- `fleet/fleet.env` — OpenStack creds + the dedicated batch-scraping MAL OAuth app (gitignored). Region/project pinned there. Collectors must use the batch MAL creds, never the prod app's.
- Repo root `.env` — admin token + MySQL creds for `requeue`; `fleet.sh` pulls from both files when generating `agent.env`.
- `ssh debian@ameo.dev` working (used for `requeue`).
- OVH quota (US-WEST-OR-1): 40 cores / 20 instances → up to 20 collectors.

## Run procedure

1. **Deploy latest anime.ameo.dev** (`just deploy`) so collection endpoints + `/metrics` are live. Verify: `curl "https://anime.ameo.dev/metrics?token=$TOKEN"`.
2. **Update Cloud Run** if the username collector changed: `just deploy-gcs`.
3. **Start username discovery** (optional, runs alongside):
   `while true; do curl -X POST "https://anime-atlas-<id>.us-west1.run.app/collect-usernames?token=$TOKEN" && sleep 4.5 && echo ""; done`
4. **Requeue for re-collection** (flips previously-collected rows back to pending):
   `./fleet.sh requeue anime 200` — prompts with current counts first. Add statuses (e.g. `"200,500"`) to also retry errors.
5. **Smoke test with one instance**: `./fleet.sh spawn 1`, then `./fleet.sh status` and watch a few successful collections (`ssh -i fleet_key debian@<ip> journalctl -u mal-collector -f`).
6. **Scale up**: `./fleet.sh spawn 7` (or more; numbering continues automatically).
7. **Monitor** via Grafana dashboard / `./fleet.sh status`. Expected rate: ~0.83 profiles/sec per instance at 1200ms pacing → 8 instances ≈ 575k profiles/day; ~1.9M public profiles ≈ 3.5 days.
8. **Drain + teardown**: when `collected=0` count reaches ~0 (agents log "Queue drained; idling"), run `./fleet.sh destroy`. Instances are hourly-billed — don't leave them up.

## Fleet commands

See `./fleet.sh` (no args for usage). Notable:
- `spawn [n]` — boots Debian 12 d2-2s with cloud-init (installs node + systemd unit), pushes `agent.cjs` + `agent.env`, starts collectors.
- `stop` / `start` — pause/resume the whole fleet.
- `provision <ip>` — re-push a rebuilt agent (`just build-agent`) to an existing host.
- Agent config knobs (edit `gen_agent_env` in fleet.sh or agent.env on hosts): `COLLECTION_TYPE=anime|manga`, `PACE_MS`.

## Manga runs

Same flow with `COLLECTION_TYPE=manga` agents and `./fleet.sh requeue manga 200`. Note: `collected-manga` has no index (full scans) and `/metrics` currently only exports the anime column — add both if doing a serious manga run.

## After the run: preprocessing + training

Once the queue drains and the fleet is destroyed, the data pipeline continues in `notebooks/README.md` (dump → filter → metadata → vectorize) and `notebooks/README_training.md` (rocm/jax training). Notes:
- Dumps are server-side + zstd via `notebooks/dump-table.sh` — never pull raw table data over the tunnel.
- Jupyter env: `just launch-jupyter` (quay.io/jupyter/scipy-notebook, repo mounted at `/home/jovyan/work`, DB via local tunnel).
- Run `docker system prune -f` before big downloads if disk is tight (reclaimed 81GB on 2026-07-30).
- The December-2025 artifacts in `data/` (`collected_animelists.csv.gz`, `mal-user-animelists.csv.xz`, `user_input_vectors.npz`, `jax_model.msgpack`) are the ONLY copies of the deployed model's training data + weights — the DB gets overwritten by each new collection run. Keep them (compressed) for baseline evals and filter-tuning re-runs.

## Troubleshooting

- Agent errors on every iteration → check the main server is up and `BODY_SIZE_LIMIT=16M` is set on the anime-zoo deployment (oversized submits 413 otherwise).
- MAL 429s → agents honor Retry-After automatically; sustained 429s mean pacing is too aggressive, raise `PACE_MS`.
- A username stuck at 500 → transient MAL/agent failure; requeue with `"500"` at the end of the run.
- Instance unreachable → `./fleet.sh ls` to refresh IPs; OVH occasionally reassigns on reboot.
