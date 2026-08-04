# info_audit artifact sources

Durable copy of the research-artifact template + build pipeline (canonical build runs from the
session scratchpad; published URL: claude.ai/code/artifact/f6534f4c-6cf3-4136-99fb-1ba315c140a8).

- `info_audit.html` — template (`__DATA__` placeholder gets the JSON blob)
- `build_artifact.py` — full DATA assembly from analysis outputs (rounds 1-4) + build;
  reads `data/aug2026/**` result JSONs and the local *.json snapshots here
- `artifact_data.json.bak` — frozen round-1 heavy data (decomposition/twins/EASE sweeps)
- `round2_data.json`, `cumshare.json`, `ease_lift_eyeballs.json` — round-2/3 snapshots

To update: edit template/build script, run `python3 build_artifact.py`, publish
`info_audit_built.html` via the Artifact tool against the URL above. Keep this dir in sync with
whatever scratchpad copy was actually published.
