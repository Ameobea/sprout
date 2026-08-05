"""Assemble DATA for the HP-sweep artifact and build it."""

import json
from pathlib import Path

SP = Path(__file__).parent
A = Path("/home/casey/anime-atlas/data/aug2026")

RUNS = [
    ("control (adam+plateau 3e-4)", "rating_floors_control.json", "probe/probe_frac1.0", "baseline", 50000, 22),
    ("control seed 2", "rating_floors_seed2.json", "probe/probe_b512_seed2", "baseline", 50000, 22),
    ("adam+cosine 3e-4", "rating_floors_hp_cosine_lr3e4.json", "probe/hp_cosine_lr3e4", "win-small", 50000, 22),
    ("adam+cosine 3e-4 seed 2", "rating_floors_hp_cosine_lr3e4_s2.json", "probe/hp_cosine_lr3e4_s2", "win-small", 50000, 20),
    ("adam+cosine 4.5e-4", "rating_floors_hp_cosine_lr45e5.json", "probe/hp_cosine_lr45e5", "win-small", 50000, 20),
    ("adam+cosine 1e-3", "rating_floors_hp_cosine_lr1e3.json", "probe/hp_cosine_lr1e3", "fail", 50000, 20),
    ("adam+plateau 6e-4", "rating_floors_hp_plateau_lr6e4.json", "probe/hp_plateau_lr6e4", "worse", 50000, 15),
    ("ademamix 3e-4", "rating_floors_hp_ademamix_lr3e4.json", "probe/hp_ademamix_lr3e4", "worse", 50000, 14),
    ("lion 1e-4", "rating_floors_hp_lion_lr1e4.json", "probe/hp_lion_lr1e4", "worse", 50000, 22),
    ("schedule-free adamw 1e-3", "rating_floors_hp_sfree_lr1e3.json", "probe/hp_sfree_lr1e3", "worse", 50000, 20),
    ("gelu activation", "rating_floors_hp_act_gelu.json", "probe/hp_act_gelu", "null", 50000, 21),
    ("batch 1024, 4.5e-4", "rating_floors_hp_bs1024_lr45e5.json", "probe/hp_bs1024_lr45e5", "null", 25000, 21),
    ("muon 2e-2 (12k)", "rating_floors_hp_muon_lr2e2.json", "probe/hp_muon_lr2e2", "worse", 12000, 61),
    ("muon 5e-3 (12k)", "rating_floors_hp_muon_lr5e3.json", "probe/hp_muon_lr5e3", "win-small", 12000, 60),
    ("muon 5e-3 (30k)", "rating_floors_hp_muon_lr5e3_30k.json", "probe/hp_muon_lr5e3_30k", "win", 30000, 267),
    ("muon 5e-3 (30k) seed 2", "rating_floors_hp_muon_lr5e3_30k_s2.json", "probe/hp_muon_lr5e3_30k_s2", "win", 30000, 235),
]


def jsonl_records(prefix):
    p = A / f"{prefix}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


rows = []
for name, fj, pf, cls, steps, mins in RUNS:
    m = json.load(open(A / fj))["model"]
    recs = jsonl_records(pf)
    h = recs[-1]["holdout"]["corrupt"] if recs else {}
    rows.append({
        "name": name, "cls": cls, "steps": steps, "mins": mins,
        "mae": m["by_trust"][0]["mae"],
        "rho": m["ordering"]["rho_by_trust"][0]["rho_mean"],
        "nll": h.get("nll_drop_per_item"),
    })

curves = {}
for key, pf in [("adam control", "probe/probe_frac1.0"), ("adam cosine", "probe/hp_cosine_lr3e4"),
                ("muon 5e-3", "probe/hp_muon_lr5e3_30k")]:
    pts = [(r["step"], r["holdout"]["corrupt"]["mae_drop_per_item"]) for r in jsonl_records(pf)
           if "holdout" in r]
    curves[key] = pts

DATA = {"rows": rows, "curves": curves}
blob = json.dumps(DATA)
html = open(SP / "hp_sweep.html").read().replace("__DATA__", blob)
out = SP / "hp_sweep_built.html"
out.write_text(html)
print(f"built {out} ({len(html)//1024}KB)")
