"""Assemble DATA for the phase-0 production-decision artifact and build it."""

import json
from pathlib import Path

SP = Path(__file__).parent
A = Path("/home/casey/anime-atlas/data/aug2026")
PROBE = A / "probe"

fr = json.load(open(A / "frontier_results.json"))
fg = json.load(open(A / "frontier_graft.json"))
fs = json.load(open(A / "frontier_stack.json"))
tmp = json.load(open(A / "temporal_hybrid.json"))
knn = json.load(open(A / "rating_knn_probe.json"))
eye = json.load(open(A / "eyeball_hybrid.json"))

bat = json.load(open(PROBE / "value_battery_cpu.json"))["models"]
for f in ("value_battery_sweep.json", "value_battery_graft.json",
          "value_battery_gate_s2.json", "value_battery_phase0.json"):
    bat.update(json.load(open(PROBE / f))["models"])

ALPHAS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3]
SRC = {"nn": fr, "blend0.35": fr, "ease": fr, "gate": fg, "concat": fg,
       "both": fs, "stack_w0.2": fs, "stack_w0.35": fs}


def curve(fam):
    src = SRC[fam]
    out = []
    for a in ALPHAS:
        s = src["filtered"][f"{fam}|k0|a{a}"]
        u = src["unfiltered"][f"{fam}|k0|a{a}"]
        out.append({"alpha": a, "overall": s["overall_r250"],
                    "mid": s["r250_tier1000_3000"], "tail": s["r250_tier3000_6000"],
                    "pop10": round(s["mean_top10_poprank"]),
                    "franch": u["franchise_share_top10"]})
    return out


frontier = {fam: curve(fam) for fam in ["nn", "blend0.35", "concat", "stack_w0.35"]}

matched = []
for fam, label in [("nn", "NN (fresh-logq)"), ("ease", "EASE"), ("blend0.35", "blend-on-NN w=0.35"),
                   ("gate", "graft gate"), ("concat", "graft concat"), ("both", "graft both"),
                   ("stack_w0.2", "graft + stack w=0.2"), ("stack_w0.35", "graft + stack w=0.35")]:
    c = curve(fam)
    best = min(c, key=lambda p: abs(p["overall"] - 0.742))
    matched.append({"name": label, "alpha": best["alpha"], "overall": best["overall"],
                    "mid": best["mid"], "tail": best["tail"], "pop10": best["pop10"],
                    "franch": best["franch"]})


def k99(key):
    u = bat[key]["keep0.99"]["unfiltered"]
    f = bat[key]["keep0.99"]["filtered"]
    return {"medrank": round(u["median_rank"]), "unf": u["r250"], "filt": f["r250"],
            "tail": f["r250_tier3000_6000"], "nov": f["r250_novq4"],
            "fr": u["franchise_share_top10"]}


repl = [{"name": n, **k99(k)} for n, k in [
    ("NN control seed 1", "b512_ctl"), ("NN control seed 2", "b512_seed2"),
    ("graft gate seed 1", "graft_gate"), ("graft gate seed 2", "graft_gate_s2"),
    ("graft concat seed 1", "graft_concat"), ("graft concat seed 2", "graft_concat_s2"),
    ("graft both (single run)", "graft_both")]]

t3 = tmp["scorers"]
temporal = {
    "franchise_share": f"{tmp['target_franchise_share_mean']*100:.0f}%",
    "series": [], "table": []}
for key, label in [("nn", "NN"), ("concat", "concat graft"), ("blend", "blend policy"), ("ease", "EASE")]:
    r = t3[key]["0.3"]
    mid = r["by_popularity_tier"].get("1k-3k", {}).get("recall@50", 0)
    midf = r["filtered_by_tier"].get("1k-3k", {}).get("recall@50", 0)
    temporal["series"].append({"name": label, "vals": [r["overall"]["recall@50"], mid, midf]})
    temporal["table"].append({"name": label, "r50": r["overall"]["recall@50"], "mid": mid,
                              "midf": midf, "med": round(r["overall"]["median_target_rank"]),
                              "pop10": round(r["rec_pop_top10"])})

rating = [
    {"name": "predict zero", "v": knn["zero"]},
    {"name": "per-item means", "v": knn["item_mean"]},
    {"name": "additive item+user (r1)", "v": 0.502},
    {"name": "presence-cosine kNN", "v": knn["cos_top20_shrunk"], "knn": True},
    {"name": "rating-correlation kNN (best)", "v": knn["rsim_top20"], "knn": True},
    {"name": "NN rating head", "v": 0.447, "em": True},
]

eyeballs = []
for prof, scorer in [("ameo___", "concat_za0.45"), ("snapsauce", "concat_za0.45")]:
    e = eye[prof]
    eyeballs.append({"title": f"{prof} — concat graft, z-α = 0.45",
                     "sub": f"{e['n_items']}-item profile · struck = prod filter would remove",
                     "recs": e["lists"][scorer]})

kpis = [
    {"label": "Stack vs blend-on-NN at matched overall ≈ 0.74", "value": "0.733 / 0.592",
     "detail": "mid-band / deep-tail r@250 vs 0.714 / 0.557 — dominates the round-3 winner on both axes"},
    {"label": "Peak filtered overall recall", "value": "0.797",
     "detail": "graft+stack w=0.2 at α=0.8 — new ceiling (blend-on-NN 0.783, NN 0.747)"},
    {"label": "Temporal guardrail (future watches)", "value": "clean",
     "detail": "grafts within noise of NN overall; hybrid mid-band +72%; EASE-alone fails the product task"},
    {"label": "Knob transfer", "value": "ρ +0.30",
     "detail": "graft logits' popularity correlation ≈ pure NN — the (α,k) remap machinery re-anchors, not redesigns"},
]

frontier_callout = (
    "<b>The stack ends the blend-vs-graft debate by refusing the choice.</b> The graft absorbs "
    "EASE's signal cleanly (no franchise inflation, learned scheduling, one model) but leaves some "
    "mid-band on the table; post-hoc access to the full-resolution pairwise scores recovers it. "
    "Stacking w≈0.2–0.35 of z-normed EASE-lift onto the <i>graft's</i> logits beats blend-on-NN "
    "everywhere because the base model is simply better — at matched overall ≈ 0.742: mid-band "
    "0.733 vs 0.714, deep-tail 0.592 vs 0.557, with a nichest-in-class ceiling. The literature "
    "agrees with the mechanism (VASP: keep the full-resolution linear path; Steck: 60% of EASE's "
    "weights are negative and any compression loses that dissimilarity signal — which is exactly "
    "why the 256-d concat projection alone can't carry it all)."
)

data = {"kpis": kpis, "frontier": frontier, "matched": matched,
        "frontier_callout": frontier_callout, "repl": repl, "temporal": temporal,
        "rating": rating, "eyeballs": eyeballs}

blob = json.dumps(data, separators=(",", ":"))
html = open(SP / "phase0_decision.html").read().replace("__DATA__", blob)
open(SP / "phase0_decision_built.html", "w").write(html)
wrapped = ('<!doctype html><html data-theme="light"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width, initial-scale=1"></head><body>'
           + html + "</body></html>")
open(SP / "phase0_check.html", "w").write(wrapped)
print("built", len(blob) // 1024, "KB data,", len(html) // 1024, "KB html")
