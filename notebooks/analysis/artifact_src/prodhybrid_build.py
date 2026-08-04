"""Assemble DATA for the prod-vs-hybrid comparison artifact and build it."""

import json
from pathlib import Path

SP = Path(__file__).parent
A = Path("/home/casey/anime-atlas/data/aug2026")
PROBE = A / "probe"

bat = json.load(open(PROBE / "value_battery_cpu.json"))["models"]
for f in ("value_battery_sweep.json", "value_battery_graft.json",
          "value_battery_gate_s2.json", "value_battery_phase0.json"):
    bat.update(json.load(open(PROBE / f))["models"])
fr = json.load(open(A / "frontier_results.json"))
fg = json.load(open(A / "frontier_graft.json"))
fs = json.load(open(A / "frontier_stack.json"))
tmp = json.load(open(A / "temporal_hybrid_lwfine.json"))["scorers"]
ptwd = json.load(open(A / "ptw_eval_lwfine.json"))
eye = json.load(open(A / "eyeball_hybrid_fresh.json"))["ameo___-aug2026"]

LAT = {"prod": (1.3, 1.6), "hyb": (1.6, 2.0)}


def pct(p, h, dec=1):
    return f"{(h - p) / p * 100:+.{dec}f}%"


def battery_block(ctx, title, sub):
    p, h = bat["prod_fresh_logq"][ctx], bat["graft_concat"][ctx]
    pu, pf, hu, hf = p["unfiltered"], p["filtered"], h["unfiltered"], h["filtered"]
    rows = []

    def num(name, pv, hv, dir, fmtv=lambda v: f"{v:.3f}", delta=None):
        rows.append({"name": name, "prod": fmtv(pv), "hyb": fmtv(hv),
                     "delta": delta or pct(pv, hv), "dir": dir})

    mr_p, mr_h = pu["median_rank"], hu["median_rank"]
    num("median rank of dropped item", mr_p, mr_h,
        1 if mr_h < mr_p else (-1 if mr_h > mr_p else 0),
        fmtv=lambda v: str(round(v)), delta=f"{mr_h - mr_p:+.0f}")
    for name, key, src in [("recall@250, raw", "r250", "u"), ("recall@250, filtered", "r250", "f"),
                           ("mid band 1k–3k, filtered", "r250_tier1000_3000", "f"),
                           ("deep tail 3k–6k, filtered", "r250_tier3000_6000", "f"),
                           ("novelty-q4 recall, filtered", "r250_novq4", "f")]:
        pv = (pu if src == "u" else pf)[key]
        hv = (hu if src == "u" else hf)[key]
        d = (hv - pv) / pv
        num(name, pv, hv, 1 if d > 0.005 else (-1 if d < -0.005 else 0))
    num("franchise share of raw top-10", pu["franchise_share_top10"], hu["franchise_share_top10"], 0,
        fmtv=lambda v: f"{v * 100:.0f}%", delta=f"{(hu['franchise_share_top10'] - pu['franchise_share_top10']) * 100:+.0f}pp")
    num("top-10 mean pop rank, filtered", pf["mean_top10_poprank"], hf["mean_top10_poprank"], 0,
        fmtv=lambda v: str(round(v)), delta=f"{hf['mean_top10_poprank'] - pf['mean_top10_poprank']:+.0f}")
    return {"title": title, "sub": sub, "rows": rows}


master = [
    battery_block("keep0.99", "Full context", "1% of profile hidden · the headline regime"),
    battery_block("keep0.9", "Moderate corruption", "10% of profile hidden"),
    battery_block("k8", "Cold start", "8-item profiles"),
]

tp, th = tmp["nn"]["0.3"], tmp["concat"]["0.3"]
tmp_rows = []
for name, pv, hv, dir_by, fmtv, delta in [
    ("recall@50, overall", tp["overall"]["recall@50"], th["overall"]["recall@50"], "pct", None, None),
    ("recall@50, mid band filtered", tp["filtered_by_tier"]["1k-3k"]["recall@50"],
     th["filtered_by_tier"]["1k-3k"]["recall@50"], "pct", None, None),
    ("median rank of future watch", tp["overall"]["median_target_rank"], th["overall"]["median_target_rank"],
     "lower", lambda v: str(round(v)), f"{th['overall']['median_target_rank'] - tp['overall']['median_target_rank']:+.0f}"),
    ("top-10 mean pop rank", tp["rec_pop_top10"], th["rec_pop_top10"], "neutral",
     lambda v: str(round(v)), f"{th['rec_pop_top10'] - tp['rec_pop_top10']:+.0f}"),
]:
    if dir_by == "pct":
        d = (hv - pv) / pv
        dirv = 1 if d > 0.005 else (-1 if d < -0.005 else 0)
    elif dir_by == "lower":
        dirv = 1 if hv < pv else (-1 if hv > pv else 0)
    else:
        dirv = 0
    tmp_rows.append({"name": name, "prod": (fmtv or (lambda v: f"{v:.3f}"))(pv),
                     "hyb": (fmtv or (lambda v: f"{v:.3f}"))(hv),
                     "delta": delta or pct(pv, hv), "dir": dirv})
master.append({"title": "Future watches (temporal)", "sub":
               "prod combined score, lw = 0.3, shared rating head", "rows": tmp_rows})

pp, ph = ptwd["scorers"]["prod"], ptwd["scorers"]["hybrid"]
ptw_rows = []
for name, key, src, low_better in [
    ("median rank of PTW item", "median_rank", "unfiltered", True),
    ("recall@250, raw", "r250", "unfiltered", False),
    ("recall@250, filtered", "r250", "filtered", False),
    ("mid band 1k–3k, filtered", "r250_tier1000_3000", "filtered", False),
    ("deep tail 3k–6k, filtered", "r250_tier3000_6000", "filtered", False),
]:
    pv, hv = pp["0.3"][src][key], ph["0.3"][src][key]
    if low_better:
        ptw_rows.append({"name": name, "prod": str(round(pv)), "hyb": str(round(hv)),
                         "delta": f"{hv - pv:+.0f}", "dir": 1 if hv < pv else (-1 if hv > pv else 0)})
    else:
        d = (hv - pv) / pv
        ptw_rows.append({"name": name, "prod": f"{pv:.3f}", "hyb": f"{hv:.3f}",
                         "delta": pct(pv, hv), "dir": 1 if d > 0.005 else (-1 if d < -0.005 else 0)})
master.append({"title": "Declared intent (PTW)", "sub":
               "prod combined score, lw = 0.3, each model's own rating head", "rows": ptw_rows})

ALPHAS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3]


def curve(fam, src):
    return [{"alpha": a, "overall": src["filtered"][f"{fam}|k0|a{a}"]["overall_r250"],
             "mid": src["filtered"][f"{fam}|k0|a{a}"]["r250_tier1000_3000"],
             "tail": src["filtered"][f"{fam}|k0|a{a}"]["r250_tier3000_6000"],
             "pop10": src["filtered"][f"{fam}|k0|a{a}"]["mean_top10_poprank"]} for a in ALPHAS]


knob = {"nn": curve("nn", fr), "concat": curve("concat", fg), "stack": curve("stack_w0.35", fs)}

LWS = ["0.0", "0.1", "0.15", "0.2", "0.25", "0.3", "0.4", "0.5", "0.7", "1.0"]
lw = {m: [{"lw": float(w), "r50": tmp[m][w]["overall"]["recall@50"],
           "midf": tmp[m][w]["filtered_by_tier"]["1k-3k"]["recall@50"]} for w in LWS]
      for m in ("nn", "concat")}

ptw_curves = {m: [{"lw": float(w), "fr250": ptwd["scorers"][s][w]["filtered"]["r250"],
                   "fmid": ptwd["scorers"][s][w]["filtered"]["r250_tier1000_3000"]} for w in LWS]
              for m, s in (("nn", "prod"), ("concat", "hybrid"))}


def interp(pts, xk, yk, x):
    xs = [p[xk] for p in pts]
    ys = [p[yk] for p in pts]
    import bisect
    x = min(max(x, xs[0]), xs[-1])
    i = max(1, bisect.bisect_left(xs, x))
    f = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
    return ys[i - 1] + f * (ys[i] - ys[i - 1])


def eff(s):
    return (2 / 3) * s ** 3 + s / 3 + 0.01


slider = []
for s, mark in [(0.0, None), (0.2, None), (0.4, "default"), (0.5, None), (0.6, "proposed"),
                (0.7, None), (1.0, None)]:
    e = eff(s)
    label = {"default": f"{s:.1f} — current default", "proposed": f"{s:.1f} — proposed"}.get(mark, f"{s:.1f}")
    slider.append({
        "label": label, "mark": mark, "eff": f"{min(e, 1.0):.2f}" + ("+" if e > 1 else ""),
        "tr50": f"{interp(lw['concat'], 'lw', 'r50', e):.3f}",
        "tmid": f"{interp(lw['concat'], 'lw', 'midf', e):.3f}",
        "pr250": f"{interp(ptw_curves['concat'], 'lw', 'fr250', e):.3f}",
        "pmid": f"{interp(ptw_curves['concat'], 'lw', 'fmid', e):.3f}",
    })

pu03, hu03 = pp["0.3"]["unfiltered"], ph["0.3"]["unfiltered"]
pop03 = ptwd["scorers"]["pop"]["1.0"]["unfiltered"]
ptw = {
    "n_users": f"{ptwd['n_users']:,}",
    "mean_ptw": f"{ptwd['mean_ptw_per_user']:.0f}",
    "franch": f"{ptwd['target_franchise_share_mean'] * 100:.0f}%",
    "pop_note": (f"median rank {pop03['median_rank']:.0f} vs the models' ≈{hu03['median_rank']:.0f}, "
                 "and zero mid/tail recall"),
    "rating_tie": (f"{pp['0.0']['unfiltered']['r250']:.3f} vs "
                   f"{ph['0.0']['unfiltered']['r250']:.3f}"),
    "curves": ptw_curves,
    "callout": (
        "<b>Declared intent confirms the watch-history evals.</b> The hybrid ranks a user's "
        f"plan-to-watch list ahead of prod at every lw (at lw = 0.3: median rank "
        f"{hu03['median_rank']:.0f} vs {pu03['median_rank']:.0f}, filtered overall "
        f"{ph['0.3']['filtered']['r250']:.3f} vs {pp['0.3']['filtered']['r250']:.3f}, deep tail "
        f"+{(ph['0.3']['filtered']['r250_tier3000_6000'] / pp['0.3']['filtered']['r250_tier3000_6000'] - 1) * 100:.0f}%), "
        "and the shape matches temporal: the combined score beats presence-only overall, while the "
        "mid band keeps rising toward presence-heavy mixes. PTW is the cleanest signal we have that "
        "isn't watch-history reconstruction — it is what users say they want next."),
}

nn_titles = {r["title"] for r in eye["lists"]["nn_prodmix_lw0.3"]}
hy_titles = {r["title"] for r in eye["lists"]["concat_prodmix_lw0.3"]}
eyeballs = []
for key, label, other in [("nn_prodmix_lw0.3", "Prod", hy_titles), ("concat_prodmix_lw0.3", "Hybrid", nn_titles)]:
    eyeballs.append({"title": f"{label} — prod-mix score, lw = 0.3",
                     "sub": f"ameo___ · Aug 2026 profile · {eye['n_items']} in-corpus items",
                     "recs": [{**r, "shared": r["title"] in other} for r in eye["lists"][key]]})

kpis = [
    {"label": "Median rank of the next watch (full context)", "value": "59 → 46",
     "detail": "the held-out item climbs 13 places; independent second seed lands at 46 exactly"},
    {"label": "Deep-tail recall (3k–6k), filtered", "value": "+21.9%",
     "detail": "0.292 → 0.356 — the catalog region prod reaches worst is where the hybrid gains most"},
    {"label": "Hidden-gems mid band (1k–3k), filtered", "value": "+5.0%",
     "detail": "0.624 → 0.655 at full context; +6.5% at keep0.9 — the product-goal band"},
    {"label": "Cold start (8-item profiles)", "value": "≈ parity",
     "detail": "medrank 475 vs 454, recall within 2% — the one regime prod leads; the graft fades gracefully"},
]

pk_nn = max(knob["nn"], key=lambda p: p["overall"])
pk_cc = max(knob["concat"], key=lambda p: p["overall"])
pk_st = max(knob["stack"], key=lambda p: p["overall"])
knob_callout = (
    f"<b>The hybrid is prod's curve moved up, not a different knob.</b> Peak filtered overall goes "
    f"{pk_nn['overall']:.3f} (prod, α = {pk_nn['alpha']}) → {pk_cc['overall']:.3f} (hybrid, "
    f"α = {pk_cc['alpha']}), and the gap holds at every α — including the niche end, where prod "
    f"historically decays fastest. The optional stack reshapes the trade entirely: at full-niche "
    f"α = 0 it scores {knob['stack'][0]['overall']:.3f} overall — better than prod's site-default "
    f"setting — with a top-10 sitting at pop rank ≈ {knob['stack'][0]['pop10']:.0f}. Its α-response "
    f"is much flatter, which is exactly why the knob linearization needs re-anchoring before the "
    f"stack can default on (peak {pk_st['overall']:.3f} at α = {pk_st['alpha']})."
)

ops = [
    {"k": "Latency", "v": f"median {LAT['prod'][0]} → {LAT['hyb'][0]} ms per request "
     f"(p95 {LAT['prod'][1]} → {LAT['hyb'][1]}) on the dev box at bf16, 120-item profile — "
     "the EASE row-sum + 256-d projection costs ~0.3 ms"},
    {"k": "Serving assets", "v": "hybrid additionally loads the 6000×6000 EASE B matrix "
     "(144 MB f32; bf16 would halve it) plus a 24 KB μ vector; the checkpoint itself grows 222 → 229 MB "
     "(projection layer + widened decoder input)"},
    {"k": "Availability", "v": "both models live in the dev model server "
     "(<span class=\"mono\">models-local.json</span>, port 5709) and the app's model dropdown; "
     "a prod beta is one more entry in <span class=\"mono\">deploy/models.json</span> with the same name"},
    {"k": "Rating path", "v": "byte-compatible — same decoder shape reading the bottleneck only, "
     "so contribution analysis, holdout diagnostics, and the presence/rating slider all work unchanged"},
    {"k": "Dev-only stack flag", "v": "<span class=\"mono\">stack_weight</span> blends z-normed "
     "EASE-lift into the graft's logits before the (α,k) transform; it operates in z-units while the "
     "knob anchors were fit on raw lift, so the knob's feel shifts — re-anchor before this defaults on"},
]

methods = (
    "Value battery: probe_value_eval on 3,000 seed-999 holdout users, serve score = lift + log-pop "
    "(α = 1.0, k = 0 — the knob's neutral point); franchise filter = union-find over "
    "sequel/prequel/parent/side-story relations. Knob curves: frontier_eval protocol, 5,000 users, "
    "seed-123 pool / 777 corruption, identical across all three curves — directly overlayable. "
    "Temporal: frozen temporal_v3 fixtures (input = profile at 2025-06-24, targets = subsequent "
    "watches), prod combined score, prod rating head shared across scorers. Latency: 30 cache-busting "
    "requests against the dev server, this machine. PTW eval: 4,000 seed-999 holdout users with "
    "≥3 unrated in-corpus plan-to-watch entries (capped at 25/user, ~81k targets), model input "
    "reconstructed from the raw profile CSV and verified index-exact against the training vectors "
    "(0 mismatches); PTW items are never model input by design, so they are genuinely unseen. "
    "Combined score computed in log space (rank-equivalent to prod's softmax^lw · (rating+1)^(1−lw)); "
    "each model uses its own rating head there. Hybrid column = probe checkpoint seed 1 throughout; "
    "seed 2 agrees within 0.008 on the battery. Data: probe/value_battery_*.json, "
    "frontier_{results,graft,stack}.json, temporal_hybrid_lwfine.json, ptw_eval_lwfine.json, "
    "eyeball_hybrid_prodmix.json; script analysis/ptw_eval.py."
)

dom = {
    "recall": [0.5, 0.82], "recallTicks": [0.5, 0.6, 0.7, 0.8],
    "mid": [0.35, 0.78], "midTicks": [0.4, 0.5, 0.6, 0.7],
    "tail": [0, 0.72], "tailTicks": [0, 0.2, 0.4, 0.6],
    "pop": [0, 3300], "popTicks": [0, 1000, 2000, 3000],
    "lwo": [0, 0.26], "lwoTicks": [0, 0.1, 0.2],
    "lwm": [0, 0.07], "lwmTicks": [0, 0.02, 0.04, 0.06],
    "ptwo": [0.1, 0.55], "ptwoTicks": [0.1, 0.2, 0.3, 0.4, 0.5],
    "ptwm": [0.05, 0.3], "ptwmTicks": [0.1, 0.2, 0.3],
}

data = {"kpis": kpis, "master": master, "knob": knob, "knob_callout": knob_callout,
        "lw": lw, "ptw": ptw, "slider": slider, "eyeballs": eyeballs, "ops": ops,
        "methods": methods, "dom": dom}

blob = json.dumps(data, separators=(",", ":"))
html = open(SP / "prod_vs_hybrid.html").read().replace("__DATA__", blob)
open(SP / "prod_vs_hybrid_built.html", "w").write(html)
wrapped = ('<!doctype html><html data-theme="light"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width, initial-scale=1"></head><body>'
           + html + "</body></html>")
open(SP / "prodhybrid_check.html", "w").write(wrapped)
print("built", len(blob) // 1024, "KB data,", len(html) // 1024, "KB html")
