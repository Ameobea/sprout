"""Full artifact DATA assembly + build. Replays the round 1-3 assembly (recovered
from the prior session transcript) from source files, then adds round 4.
Idempotent: always builds from artifact_data.json.bak + result JSONs."""

import json
from pathlib import Path

SP = Path(__file__).parent
REPO = Path("/home/casey/anime-atlas")
PROBE = REPO / "data/aug2026/probe"

# ---- round 1 base (slimmed exactly as the published build) ----
full = json.load(open(SP / "artifact_data.json.bak"))
d = {}
fl = dict(full["floors"]); fl.pop("profile_size_hist", None)
d["floors"] = fl
d["decomp_logq"] = dict(full["decomp_logq"])
d["decomp_logq_ranks"] = {"keep_sweep": [
    {k: s[k] for k in ("keep_frac_target", "median_rank", "recall@50", "recall@250",
                       "nll_drop_per_item", "mae_drop_per_item")}
    for s in full["decomp_logq_ranks"]["keep_sweep"]]}
tw = full["twins"]
d["twins"] = {k: tw[k] for k in ("j_edges", "ctx_edges", "bins", "n_candidate_pairs", "n_exact_groups")}
ez = full["ease"]
d["ease"] = {k: ez[k] for k in ("keep_sweep", "popularity_rank_stats", "best_lam")}
d["ladder"] = full["ladder"]
d["additive_matched"] = full["additive_matched"]

# ---- scaling ----
def curves(frac):
    return [json.loads(l) for l in open(PROBE / f"probe_frac{frac}.jsonl")]

points = []
for frac, users, label in [("0.03", 42495, "42k"), ("0.1", 141652, "142k"),
                           ("0.3", 424958, "425k"), ("1.0", 1416529, "1.42M")]:
    r = curves(frac)[-1]
    h, t = r["holdout"]["corrupt"], r["train"]["corrupt"]
    points.append(dict(users=users, label=label,
        holdout_nll=h["nll_drop_per_item"], train_nll=t["nll_drop_per_item"],
        holdout_mae=h["mae_drop_per_item"], train_mae=t["mae_drop_per_item"]))

gap_series = []
for frac, name, dy in [("1.0", "90% data", -8), ("0.1", "10% data", -8), ("0.03", "3% data", -8)]:
    pts = []
    for r in curves(frac):
        g = r["holdout"]["corrupt"]["nll_drop_per_item"] - r["train"]["corrupt"]["nll_drop_per_item"]
        pts.append([r["step"] / 1000, round(g, 4)])
    gap_series.append(dict(name=name, pts=pts, dy=dy))

scaling_extra = """<p class="callout" style="margin-top:14px"><b>More data is an exhausted lever.</b> Each ~3.3&times; increase in
users buys half the previous gain (0.107 &rarr; 0.049 &rarr; 0.022 nats); the geometric tail puts the
infinite-data ceiling at &asymp;7.04 nats vs 7.061 today, i.e. &asymp;0.02 nats and &asymp;0.004 z-MAE
of headroom from data volume at this architecture. Memorization is nil at full scale (gap
+0.024 nats; the production checkpoint scores identically on users it trained on vs. a
model that never saw them) &mdash; but at 3% data the same architecture overfits freely, so capacity
is not the binding constraint. <span id="mixdrop-slot"><b>Dropout-mixture probe: negative.</b> Trained across keep 0.52&ndash;0.975 \
(vs. the recipe’s 0.44&ndash;0.76), full-context ranking is unchanged — median rank 60 / r@50 0.468 \
vs. the control’s 60 / 0.466 — and every other metric is slightly worse (its lower mean rate \
echoes the closed “0.4 optimal” sweep). Training at the serve regime does not unlock the \
full-context gap; what EASE uses is item-item detail the bottleneck discards.</span></p>"""

d["scaling"] = dict(points=points, nll_ydom=[6.6, 7.35], nll_yticks=[6.7, 6.9, 7.1, 7.3],
    gap_ydom=[0, 0.6], gap_yticks=[0, 0.2, 0.4, 0.6], gap_series=gap_series, extra_html=scaling_extra)

# ---- round 2 ----
r2 = json.load(open(SP / "round2_data.json"))
r2["cumshare"] = json.load(open(SP / "cumshare.json"))
graft = json.load(open(PROBE / "probe_ease3ch_eval.json"))

def row(cells, tag="td"):
    return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"

g99, g90, g60 = graft["keep0.99"], graft["keep0.9"], graft["keep0.6"]
r2["graft_html"] = f"""
<p class="sec-note"><b>Negative — and mechanistically informative.</b> A 3-channel probe fed the
per-user z-normed EASE score vector as an input channel (computed in-graph from the corrupted
presence, full production recipe, same 10% user holdout). Full-context ranking barely moves:
median rank <b>{g99['median_rank']:.0f}</b> vs the 2-channel control's 60 (r@50 {g99['recall@50']:.3f} vs 0.466)
— nowhere near EASE's 33. The EASE signal enters on the input side and gets compressed through the
same 512-d bottleneck that discards the raw item-item detail in the first place. <b>Input-side
grafting cannot fix an output-side constraint.</b> Fusion that works must happen after the decoder:
score-level blending (§9, already beats both models), or — as an architecture experiment — a
learned item-item skip connection straight from input to logits, which is essentially "EASE as a
layer" trained jointly. Given the blend banks most of the win with zero training, the skip
experiment is optional polish, judged on rank-at-full-context + tail tiers.</p>
<div class="card"><h3>Graft (3ch) vs control (2ch), holdout users</h3>
<details class="tbl" open><summary>Results</summary><table>
{row(['config','graft medrank','control medrank','graft r@50','control r@50','EASE ref medrank'],'th')}
{row(['keep 0.99', f"{g99['median_rank']:.0f}", '60', f"{g99['recall@50']:.3f}", '0.466', '33'])}
{row(['keep 0.9', f"{g90['median_rank']:.0f}", '77', f"{g90['recall@50']:.3f}", '0.412', '60'])}
{row(['keep 0.6', f"{g60['median_rank']:.0f}", '~140', f"{g60['recall@50']:.3f}", '~0.28', '151'])}
{row(['k=8 items', f"{graft['k8']['median_rank']:.0f}", '459*', f"{graft['k8']['recall@50']:.3f}", '0.122*', '495*'])}
</table></details>
<p class="subtitle">* control/EASE k=8 refs from the §8 sweep (general-user pool; memorization is nil so comparable).</p></div>"""

fx = json.load(open(REPO / "data/aug2026/corpus12k/results_eval_fixed.json"))
res = json.load(open(REPO / "data/aug2026/corpus12k/results.json"))
b0_99, b2_99 = fx["beta0.0_keep0.99"], fx["beta0.217_keep0.99"]
r2["corpus12k_html"] = f"""
<p class="sec-note">From the raw 551M-row dump ({res['n_distinct_items']:,} distinct items):
the current 6k corpus already covers <b>{res['share_top_6000']*100:.1f}%</b> of presence entries;
ranks 6k–12k add just <b>{res['share_6k_12k_band']*100:.2f}%</b> (item counts 768–4,953 in that band).
An EASE built at 12k (same recipe, 1.58M users) answers the two decision questions directly:</p>
<ul class="tight">
<li><b>Doubling the corpus does not improve existing predictions.</b> On targets inside the old 6k
corpus (6k-restricted candidate pool), 12k-EASE scores median rank {b2_99['6kpool']['medrank']:.0f} vs the
6k-EASE's 33 — identical. The band's extra context carries no marginal signal for head items,
consistent with the data-scaling saturation.</li>
<li><b>But the band itself is genuinely recommendable.</b> With pure affinity scores (β=0), held-out
band targets reach median rank {b0_99['band']['medrank']:.0f} / recall@250 <b>{b0_99['band']['r@250']:.3f}</b>
at full context — against a popularity-rank median of 8,926. Real deep-tail discovery signal exists
beyond the current corpus; it just needs logQ-style thin-evidence serving (any popularity-mixed
score buries it: at the standard mix the band's recall@250 is {b2_99['band']['r@250']:.3f}).</li>
<li><b>Costs are asymmetric.</b> NN at 12k roughly doubles its parameter count; EASE at 12k is a
4× memory bump (576MB f32 / 288MB bf16) with trivial compute. A 12k EASE component could serve the
band without touching the NN at all.</li>
<li>Side-finding: β=0 EASE also ranks <i>better overall</i> than the NLL-calibrated mix (median rank
{b0_99['all']['medrank']:.0f} vs {b2_99['all']['medrank']:.0f} at full context) — EASE serving wants far less
popularity than its likelihood calibration suggests, with the §7 franchise-magnetism caveat.</li>
</ul>
<div class="card"><h3>Where the presence mass lives</h3>
<p class="subtitle">cumulative share of all {res['total_presence_entries']//1_000_000}M presence entries by popularity rank</p>
<div class="chart-wrap"><div id="ch-cumshare"></div></div>
<details class="tbl"><summary>Table view</summary><div id="tbl-cumshare"></div></details></div>"""

r2["implications2_html"] = """
<ul class="tight">
<li><b>Serving direction with the strongest evidence: profile-size-adaptive score blend.</b>
w → 0 below ~100 items (protects cold start and the interactive recommender, where the NN wins
everything and EASE degenerates to popularity); w ≈ 0.35–0.5 for large profiles (median rank
52 → 29 at full context). Tier-aware blending (less EASE weight in the deep tail) would keep the
NN's tail reach; the whole thing is a serving-layer change — no retraining, and EASE inference is
cheaper than the NN forward (~1.4M vs ~50M MACs, B = 72MB bf16, exact LOO by row subtraction).</li>
<li><b>The interactive recommender needs no backup.</b> NN beats EASE at every absolute context
size k=4–128 and in the smallest profile bin; blending there actively hurts tail discovery.</li>
<li><b>Input-side grafting is refuted; the bottleneck is the chokepoint in both directions.</b>
If architecture work is ever pursued, the shaped experiment is an output-side item-item skip
connection ("EASE as a layer", trained jointly), judged on rank-at-full-context + tail tiers —
but the free blend already banks most of the measurable win.</li>
<li><b>Corpus 12k reframed: a niche-end product feature, not an accuracy lever.</b> Zero gain for
existing predictions; real, reachable discovery signal in the 6k–12k band (recall@250 0.516 with
raw affinity). Cheapest capture: a 12k EASE component for the niche end of the knob; an NN-side 12k
retrain doubles params for no head benefit.</li>
<li><b>EASE's serving knob must be calibrated on lists, not likelihood</b> — the NLL-fit popularity
mix buries the tail and still under-ranks overall; raw affinity wins metrics but is
franchise-magnetic, so the extra-season filter port (backlog) becomes a prerequisite for any EASE
serving path.</li>
<li><b>Single-run caveats:</b> graft, mixdrop, and the 12k-EASE numbers are one run each; the blend
and stratified results come from one 5k-user protocol. Confirm the load-bearing ones (blend win,
size-bin crossover) with a second seed before any prod decision.</li>
</ul>"""
d["round2"] = r2

# ---- round 3 ----
fr = json.load(open(REPO / "data/aug2026/frontier_results.json"))
le = json.load(open(SP / "ease_lift_eyeballs.json"))
r3 = {
    "alphas": fr["alphas"],
    "alpha_keys": [str(a) for a in fr["alphas"]],
    "filtered": {k: v for k, v in fr["filtered"].items() if "|k0|" in k and not k.startswith("blend0.5")},
    "rho_pop": fr["rho_pop"],
    "lift_eyeball_ameo": le["ameo___"]["alpha0.6"],
}
r3["implications3_html"] = """
<h3 style="margin-top:18px">Where this leaves the hybrid</h3>
<ul class="tight">
<li><b>Casey's instinct was right, quantitatively:</b> most of naive EASE's aggregate advantage was
franchise fragments plus popularity mixing — the exact "easy to recommend Death Note" failure mode.
Post-filter, its standalone edge is modest (+0.05 r@50).</li>
<li><b>What remains is worth having:</b> a blend of NN-lift and EASE-lift, served through the same
(α, k) knob, strictly dominates the NN-only frontier post-filter — +0.15 mid-band / +0.22 deep-tail
recall at matched overall ≈ 0.74, with nicher lists. The gain concentrates in the 50–3k popularity
band the (α,k) work identified as where recommendation value lives.</li>
<li><b>Serving shape that drops out of rounds 2+3:</b> score = (1−w)·z(lift_NN) + w·z(lift_EASE) +
α·log&nbsp;pop, with w ramping 0 → ~0.35 as profile size grows past ~100 items, α on the existing
niche slider, extra-season filter mandatory. All components exist in prod already except the B
matrix (72MB bf16, ~1.4M MACs/req) and the μ vector.</li>
<li><b>EASE-lift needs its own knob anchor points</b> — its lift is anti-popularity-correlated
(ρ −0.48 vs NN's +0.30), so equal α means nicher-feeling lists than the NN path; the unified-slider
remap must be re-fit for the blended score (same alpha_k_sweep methodology, one afternoon).</li>
<li><b>Open confirmations before a prod decision:</b> second seed for the frontier (single 5k-user
run), an eyeball pass on blended lists at the anchor settings, and the same frontier on the
temporal fixtures (future-watch targets) as the guardrail — the current protocol predicts held-out
catalog items, not future watches.</li>
</ul>"""
d["round3"] = r3

# ---- round 4 ----
bat = json.load(open(PROBE / "value_battery_cpu.json"))["models"]
bat.update(json.load(open(PROBE / "value_battery_sweep.json"))["models"])
for extra_bat in ("value_battery_graft.json", "value_battery_gate_s2.json"):
    p = PROBE / extra_bat
    if p.exists():
        bat.update(json.load(open(p))["models"])


def jsonl_final(name):
    rec = None
    for line in open(PROBE / f"{name}.jsonl"):
        rec = json.loads(line)
    h = rec["holdout"]["corrupt"]
    return h["presence_loss"], h["nll_drop_per_item"], h["nll_kept_per_item"]


widths = [128, 256, 512, 1024, 2048]
names = ["probe_bneck128", "probe_bneck256", "probe_frac1.0", "probe_bneck1024", "probe_bneck2048"]
total, drop, kept = zip(*[jsonl_final(n) for n in names])
s2_total, s2_drop, _ = jsonl_final("probe_b512_seed2")
bnames = ["bneck128", "bneck256", "b512_ctl", "bneck1024", "bneck2048"]

round4 = {
    "kpis": [
        {"label": "Raw recall's blind spot (EASE @ k8 cold start)", "value": "0.35 / 0.00",
         "detail": "overall r@250 vs franchise-filtered deep-tail r@250 — a top-10 with mean popularity rank 13 scores “fine” on raw recall"},
        {"label": "Bottleneck width above 512, full-context ranking", "value": "flat",
         "detail": "medrank 58 → 58 → 62 at 512/1024/2048; EASE's medrank 36 is out of reach at every width — the loss is in the sum-pooling encoder, not the latent"},
        {"label": "Where big-width loss gains go", "value": "reconstruction",
         "detail": "dropped-item NLL is best at 256 and worsens above 512 — loss curves flattered input memorization, which is how the old intuition formed"},
        {"label": "Post-bottleneck graft", "value": "training…",
         "detail": "gate + concat runs in flight"},
    ],
    "metric_trap": {
        "groups": ["overall r@250 (raw)", "deep-tail r@250 (filt)", "novelty-q4 r@250 (filt)"],
        "nn": [bat["b512_ctl"]["k8"]["unfiltered"]["r250"],
               bat["b512_ctl"]["k8"]["filtered"]["r250_tier3000_6000"],
               bat["b512_ctl"]["k8"]["filtered"]["r250_novq4"]],
        "ease": [bat["ease"]["k8"]["unfiltered"]["r250"],
                 bat["ease"]["k8"]["filtered"]["r250_tier3000_6000"],
                 bat["ease"]["k8"]["filtered"]["r250_novq4"]],
        "nn_pop10": round(bat["b512_ctl"]["k8"]["filtered"]["mean_top10_poprank"]),
        "ease_pop10": round(bat["ease"]["k8"]["filtered"]["mean_top10_poprank"]),
    },
    "value_full": {"groups": ["overall r@250", "deep-tail r@250", "novelty-q4 r@250"], "series": []},
    "bneck": {
        "widths": widths, "total": list(total), "drop_nll": list(drop), "kept_nll": list(kept),
        "total_delta": [t - total[2] for t in total],
        "drop_delta": [x - drop[2] for x in drop],
        "seed2_total": s2_total, "seed2_drop": s2_drop,
    },
    "bneck_rank": {
        "medrank": [round(bat[m]["keep0.99"]["unfiltered"]["median_rank"]) for m in bnames],
        "r250_unf": [bat[m]["keep0.99"]["unfiltered"]["r250"] for m in bnames],
        "filt_tail": [bat[m]["keep0.99"]["filtered"]["r250_tier3000_6000"] for m in bnames],
        "novq4": [bat[m]["keep0.99"]["filtered"]["r250_novq4"] for m in bnames],
        "ease_r250": bat["ease"]["keep0.99"]["unfiltered"]["r250"],
        "ease_medrank": round(bat["ease"]["keep0.99"]["unfiltered"]["median_rank"]),
        "medrank_512": round(bat["b512_ctl"]["keep0.99"]["unfiltered"]["median_rank"]),
    },
}

for name, key in [("NN (b512, α_add=1.0)", "b512_ctl"), ("EASE (s + 0.217·log pop)", "ease"),
                  ("blend w=0.35 α=0.45", "blend35_a45"), ("blend + w→0 under 64 kept", "blend35_a45_minctx64")]:
    f = bat[key]["keep0.99"]["filtered"]
    round4["value_full"]["series"].append({
        "name": name,
        "vals": [f["r250"], f["r250_tier3000_6000"], f["r250_novq4"]],
        "pop10": round(f["mean_top10_poprank"]),
    })

graft_specs = [("control (b512)", "b512_ctl"), ("input graft 3ch (r2)", "graft3ch"),
               ("gate", "graft_gate"), ("concat", "graft_concat")]
have = [(n, k) for n, k in graft_specs if k in bat]
if len(have) > 1:
    round4["graft"] = {
        "groups": ["overall r@250 (unf)", "deep-tail r@250 (filt)", "novelty-q4 r@250 (filt)"],
        "series": [{
            "name": n,
            "medrank": round(bat[k]["keep0.99"]["unfiltered"]["median_rank"]),
            "vals": [bat[k]["keep0.99"]["unfiltered"]["r250"],
                     bat[k]["keep0.99"]["filtered"]["r250_tier3000_6000"],
                     bat[k]["keep0.99"]["filtered"]["r250_novq4"]],
        } for n, k in have],
    }

gate_eval_p = PROBE / "probe_graft_gate_eval.json"
if gate_eval_p.exists():
    ge = json.load(open(gate_eval_p))
    cfgs = ["k8", "k16", "keep0.6", "keep0.9", "keep0.99"]
    ymax = max(ge[c]["gate_p90"] for c in cfgs) * 1.25
    round4["gate"] = {
        "configs": cfgs,
        "mean": [ge[c]["gate_mean"] for c in cfgs],
        "p10": [ge[c]["gate_p10"] for c in cfgs],
        "p90": [ge[c]["gate_p90"] for c in cfgs],
        "ymax": ymax,
        "yticks": [round(ymax * q, 2) for q in (0, 0.25, 0.5, 0.75, 1.0)],
    }

UNIFIED_ROWS = [
    ("NN control (b512)", "b512_ctl", "α_add=1.0 raw (likelihood-neutral)"),
    ("EASE", "ease", "β=0.217 raw (NLL-calibrated)"),
    ("3ch input graft (r2)", "graft3ch", "α_add=1.0 raw"),
    ("graft gate", "graft_gate", "α_add=1.0 raw"),
    ("graft gate seed-2", "graft_gate_s2", "α_add=1.0 raw"),
    ("graft concat", "graft_concat", "α_add=1.0 raw"),
    ("NN @ z-α=0.45", "nn_a45", "α_z=0.45 (≈0.23 raw) — nicher"),
    ("blend w=0.20", "blend20_a45", "α_z=0.45 (≈0.23 raw) — nicher"),
    ("blend w=0.35", "blend35_a45", "α_z=0.45 (≈0.23 raw) — nicher"),
    ("blend policy (w→0 <64)", "blend35_a45_minctx64", "α_z=0.45 (≈0.23 raw) — nicher"),
]
if "graft_gate_s2" not in bat:
    UNIFIED_ROWS = [r for r in UNIFIED_ROWS if r[1] != "graft_gate_s2"]
round4["unified"] = {
    "configs": ["keep0.99", "keep0.9", "k16", "k8"],
    "cols": ["medrank", "unf r@250", "franch top-10", "filt r@250", "filt 250–1k",
             "filt 1k–3k", "filt 3k–6k", "novelty-q4", "filt top-10 pop"],
    "rows": [{
        "name": name, "knob": knob,
        "per_cfg": {cfg: [
            round(bat[key][cfg]["unfiltered"]["median_rank"]),
            round(bat[key][cfg]["unfiltered"]["r250"], 3),
            round(bat[key][cfg]["unfiltered"]["franchise_share_top10"], 2),
            round(bat[key][cfg]["filtered"]["r250"], 3),
            round(bat[key][cfg]["filtered"]["r250_tier250_1000"], 3),
            round(bat[key][cfg]["filtered"]["r250_tier1000_3000"], 3),
            round(bat[key][cfg]["filtered"]["r250_tier3000_6000"], 3),
            round(bat[key][cfg]["filtered"]["r250_novq4"], 3),
            round(bat[key][cfg]["filtered"]["mean_top10_poprank"]),
        ] for cfg in ["keep0.99", "keep0.9", "k16", "k8"]},
    } for name, key, knob in UNIFIED_ROWS],
}

extra_p = SP / "round4_extra.json"
if extra_p.exists():
    round4.update(json.load(open(extra_p)))
d["round4"] = round4

blob = json.dumps(d, separators=(",", ":"))
html = open(SP / "info_audit.html").read().replace("__DATA__", blob)
open(SP / "info_audit_built.html", "w").write(html)
wrapped = ('<!doctype html><html data-theme="light"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width, initial-scale=1"></head><body>'
           + html + "</body></html>")
open(SP / "render_check_light.html", "w").write(wrapped)
print("built", len(blob) // 1024, "KB data,", len(html) // 1024, "KB html")
