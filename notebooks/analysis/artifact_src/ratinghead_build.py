"""Assemble DATA for the rating-head artifact and build it."""

import json
from pathlib import Path

import numpy as np

SP = Path(__file__).parent
A = Path("/home/casey/anime-atlas/data/aug2026")

census = json.load(open(A / "rating_scheme_census.json"))
fc = json.load(open(A / "rating_floors_control.json"))
fp = json.load(open(A / "rating_floors_prod.json"))
fr = json.load(open(A / "rating_floors_rprior.json"))
tw_all = json.load(open(A / "twin_noise_all.json"))
tw_tr = json.load(open(A / "twin_noise_trusted.json"))

pop_hist = np.load(A / "rating_census.npz")["hist"].astype(np.int64)[:, 1:].sum(0)

CLUSTER_NAMES = ["wide 5–10", "7–8 core", "8–9 core", "tens-leaning", "7 core",
                 "harsh mid-band", "8 spike", "9–10 heavy", "9 spike", "near-all-10s"]

R2PI = float(np.sqrt(2 / np.pi))


def sig(b):
    return np.sqrt(b["var_dz"] / 2) if b["count"] else None


def twin_rows():
    je, ce = tw_all["j_edges"], tw_all["ctx_edges"]
    rows = []
    for j, cidx in [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]:
        a, t = tw_all["bins"][j][cidx], tw_tr["bins"][j][cidx]
        if t["count"] < 2000:
            continue
        sa, st = sig(a), sig(t)
        rows.append({
            "band": f"J [{je[j]:.2f},{je[j+1]:.2f}) × ctx [{ce[cidx]:.1f},{ce[cidx+1]:.1f})",
            "sa": sa, "st": st, "fa": sa * R2PI, "ft": st * R2PI, "n": t["count"],
        })
    return rows


def strata():
    cols = ["trusted", "one-sitting", "degenerate", "tier 0–250", "250–1k", "1k–3k", "3k–6k"]
    rows = []
    for name, src, key, em in [
        ("global mean", fc, "global_mean", False),
        ("item mean", fc, "item_mean", False),
        ("additive (feasible)", fc, "additive_feasible", False),
        ("model — control probe", fc, "model", True),
        ("model — prod ckpt", fp, "model", False),
        ("model — item-prior probe", fr, "model", False),
    ]:
        r = src[key]
        rows.append({"name": name, "overall": r["overall"]["mae"], "em": em,
                     "vals": [r["by_trust"][i]["mae"] for i in range(3)]
                     + [r["by_tier"][i]["mae"] for i in range(4)]})
    return {"cols": cols, "rows": rows}


def load_if(name):
    p = A / name
    return json.load(open(p)) if p.exists() else None


def jsonl_last(name):
    p = A / name
    if not p.exists():
        return None
    return json.loads(open(p).read().strip().split("\n")[-1])


def tm(f):
    m = f["model"]
    return m["by_trust"][0]["mae"], m["ordering"]["rho_by_trust"][0]["rho_mean"]


def round2_split():
    sr = load_if("rating_floors_split_rating.json")
    pr = jsonl_last("probe/probe_split_presence.jsonl")
    ctl = jsonl_last("probe/probe_frac1.0.jsonl")["holdout"]["corrupt"]
    s2 = jsonl_last("probe/probe_b512_seed2.jsonl")["holdout"]["corrupt"]
    rows = [
        {"name": "multitask control", "vals": [0.4507, 0.6044, ctl["nll_drop_per_item"], ctl["nll_kept_per_item"]], "em": True},
        {"name": "multitask seed-2", "vals": [0.4516, 0.6020, s2["nll_drop_per_item"], s2["nll_kept_per_item"]]},
    ]
    if sr:
        m, r = tm(sr)
        rows.append({"name": "rating-only", "vals": [m, r, None, None]})
    if pr:
        h = pr["holdout"]["corrupt"]
        rows.append({"name": "presence-only", "vals": [None, None, h["nll_drop_per_item"], h["nll_kept_per_item"]]})
    return {"cols": ["trusted MAE", "trusted ρ", "presence nll (held-out)", "presence nll (kept)"], "rows": rows}


def round2_temporal():
    rows = [{"name": "control (2-channel)", "vals": ["—", 0.4507, 0.6044], "em": True}]
    for name, path, cov in [("+ start_date rank", "rating_floors_temporal_startday.json", "18.4%"),
                            ("+ updated_at rank", "rating_floors_temporal_updsec.json", "100%")]:
        f = load_if(path)
        if f:
            m, r = tm(f)
            rows.append({"name": name, "vals": [cov, m, r]})
    return {"cols": ["coverage", "trusted MAE", "trusted ρ"], "rows": rows}


def round2_era():
    rows = []
    for label, path in [("control probe", "temporal_era_debias_slopeonly_lb30.json"),
                        ("seed-2 probe", "temporal_era_debias_seed2.json"),
                        ("seed-2 + EASE blend", "temporal_era_debias_blend.json")]:
        f = load_if(path)
        if not f:
            continue
        r, e = f["raw"], f["era_debias"]
        rows.append({"name": label, "vals": [r["all"]["mae"], r["late"]["mae"], f"{r['late']['bias']:+.4f}"]})
        rows.append({"name": "  + era-slope debias", "vals": [e["all"]["mae"], e["late"]["mae"], f"{e['late']['bias']:+.4f}"],
                     "good": [0, 1], "em": label == "seed-2 + EASE blend"})
    return {"cols": ["trusted MAE (all rows)", "late-era MAE (serve proxy)", "late-era bias"], "rows": rows}


def round2_absch():
    f = load_if("rating_floors_absch.json")
    if not f:
        return None
    s2 = load_if("rating_floors_absch_s2.json")
    note = ("First seed: <b>trusted −0.0020 vs control / −0.0029 vs seed-2, outside the seed band</b>, "
            "ρ top-of-band, and the α-clip stratum (generous raters, where the alpha-mix destroys the "
            "1-vs-2 distinction) improves 0.2702→0.2681 — mechanism-consistent, not noise. In-model "
            "change: ships at the next full retrain, pairing naturally with the EASE-graft recipe. "
            "Open design caveat: a z-duplicate control would separate abs-information from generic "
            "second-channel capacity.")
    verdict, chip = "WIN — SEED 2 RUNNING", "gold"
    if s2:
        m2, r2 = tm(s2)
        band = "CONFIRMED" if m2 < 0.4507 - 0.0005 else "NOT CONFIRMED"
        note += f" Second seed: trusted MAE {m2:.4f} / ρ {r2:.4f} — {band.lower()}."
        verdict, chip = (f"WIN {band}", "pass") if band == "CONFIRMED" else ("SEED-2 CONTRADICTS", "null")
    return probe_panel(A / "rating_floors_absch.json", "abs-channel", note, verdict=verdict, chip=chip)


def round2_bothfixed():
    f = load_if("rating_floors_split_bothfixed.json")
    if not f:
        return None
    pr = jsonl_last("probe/probe_split_bothfixed.jsonl")
    m, r = tm(f)
    h = pr["holdout"]["corrupt"] if pr else None
    matches = abs(m - 0.4507) < 0.0015 and (h is None or abs(h["nll_drop_per_item"] - 7.0611) < 0.015)
    note = (f"Fixed equal weights land at trusted {m:.4f} / ρ {r:.4f}"
            + (f", presence nll_drop {h['nll_drop_per_item']:.4f}" if h else "") +
            " vs learned-weighting control 0.4507 / 0.6044 / 7.0611. " +
            ("<b>Within the seed band on both heads — the multitask benefit is multitasking itself; "
             "the uncertainty-weighting scheme is exonerated (and is doing ~nothing beyond loss "
             "bookkeeping).</b>" if matches else
             "<b>Materially different from the learned-weighting control — training dynamics from the "
             "weighting scheme are load-bearing; the split verdict stands but the mechanism attribution "
             "shifts.</b>"))
    verdict = "MATCHES — WEIGHTING EXONERATED" if matches else "DYNAMICS MATTER"
    return probe_panel(A / "rating_floors_split_bothfixed.json", "fixed-weight", note,
                       verdict=verdict, chip="pass" if matches else "gold")


def probe_panel(path, label, note, verdict="SEE TABLE", chip="gold"):
    p = Path(path)
    if not p.exists():
        return None
    f = json.load(open(p))
    rows = [
        {"name": "overall MAE", "c": fc["model"]["overall"]["mae"], "p": f["model"]["overall"]["mae"]},
        {"name": "trusted MAE", "c": fc["model"]["by_trust"][0]["mae"], "p": f["model"]["by_trust"][0]["mae"]},
        {"name": "one-sitting MAE", "c": fc["model"]["by_trust"][1]["mae"], "p": f["model"]["by_trust"][1]["mae"]},
        {"name": "degenerate MAE", "c": fc["model"]["by_trust"][2]["mae"], "p": f["model"]["by_trust"][2]["mae"]},
        {"name": "deep-tail MAE (3k–6k)", "c": fc["model"]["by_tier"][3]["mae"], "p": f["model"]["by_tier"][3]["mae"]},
        {"name": "rho (within-user)", "c": fc["model"]["ordering"]["rho_mean"], "p": f["model"]["ordering"]["rho_mean"]},
    ]
    return {"label": label, "rows": rows, "note": note, "verdict": verdict, "chip": chip}


DATA = {
    "kpis": [
        {"label": "Rating-gradient mask", "value": "exact", "detail": "0 leaks at 3 levels: data, loss, end-to-end grads"},
        {"label": "Twin floor, trusted-only", "value": "0.25–0.30", "detail": "unchanged after removing untrusted raters — the noise is human"},
        {"label": "Within-user ordering ρ", "value": "0.60", "detail": "vs 0.50 item-quality baseline and ≈0.85 noise ceiling"},
        {"label": "Degenerate free-win", "value": "0.140", "detail": "model MAE on degenerate raters vs 0.451 trusted — aggregate MAE is polluted"},
    ],
    "pop_hist": pop_hist.tolist(),
    "mask": [
        ["Data", "366.7M entries: 0 duplicate indices; mu-fill constant per user; dropped-sentinel strictly below it; 100k users re-derived from raw CSV match bit-for-bit", "0 violations"],
        ["Loss", "∂loss/∂pred exactly zero at every unrated position, nonzero at every rated one", "exact"],
        ["End-to-end", "prod checkpoint, real batch: every param gradient bitwise invariant to unrated-target perturbation; single rated perturbation detected", "bitwise"],
    ],
    "clusters": [dict(c, name=CLUSTER_NAMES[i]) for i, c in enumerate(census["clusters"])],
    "taxonomy": [
        {"name": "flagged degenerate (current rule)", "pct": census["pct_degenerate"], "sigma": None, "tstd": None},
        {"name": "one-sitting raters", "pct": census["pct_one_sitting"], "sigma": None, "tstd": None},
    ] + [
        {"name": lab, "pct": census["taxonomy"][k]["pct_of_rated"],
         "sigma": census["taxonomy"][k]["mean_sigma"], "tstd": census["taxonomy"][k]["mean_target_std"]}
        for k, lab in [
            ("all_one_score", "all one score"), ("mode_ge_090", "mode ≥ 90%"),
            ("mean_ge_95", "mean ≥ 9.5"), ("never_below_7", "never below 7"),
            ("never_below_5", "never below 5"), ("distinct_le_3", "≤3 distinct scores"),
            ("sigma_lt_05", "σ < 0.5"), ("full_range_8plus", "≥8 distinct scores"),
            ("uses_1_and_10", "uses both 1 and 10"),
        ]
    ],
    "ladder": [
        {"name": "global mean", "mae": fc["global_mean"]["overall"]["mae"], "color": "muted"},
        {"name": "item mean (global quality)", "mae": fc["item_mean"]["overall"]["mae"], "color": "gold"},
        {"name": "+ per-user offset (additive)", "mae": fc["additive_feasible"]["overall"]["mae"], "color": "gold"},
        {"name": "model — control probe (clean)", "mae": fc["model"]["overall"]["mae"], "color": "blue", "em": True},
        {"name": "model — prod ckpt (trained on eval users)", "mae": fp["model"]["overall"]["mae"], "color": "blue"},
    ],
    "variance": {"item": 0.2472, "additive": 0.3379, "model": 0.4562, "max_explainable": 0.7992},
    "sigma_strata": [
        {"label": "σ < 0.7", "mae": fc["model"]["by_sigma"][0]["mae"]},
        {"label": "0.7–1.1", "mae": fc["model"]["by_sigma"][1]["mae"]},
        {"label": "1.1–1.6", "mae": fc["model"]["by_sigma"][2]["mae"]},
        {"label": "σ ≥ 1.6", "mae": fc["model"]["by_sigma"][3]["mae"]},
    ],
    "strata": strata(),
    "degen_note": ("<b>The dec2025 “trust filtering is harmful” verdict was judged on aggregate LOO MAE with "
                   "degenerate raters in the eval set.</b> Their rows cost the model 0.140 MAE — near-constant "
                   "targets copyable from the input channel — so keeping their gradient looks good in aggregate "
                   "regardless of what it does to everyone else. Every probe in §6 is judged on trusted rows only."),
    "rho": {
        "ceil_lo": 0.68, "ceil_hi": 0.89,
        "points": [
            {"name": "item quality", "v": fc["item_mean"]["ordering"]["rho_mean"], "color": "muted"},
            {"name": "control", "v": fc["model"]["ordering"]["rho_mean"], "color": "blue"},
            {"name": "prod", "v": fp["model"]["ordering"]["rho_mean"], "color": "blue", "lift": 16},
        ],
    },
    "twins": twin_rows(),
    "rprior": [
        {"name": "overall MAE", "c": fc["model"]["overall"]["mae"], "p": fr["model"]["overall"]["mae"]},
        {"name": "trusted MAE", "c": fc["model"]["by_trust"][0]["mae"], "p": fr["model"]["by_trust"][0]["mae"]},
        {"name": "deep-tail MAE (3k–6k)", "c": fc["model"]["by_tier"][3]["mae"], "p": fr["model"]["by_tier"][3]["mae"]},
        {"name": "rho (within-user)", "c": fc["model"]["ordering"]["rho_mean"], "p": fr["model"]["ordering"]["rho_mean"]},
    ],
    "trustmask": probe_panel(A / "rating_floors_trustmask.json", "trustmask",
        "Paired run: identical user split, eval rows, and recipe; only the rated flags of untrusted "
        "users are zeroed. <b>Trusted rows move +0.0002 MAE / −0.0008 rho — a no-op for real users.</b> "
        "All movement is on the untrusted rows themselves (the model stops memorizing their constants). "
        "The dec2025 “harmful” verdict was eval pollution; equally, “untrusted ratings are "
        "net-informative” is dead. Prod data prep stays cleanup_notrust: simpler, presence co-occurrence "
        "still counts, and the degenerate rating gradient is harmless.",
        verdict="NO EFFECT ON TRUSTED — CLOSED", chip="null"),
    "ztarget": probe_panel(A / "rating_floors_ztarget_conv.json", "pure-z target",
        "Predictions converted to mixed units via the per-user affine (user stats are known at serve, "
        "so this is a legitimate serving configuration — though it hands the model its calibration). "
        "<b>Ordering does not improve: rho −0.005 in every stratum.</b> The MAE gains concentrate "
        "exactly where calibration-externalization pays (degenerate −0.040, σ<0.7 −0.031) — the "
        "component §4 showed was already solved. Since pure-z also equalizes per-user target scale, "
        "this result argues against heteroscedastic loss-weighting too. Not a winner.",
        verdict="NULL ON ORDERING", chip="null"),
    "newinfo": (lambda paths: [
        {"name": n, "mae": (j := json.load(open(A / p)))["model"]["by_trust"][0]["mae"],
         "rho": j["model"]["ordering"]["rho_by_trust"][0]["rho_mean"],
         "tail": j["model"]["by_tier"][3]["mae"], "lowsig": j["model"]["by_sigma"][0]["mae"], "em": em}
        for n, p, em in paths if (A / p).exists()
    ])([
        ("control (NN head)", "rating_floors_control.json", False),
        ("graft: presence-EASE concat", "rating_floors_ratgraft_presence.json", False),
        ("graft: residual-EASE concat", "rating_floors_ratgraft_residual.json", False),
        ("graft: residual-EASE gate", "rating_floors_ratgraft_gate.json", False),
        ("blend: fixed w=0.5", "rating_floors_blend_w0.5.json", False),
    ]) + [{"name": "blend: σ-ramp w=0.5·min(σᵤ,1) — SHIP", "mae": 0.4445, "rho": 0.6127,
           "tail": 0.4842, "lowsig": 0.2177, "em": True}],
    "split": round2_split(),
    "temporal": round2_temporal(),
    "era": round2_era(),
    "absch": round2_absch(),
    "bothfixed": round2_bothfixed(),
    "implications": [
        "<b>The plumbing is clean.</b> Gradient masking is exact at every level; data-side corruption is not where rating error comes from.",
        "<b>Aggregate MAE is the wrong scoreboard.</b> It mixes a 0.14-MAE trivial task (degenerates), a 0.22 near-constant task (σ<0.7), and a 0.52 hard task (σ≥1.6). Rating-head changes get judged on trusted-user MAE + within-user rho.",
        "<b>Calibration is done; ordering is the frontier.</b> Per-user bias is 0.13 std and debiasing helps nothing. The head's value-add is +0.10 rho over item quality, with ~0.25 rho of provable headroom below the noise ceiling.",
        "<b>The floor is human.</b> Trusted-only twins reproduce the 0.25–0.30 z-MAE floor almost exactly. Data cleaning cannot lower it; only richer information (content, context) can move the model closer to it.",
        "<b>Item-mean prior: closed.</b> The rating analog of the logQ prior is a clean null — item means are the easy 25% of the signal, and there is no serving distortion to fix on the rating side.",
        "<b>Trust axis: closed, confound resolved.</b> Masking untrusted raters' gradients is a no-op for trusted users (+0.0002 MAE). The dec2025 “harmful” read was degenerate rows inflating aggregate eval; keeping cleanup_notrust is fine — just never score on those rows.",
        "<b>Target engineering is exhausted.</b> Item-prior null, pure-z null-on-ordering, trust-mask null-on-trusted — within-user ordering is invariant to how the target is parameterized. The head already extracts what the CF signal + sum-pooled bottleneck offer.",
        "<b>The remaining levers are new information, not new losses — and the first one worked.</b> The rating-residual item-item channel (§7) is a confirmed, seed-robust, strictly-dominating win as a σ-ramped serve-side blend: trusted MAE −0.006, trusted ρ +0.008, deep tail −0.008, zero retraining. The presence co-occurrence channel adds nothing (B1 null) — it's the residual structure specifically.",
        "<b>In-model integration lost to the post-hoc blend three times</b> (input-graft in the audit era, concat, gate). The per-item channel doesn't survive projection, and a learned gate under-weights it. Revisit at the next full retrain with a per-item additive design initialized at the blend optimum; until then the blend is the ship.",
        "<b>Round 2 — the architecture is right.</b> Splitting into presence-only and rating-only models loses on both sides: the shared bottleneck is a mutual regularizer (rating gradients stop presence memorization; presence gradients feed the rating head). One model, one representation, kept on merit, not just elegance.",
        "<b>Round 2 — time can't enter the bag, but it can fix the exit.</b> The temporal input channel is structurally null under the blind-bag contract (the target's own era is hidden and eval eras are uniform). The same signal ships as a closed-form serve-side era-slope debias instead: −0.002 trusted MAE on both seeds, stacking with the EASE blend for a cumulative −1.6% serve stack, and it removes the +0.024 current-taste over-prediction bias — the number users actually see.",
        "<b>Round 2 — the input feature was leaving information on the table.</b> An absolute-score channel next to the z-mix is worth −0.002 to −0.003 trusted MAE (seed-2 pending), concentrated in the α-clip stratum the census predicted. Queue it for the next full retrain alongside the EASE-graft.",
        "<b>Untried levers:</b> content features (also the tail fix — MAE degrades 0.43→0.49 where evidence thins), un-clipped low scores.",
    ],
    "methods": (
        "Protocol: 20k probe-holdout users (seed 999 split, never trained by probes), train-style corruption "
        "(drop 40%±40%, seed 555), all metrics on dropped-rated items (1.346M rows; kept-rated 2.03M for feasible "
        "debiasing). Scripts: <span class=\"mono\">analysis/rating_floors_dump.py</span> / "
        "<span class=\"mono\">rating_floors_analyze.py</span> (baselines: shrunk item means λ=50 from train users, "
        "LOO-style user offsets λ=10 from kept items), <span class=\"mono\">rating_mask_verify.py</span> (3 stages), "
        "<span class=\"mono\">rating_census_extract.py</span> + <span class=\"mono\">rating_scheme_census.py</span> "
        "(k-means k=10 on L1-normalized score histograms, 400k sample, users ≥10 ratings), "
        "<span class=\"mono\">twin_noise.py --user-mask</span> (trusted = not one-sitting, not degenerate; original "
        "all-population bins recovered from the audit artifact), <span class=\"mono\">reconstruct_raw_scores.py</span> "
        "(exact alpha-mix inversion, max dev 3.7e-6), <span class=\"mono\">compute_rating_prior.py</span> + "
        "<span class=\"mono\">train_probe_ratingprior.py</span> / <span class=\"mono\">train_probe_ztarget.py</span> "
        "(train_probe protocol: PRNGKey 0, full train pool, 50k steps). Rho ceiling: per-user attenuation bound "
        "√(1−σε²/Var_u(target)) over ≥8-item users; σε from genuine-band twins. Vectors: "
        "<span class=\"mono\">user_input_vectors_cleanup_notrust.npz</span> (1,573,921 users; census row-aligned, "
        "verified). Prod-vs-control gap includes prod having trained on the eval users. Round 2: "
        "<span class=\"mono\">temporal_extract.py</span> / <span class=\"mono\">raw_temporal_extract.py</span> "
        "(entry-aligned start_date / updated_at, alignment exact on all 1,573,921 users), "
        "<span class=\"mono\">temporal_signal_check.py</span> (permutation nulls), "
        "<span class=\"mono\">train_probe_split.py</span> / <span class=\"mono\">train_probe_temporal.py</span> / "
        "<span class=\"mono\">train_probe_absch.py</span> (same probe protocol, inline floors dumps), "
        "<span class=\"mono\">temporal_era_analyze.py</span> / <span class=\"mono\">temporal_era_debias.py</span> "
        "(slope fit on kept rows, λ_a=∞ / λ_b=30, judged on drop rows; late-era = era rank &gt; 0.8)."
    ),
}

blob = json.dumps(DATA)
html = open(SP / "rating_head.html").read().replace("__DATA__", blob)
out = SP / "rating_head_built.html"
out.write_text(html)
print(f"built {out} ({len(html)//1024}KB)")
