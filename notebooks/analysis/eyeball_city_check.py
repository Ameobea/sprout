"""One-off: why does City The Animation top the web app but not offline lists?"""
import csv, json, sys
import numpy as np
sys.path.insert(0, ".")
import jax.numpy as jnp
from model import CONF
from analysis.frontier_eval import build_components
from analysis.probe_value_eval import load_graft, load_prod
from analysis.eyeball_hybrid import *  # noqa

CITY = 59898
NEW_ITEM = 63403

cs = CONF["corpus_size"]
d = np.load("../data/aug2026/user_input_vectors_cleanup_notrust.npz")
indices = d["indices"].astype(np.int32)
counts = np.bincount(indices, minlength=cs).astype(np.float64)
log_pop = np.log(np.maximum(counts, 1.0) / np.maximum(counts, 1.0).sum())

corpus_ids = np.array(json.load(open("../data/corpus_ids_aug2026.json")), dtype=np.int64)
id_to_idx = {int(a): i for i, a in enumerate(corpus_ids)}
titles = {}
for row in csv.DictReader(open("../data/processed-metadata_aug2026.csv", encoding="utf-8")):
    titles[int(row["id"])] = row["title_english"] or row["title"]
pop_rank = np.argsort(np.argsort(-counts))
dsu = build_components("../data/processed-metadata_aug2026.csv")
comp_of_corpus = np.array([dsu.find(int(a)) for a in corpus_ids], dtype=np.int64)

B = np.load("../data/aug2026/ease_B6k_lam200.npy")
nn_params, nn_fwd = load_prod("../data/aug2026/jax_model_fresh_logq.msgpack", 512, cs)
cc_params, cc_fwd = load_graft("../data/aug2026/probe/probe_graft_concat.msgpack", "concat", cs=6000, bd=512, B=B) \
    if False else load_graft("../data/aug2026/probe/probe_graft_concat.msgpack", "concat", 512, cs, B)

from model import Recommender
from analysis.train_probe_graft import GraftRecommender
nn_model = Recommender()
cc_model = GraftRecommender(mode="concat")
Bj = jnp.asarray(B, dtype=jnp.float32)

fix = json.load(open("eval/fixtures/ameo___.json"))
prof = {}
for e in fix:
    ls = e.get("list_status") or {}
    sc = ls.get("score", 0) or 0
    if ls.get("status") == "plan_to_watch" and sc == 0:
        continue
    ci = id_to_idx.get(e["node"]["id"])
    if ci is not None:
        prof[ci] = sc

ci_city = id_to_idx.get(CITY)
ci_new = id_to_idx.get(NEW_ITEM)
print(f"City idx {ci_city} pop_rank {pop_rank[ci_city]} | new item {NEW_ITEM} ({titles.get(NEW_ITEM)}) idx {ci_new}")
prof_comps = None

def build_x(pr):
    owned = np.array(sorted(pr), dtype=np.int64)
    rated = np.array([pr[c] for c in owned], dtype=np.float64)
    rz = np.where(rated > 0, (rated - rated[rated > 0].mean()) / (rated[rated > 0].std() + 1e-9), 0.0)
    x = np.zeros((1, cs * 2), dtype=np.float32)
    x[0, owned] = 1.0
    x[0, cs + owned] = rz
    return owned, jnp.asarray(x)

def outputs(x):
    nn_lg = np.array(nn_fwd(nn_params, x))[0].astype(np.float64)
    cc_lg = np.array(cc_fwd(cc_params, x))[0].astype(np.float64)
    _, rt_n, _, _ = nn_model.apply({"params": nn_params}, x, training=False)
    e = x[:, :cs] @ Bj
    e = (e - jnp.mean(e, axis=1, keepdims=True)) / (jnp.std(e, axis=1, keepdims=True) + 1e-6)
    _, rt_c, _, _, _ = cc_model.apply({"params": cc_params}, x, e, training=False)
    return {"prod": (nn_lg, np.array(rt_n)[0].astype(np.float64)),
            "hybrid": (cc_lg, np.array(rt_c)[0].astype(np.float64))}

CFGS = [("offline prodmix (lw.3, neutral)", 0.3, 1.0, 0),
        ("webapp old default (lw_eff.186, a.7 k2750)", 0.186, 0.7, 2750),
        ("webapp new default (lw_eff.354, a.7 k2750)", 0.354, 0.7, 2750),
        ("presence-only, web knob (a.7 k2750)", 1.0, 0.7, 2750)]

for pname, pr in [("stale-75", dict(prof)), ("fresh-76", ({**prof, ci_new: 10} if ci_new is not None else dict(prof)))]:
    owned, x = build_x(pr)
    outs = outputs(x)
    franch = np.isin(comp_of_corpus, np.unique(comp_of_corpus[owned]))
    print(f"\n=== profile {pname} ({len(owned)} items) City-in-profile-franchise: {bool(franch[ci_city])}")
    for mname, (lg, rt) in outs.items():
        for label, lw, alpha, k in CFGS:
            lam = counts / (counts + k) if k else np.ones(cs)
            s = lam * lg + alpha * log_pop
            p = np.exp(s - s.max()); p /= p.sum()
            sc = np.power(p, lw) * np.power(np.maximum(rt + 1, 1e-3), 1 - lw)
            sc[owned] = -np.inf
            order = np.argsort(-sc)
            rank = int(np.where(order == ci_city)[0][0])
            top5 = ", ".join(titles.get(int(corpus_ids[i]), "?") for i in order[:5])
            print(f"  {mname:6s} | {label:44s} City rank {rank:4d} | top5: {top5}")
