"""
Build the atlas embedding JSON in the exact schema the webapp expects
(src/embedding.ts RawEmbedding — the emblaze-era format, NOT the flat array
that project_model_embedding.ipynb's export cell produces).

Two-step flow: extract_embeddings.py (rocm container) -> this (jupyter container).

Usage: gen_atlas_embedding.py <anime_embeddings.npy> <corpus_ids.json> <processed-metadata.csv> <out.json>
"""

import json
import sys

import numpy as np
import pandas as pd
import umap

N_NEIGHBORS = 20
UMAP_PARAMS = dict(n_components=2, n_neighbors=20, min_dist=0.01, metric="cosine", random_state=42)


def main():
    emb_path, corpus_path, metadata_path, out_path = sys.argv[1:5]
    emb = np.load(emb_path)
    corpus_ids = json.load(open(corpus_path))
    assert len(corpus_ids) == emb.shape[0]

    md = pd.read_csv(metadata_path).set_index("id")
    years = md["aired_from_year"].to_dict()
    counts = md["rating_count"].to_dict()

    proj = umap.UMAP(**UMAP_PARAMS).fit_transform(emb)

    # embeddings come L2-normalized from extract_embeddings.py -> cosine knn via dot
    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    nn = np.argsort(-sims, axis=1)[:, :N_NEIGHBORS]

    out = {
        "ids": corpus_ids,
        "metric": "cosine",
        "n_neighbors": N_NEIGHBORS,
        "neighbors": {
            "_format": "expanded",
            "metric": "cosine",
            "n_neighbors": N_NEIGHBORS,
            "neighbors": {str(i): [int(j) for j in nn[i]] for i in range(len(corpus_ids))},
        },
        "points": {
            str(i): {
                "x": round(float(proj[i, 0]), 4),
                "y": round(float(proj[i, 1]), 4),
                "color": int(years.get(aid, 0)),
                "r": round(float(np.log(max(counts.get(aid, 1), 1))), 4),
            }
            for i, aid in enumerate(corpus_ids)
        },
    }
    with open(out_path, "wt") as f:
        json.dump(out, f)
    print(f"wrote {len(corpus_ids)} points -> {out_path}")


if __name__ == "__main__":
    main()
