"""
Canonical user-profile -> model-input preprocessing, shared by the python model
server and the eval harness. The Rust inference server (model-server-rs) mirrors
this logic and must be kept in sync if it changes.
"""

import numpy as np

from normalize_ratings import normalize_ratings

STATUS_KEEP = ("completed", "watching", "dropped")


def filter_profile_entries(entries, id_to_idx, restrict_ids=None):
    """
    entries: iterable of (anime_id, rating 0-10 with 0=unrated, watch_status)
    Returns [(corpus_idx, anime_id, rating, status)] sorted by corpus_idx, keeping
    only in-corpus items that are rated or in a watched-ish status.
    """
    kept = []
    for anime_id, rating, status in entries:
        idx = id_to_idx.get(anime_id)
        if idx is None:
            continue
        if restrict_ids is not None and anime_id not in restrict_ids:
            continue
        if rating > 0 or status in STATUS_KEEP:
            kept.append((idx, anime_id, rating, status))
    kept.sort()
    return kept


def vectorize_entries(kept):
    """
    kept: output of filter_profile_entries.
    Returns (corpus_indices, normalized_ratings, original_ratings, norm_stats).
    Dropped-without-rating becomes the -2 sentinel consumed by normalize_ratings.
    """
    corpus_indices = np.array([k[0] for k in kept], dtype=np.int32)
    original = np.array(
        [(-2 if s == "dropped" and r == 0 else r) for _, _, r, s in kept],
        dtype=np.float32,
    )
    normalized, stats = normalize_ratings(original)
    return corpus_indices, normalized.astype(np.float32), original, stats
