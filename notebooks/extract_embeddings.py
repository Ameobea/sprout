import jax
import jax.numpy as jnp
from flax import serialization
import numpy as np

# Import your existing code
from model import Recommender, CONF


def load_params(model_path, rc=False):
    """Replicates your loading logic to get the param tree"""
    rng = jax.random.PRNGKey(0)
    if rc:
        import sys
        sys.path.insert(0, "analysis")
        from train_probe_rc import RCRecommender
        model = RCRecommender()
        params = model.init(
            {"params": rng, "noise": rng},
            jnp.ones((1, CONF["corpus_size"] * 3)),
            jnp.ones((1, CONF["corpus_size"])),
        )["params"]
    else:
        model = Recommender()
        dummy_input = jnp.ones((1, CONF["corpus_size"] * 2))
        params = model.init({"params": rng, "noise": rng}, dummy_input)["params"]

    with open(model_path, "rb") as f:
        saved_bytes = f.read()

    return serialization.from_bytes(params, saved_bytes)


def get_item_embeddings(model_path, rc=False):
    # RC (3ch) checkpoints keep the same [presence | rating] slice for atlas
    # continuity; the abs-channel rows are ignored.
    params = load_params(model_path, rc=rc)

    # 1. Access the first layer weights
    first_layer_weights = params["Dense_0"]["kernel"]

    corpus_size = CONF["corpus_size"]

    # 2. Slice the weights
    # Rows 0 to 5999 correspond to the "Presence" (0/1) inputs
    w_presence = first_layer_weights[:corpus_size, :]

    # Rows 6000 to 11999 correspond to the "Rating" inputs
    w_rating = first_layer_weights[corpus_size : 2 * corpus_size, :]

    # 3. Concatenate to create the rich embedding
    # Result shape: (6000, 4096)
    # This represents "How the model sees this item" both as a binary presence
    # and as a rated object.
    embeddings = jnp.concatenate([w_presence, w_rating], axis=1)

    return embeddings


# Usage
if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else "../data/jax_model.msgpack"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "../data/anime_embeddings.npy"
    embeddings = get_item_embeddings(model_path, rc="--rc" in sys.argv)

    # Optional: L2 Normalize before saving (helps UMAP significantly)
    norms = jnp.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    print(f"Extracted embeddings shape: {embeddings.shape}")

    # Save for your visualization tool
    np.save(out_path, np.array(embeddings))
