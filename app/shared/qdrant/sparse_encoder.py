"""BM25 sparse encoder, backed by FastEmbed's ``Qdrant/bm25`` model.

This replaces a naïve, hand-rolled tokenizer (lowercase regex +
hardcoded stopword list) that — empirically — dragged hybrid search
below dense-only in eval:

  - no stemming: ``slow`` didn't match ``slower``
  - stopword list coverage was a guess game
  - no language awareness

FastEmbed's ``Qdrant/bm25`` model ships the canonical tokenizer Qdrant
expects for hybrid search: per-language stopword removal, Snowball
stemmer, hash-based indices, and values normalised so Qdrant's
``Modifier.IDF`` on the collection applies cleanly. It's the purpose-
built counterpart to the Qdrant Query API / RRF fusion path — no point
rolling our own when the vendor publishes the right thing.

Trade-offs:
    - First import triggers a ~50MB model download (cached under
      ``~/.cache/fastembed``). Pre-download in the Docker image so
      cold-starts on Fly.io don't pay the cost.
    - Default language is English, matching the dev.to corpus used
      here. Multilingual corpora would need a different model id.
"""

from __future__ import annotations

from fastembed import SparseTextEmbedding

_MODEL_NAME = "Qdrant/bm25"

# Lazy singleton — constructing SparseTextEmbedding downloads the model
# on first call. One encoder instance is fine across the app (thread-
# safe for inference) and avoids repeated init cost.
_model: SparseTextEmbedding | None = None


def _get_model() -> SparseTextEmbedding:
    global _model
    if _model is None:
        _model = SparseTextEmbedding(model_name=_MODEL_NAME)
    return _model


def encode_bm25_sparse(text: str) -> tuple[list[int], list[float]]:
    """Encode text as (indices, values) for Qdrant sparse storage.

    Delegates to ``FastEmbed``'s ``Qdrant/bm25`` model which handles
    lowercasing, stopword filtering, Snowball stemming, and hashed
    index generation. Qdrant applies the IDF modifier server-side
    (collection schema declares ``modifier=IDF``), so ``values`` here
    are TF-normalised but pre-IDF.

    Empty / whitespace-only input returns empty arrays so callers can
    skip upserting a sparse vector rather than sending an empty one
    (Qdrant rejects empty sparse vectors).
    """
    if not text or not text.strip():
        return [], []
    [embedding] = list(_get_model().embed([text]))
    indices = [int(i) for i in embedding.indices]
    values = [float(v) for v in embedding.values]
    return indices, values
