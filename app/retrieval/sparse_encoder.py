"""BM25-style sparse vector encoder.

Produces (indices, values) pairs that pair with Qdrant's sparse vector
support. Qdrant applies the IDF modifier server-side — we only ship raw
term frequencies plus a stable hash per token.

Why stable hashes instead of a vocabulary?
    A vocabulary dict would need to be shared between ingestion and search,
    persisted across restarts, and rebuilt whenever a new term showed up.
    Hashing sidesteps all of that. Collisions are vanishingly rare at
    32 bits (~1 in 4 billion) and merely merge two tokens' contributions;
    the resulting sparse vector is still semantically sound.

Why MD5 instead of Python's `hash()`?
    `hash()` is randomised per interpreter start, so the same token would
    get different indices in the ingest process and the search process.
    MD5 is stable, deterministic, and fast enough — we hash a few dozen
    tokens per query, not a hot path.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Same normalisation used for dense
    content (chunk payload) so queries hit the terms they'd expect."""
    return _TOKEN_RE.findall(text.lower())


def _stable_hash_32(token: str) -> int:
    """32-bit deterministic token → index.

    `usedforsecurity=False` flags this to bandit / CWE-327 as a
    non-security use: collisions are tolerable (they merge two tokens'
    contributions), and we're not authenticating anything — we're just
    hashing strings into a fixed-width index space that Qdrant stores.
    """
    return int.from_bytes(
        hashlib.md5(token.encode("utf-8"), usedforsecurity=False).digest()[:4],
        "big",
    )


def encode_bm25_sparse(text: str) -> tuple[list[int], list[float]]:
    """Encode text as (indices, values) for Qdrant sparse storage.

    The values are raw term frequencies. Qdrant applies the BM25 IDF
    modifier server-side (collection is created with `modifier=IDF`),
    so we don't multiply by IDF here.

    Empty or token-less text yields empty lists — callers should skip
    upserting a sparse vector in that case, not send empty vectors.
    """
    tokens = tokenize(text)
    if not tokens:
        return [], []
    counts = Counter(tokens)
    # Deterministic ordering so two encodes of the same text produce the
    # exact same arrays — helps upsert idempotency.
    items = sorted(counts.items())
    indices = [_stable_hash_32(token) for token, _ in items]
    values = [float(count) for _, count in items]
    return indices, values
