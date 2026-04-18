# 006 — FastEmbed `Qdrant/bm25` instead of a hand-rolled tokenizer

- **Status:** accepted
- **Date:** 2026-04-18 (after a 3-iteration learning loop)
- **Deciders:** @heitor-am

## Context

The first sparse-encoder implementation was a 30-line tokenizer: `re.findall(r"\w+", text.lower())` plus a hardcoded English stopword list. With this, **hybrid retrieval performed *worse* than dense-only** on the eval set:

| Strategy | P@1 | R@5 | NDCG@5 |
|---|---|---|---|
| dense_only | 0.722 | 0.972 | 0.804 |
| hybrid (naïve BM25) | **0.667** | **0.944** | **0.755** |

Investigation traced the regression to two root causes:

1. **No stemming.** The query "my app feels slow" produced sparse hits on every chunk containing the literal "slow" — but the article we wanted talked about "slower" / "slowing", which the naïve tokenizer treated as different terms.
2. **Stopword guesswork.** Common pronouns ("my", "your") dominated TF scoring on the small corpus, surfacing irrelevant chunks.

A second iteration added a manual stopword list. Hybrid got *worse still*, because removing stopwords without stemming amplified the morphological-variant problem.

The third iteration replaced the hand-rolled code with FastEmbed's `Qdrant/bm25` model. Hybrid now matches dense-only; reranker pulls cleanly ahead.

## Decision

Use `fastembed.SparseTextEmbedding(model_name="Qdrant/bm25")`. This is the *canonical* tokenizer Qdrant expects for hybrid search:

- Per-language stopword removal (Snowball stopword lists)
- Snowball stemmer (collapses `slow` / `slower` / `slowing`)
- Hashed token indices (no vocabulary file to maintain)
- TF-only output values (Qdrant applies IDF server-side via `Modifier.IDF`, see ADR-004)

Lazy singleton; one model instance for the process lifetime.

## Consequences

**Positive:**
- Hybrid stopped underperforming dense (eval verified — see PR #16 / `docs/eval-results-prod.txt`).
- The tokenizer FastEmbed produces is byte-identical to the one Qdrant assumes when applying IDF — no impedance mismatch.
- We delete a class of bugs we don't want to think about: stopword coverage, language detection, stemmer variants.

**Negative:**
- ~50MB model download on first import. Mitigated by Docker pre-warm (ADR-014).
- One more dep (`fastembed>=0.5`).
- Default model is English; multilingual corpora would need a different `model_name`.

**Trade-offs accepted:**
- We ship a binary model file in the production image. Acceptable — it's still smaller than the FastAPI dependency tree.

## Alternatives considered

- **Hand-rolled tokenizer with `nltk` stopwords + Snowball** — would work, but: (a) we'd be re-implementing FastEmbed's pipeline; (b) the IDF modifier on the server expects FastEmbed's hashed indices specifically; (c) "don't reinvent the wheel" — the user's own feedback after iteration 2.
- **`rank-bm25`** (in-process) — was the original plan in the PRD. Rejected once we moved to Qdrant Query API + RRF: doing BM25 in-process means *we* become the source of truth for the inverted index, which Qdrant already maintains.

## References

- `app/shared/qdrant/sparse_encoder.py`
- `tests/shared/test_qdrant_sparse_encoder.py`
- FastEmbed: https://github.com/qdrant/fastembed
- The eval-results trail: `docs/eval-results.txt` (dev) and `docs/eval-results-prod.txt`
