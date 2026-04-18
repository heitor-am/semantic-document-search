from __future__ import annotations

from app.shared.qdrant.sparse_encoder import encode_bm25_sparse


class TestEncodeBm25Sparse:
    def test_returns_aligned_arrays_for_content_text(self) -> None:
        indices, values = encode_bm25_sparse("python async programming")
        assert len(indices) == len(values) > 0
        assert all(isinstance(i, int) for i in indices)
        assert all(isinstance(v, float) for v in values)

    def test_empty_text_returns_empty_vectors(self) -> None:
        assert encode_bm25_sparse("") == ([], [])

    def test_whitespace_only_returns_empty_vectors(self) -> None:
        assert encode_bm25_sparse("   \n\t  ") == ([], [])

    def test_deterministic_across_calls(self) -> None:
        # Same text must produce byte-identical arrays so Qdrant upsert
        # is idempotent.
        a = encode_bm25_sparse("rust systems programming")
        b = encode_bm25_sparse("rust systems programming")
        assert a == b

    def test_stemming_unifies_morphological_variants(self) -> None:
        # Snowball stems regular -ing / -ed / plural suffixes. "testing"
        # / "tested" / "tests" / "test" should collapse to a single
        # index. (Irregular forms like "ran"/"run" don't unify — that's
        # a known stemmer limitation, not a bug here.)
        variants = ["test", "tests", "tested", "testing"]
        index_sets = [set(encode_bm25_sparse(v)[0]) for v in variants]
        common = index_sets[0]
        for s in index_sets[1:]:
            common &= s
        assert common, (
            f"stemmer should collapse {variants} to a single stem; "
            f"got disjoint indices: {index_sets}"
        )

    def test_stopwords_are_filtered(self) -> None:
        # "the and of" is all stopwords → nothing left to index.
        indices, _ = encode_bm25_sparse("the and of")
        assert indices == []

    def test_content_survives_stopword_filtering(self) -> None:
        # "my app feels slow" → "my" and "feels" (well, "feel") may or
        # may not survive depending on the stopword list, but "app" and
        # "slow" should index.
        indices, _ = encode_bm25_sparse("my app feels slow")
        assert len(indices) >= 2
