from __future__ import annotations

from app.retrieval.sparse_encoder import encode_bm25_sparse, tokenize


class TestTokenize:
    def test_lowercases_and_splits(self) -> None:
        assert tokenize("Hello WORLD  foo") == ["hello", "world", "foo"]

    def test_drops_punctuation(self) -> None:
        assert tokenize("don't, stop!") == ["don", "t", "stop"]

    def test_keeps_digit_letter_mixes(self) -> None:
        assert tokenize("python3.12 and bge-m3") == ["python3", "12", "and", "bge", "m3"]

    def test_empty_string(self) -> None:
        assert tokenize("") == []


class TestEncodeBm25Sparse:
    def test_single_occurrence_counts_as_one(self) -> None:
        indices, values = encode_bm25_sparse("python async programming")
        assert len(indices) == 3
        assert len(values) == 3
        assert all(v == 1.0 for v in values)

    def test_repeated_tokens_aggregate(self) -> None:
        indices, values = encode_bm25_sparse("python python python async")
        # Two distinct tokens: python (x3) and async (x1).
        assert len(indices) == 2
        assert sorted(values) == [1.0, 3.0]

    def test_empty_text_produces_empty_vectors(self) -> None:
        assert encode_bm25_sparse("") == ([], [])

    def test_punctuation_only_produces_empty_vectors(self) -> None:
        assert encode_bm25_sparse("!!! ??? --- ...") == ([], [])

    def test_deterministic_across_calls(self) -> None:
        # Stable hashes + sorted iteration → identical output every run.
        a = encode_bm25_sparse("python async programming python")
        b = encode_bm25_sparse("python async programming python")
        assert a == b

    def test_deterministic_across_processes(self) -> None:
        # Indirectly verified: the hash function is MD5, not the randomised
        # built-in hash(). This test documents the invariant by checking
        # that indices for a known token are in the expected 32-bit range
        # and that the mapping is stable for a known input.
        indices, _ = encode_bm25_sparse("python")
        assert len(indices) == 1
        # 32-bit range
        assert 0 <= indices[0] < 2**32
        # MD5("python") first 4 bytes big-endian — hard-coded so a change
        # in the hash function (which would break on-disk points) is loud.
        import hashlib

        expected = int.from_bytes(hashlib.md5(b"python", usedforsecurity=False).digest()[:4], "big")
        assert indices[0] == expected

    def test_indices_and_values_same_length(self) -> None:
        indices, values = encode_bm25_sparse("a b c a b a")
        assert len(indices) == len(values)

    def test_token_order_does_not_affect_output(self) -> None:
        # Different input orderings should produce the same (set of
        # (index, value) pairs) — we sort internally so the arrays
        # themselves are identical too.
        a = encode_bm25_sparse("python async programming")
        b = encode_bm25_sparse("programming async python")
        assert a == b
