from app.ingestion.parser import normalize_markdown


class TestNormalizeMarkdown:
    def test_strips_yaml_front_matter(self) -> None:
        text = "---\ntitle: Hello\nauthor: Me\n---\n\n# Body\n\nContent"
        assert normalize_markdown(text) == "# Body\n\nContent"

    def test_ignores_mid_document_dash_lines(self) -> None:
        # A horizontal rule mid-doc is NOT front-matter and must be preserved.
        text = "# Title\n\nPrelude\n\n---\n\nAfter rule"
        assert normalize_markdown(text) == "# Title\n\nPrelude\n\n---\n\nAfter rule"

    def test_normalizes_crlf_line_endings(self) -> None:
        assert normalize_markdown("one\r\ntwo\r\nthree") == "one\ntwo\nthree"

    def test_normalizes_lone_cr_line_endings(self) -> None:
        assert normalize_markdown("one\rtwo\rthree") == "one\ntwo\nthree"

    def test_collapses_excessive_blank_lines(self) -> None:
        text = "para one\n\n\n\n\npara two"
        assert normalize_markdown(text) == "para one\n\npara two"

    def test_preserves_double_blank_lines(self) -> None:
        text = "para one\n\npara two"
        assert normalize_markdown(text) == "para one\n\npara two"

    def test_strips_leading_and_trailing_whitespace(self) -> None:
        assert normalize_markdown("\n\n  body  \n\n") == "body"

    def test_strips_front_matter_without_trailing_newline(self) -> None:
        # Document that ends immediately after the closing --- delimiter still
        # has its front-matter removed.
        assert normalize_markdown("---\ntitle: X\n---") == ""

    def test_is_idempotent(self) -> None:
        text = "---\nmeta\n---\n\n# Body\r\n\r\n\r\n\r\nmore\n\n\n"
        once = normalize_markdown(text)
        assert normalize_markdown(once) == once
