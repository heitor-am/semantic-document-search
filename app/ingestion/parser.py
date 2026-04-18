import re

_FRONT_MATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def normalize_markdown(text: str) -> str:
    """Clean up markdown content from arbitrary sources.

    - Strips a YAML front-matter block if present at the top
    - Normalizes CRLF / CR line endings to LF
    - Collapses runs of 3+ blank lines to 2
    - Trims leading/trailing whitespace

    Idempotent: normalize(normalize(x)) == normalize(x).
    """
    # 1. Normalize line endings first so front-matter stripping is reliable
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Strip YAML front-matter only when it's the very first thing in the doc
    text = _FRONT_MATTER_RE.sub("", text, count=1)

    # 3. Collapse excessive blank lines
    text = _MULTI_BLANK_RE.sub("\n\n", text)

    return text.strip()
