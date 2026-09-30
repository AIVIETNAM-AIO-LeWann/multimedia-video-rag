"""Text normalization and safe SQLite FTS5 query construction."""

from __future__ import annotations

import re
import unicodedata

WORD_RE = re.compile(r"\w+", re.UNICODE)

# Very common English words that would match most captions and only slow BM25 down.
ENGLISH_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "there",
        "this",
        "to",
        "was",
        "were",
        "with",
    ]
)


def normalize_search(text: str) -> str:
    """Match the ingest normalization: NFC, casefold, collapse whitespace."""
    return " ".join(unicodedata.normalize("NFC", str(text)).casefold().split())


def remove_accents(text: str) -> str:
    """Match the ASR ``normalized_no_accent`` column (``đ`` becomes ``d``)."""
    value = normalize_search(text).replace("đ", "d")
    return "".join(
        char for char in unicodedata.normalize("NFD", value) if unicodedata.category(char) != "Mn"
    )


def tokens(text: str, *, stopwords: frozenset[str] = frozenset()) -> list[str]:
    """Word tokens in order, deduplicated, without stopwords."""
    seen: dict[str, None] = {}
    for token in WORD_RE.findall(normalize_search(text)):
        if token not in stopwords:
            seen.setdefault(token, None)
    return list(seen)


def fts_or_query(words: list[str]) -> str | None:
    """OR of quoted terms: BM25 then ranks rows by how many/which terms they contain.

    Quoting every term neutralizes FTS5 syntax (AND, NOT, *, :, parentheses) in user input.
    """
    if not words:
        return None
    return " OR ".join('"' + word.replace('"', '""') + '"' for word in words)
