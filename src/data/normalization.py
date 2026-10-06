"""Text normalization.

Two deliberately different levels:

* `normalize_text`  - the *cleaning* normalization applied to the cleaned corpus.
  It is conservative: it keeps case (optional), punctuation, numbers, units and
  medical terminology, and only removes formatting noise.

* `matching_key`    - an *aggressive* canonical form used only for comparison
  (duplicate detection, leakage detection, ground-truth matching). It is never
  shown to the retriever, so it can safely drop case and punctuation.
"""

from __future__ import annotations

import html
import re
import unicodedata

# Invisible characters that frequently sneak in from web scraping / copy-paste.
_ZERO_WIDTH = re.compile(r"[​‌‍⁠﻿­]")
_HTML_TAG = re.compile(r"<\s*/?\s*[a-zA-Z][^<>]{0,200}>")
_BR_TAG = re.compile(r"<\s*(br|/p|/li|/div)\s*/?\s*>", re.IGNORECASE)
_MULTI_SPACE = re.compile(r"[ \t\f\v]+")
_MULTI_NEWLINE = re.compile(r"\n\s*\n+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")

_QUOTE_MAP = str.maketrans(
    {
        "‘": "'", "’": "'", "‚": "'", "‛": "'",
        "“": '"', "”": '"', "„": '"', "‟": '"',
        "–": "-", "—": "-", "−": "-",
        "…": "...",
    }
)

# Web boilerplate that carries no medical information. Kept intentionally
# narrow and anchored on unambiguous phrases so that medical text is never hit.
DEFAULT_BOILERPLATE_PATTERNS: tuple[str, ...] = (
    r"click here[^.]*\.?",
    r"subscribe to our newsletter[^.]*\.?",
    r"\badvertisement\b\.?",
    r"(?:©|\(c\)|copyright)\s*\d{4}.{0,80}?all rights reserved\.?",
    r"all rights reserved\.?",
    r"share this (?:article|page)(?: on [a-z ]+)?\.?",
    r"read more\s*(?:»|>>|\.\.\.)?",
    r"\[\s*(?:edit|citation needed)\s*\]",
)
# Literal spaces in the patterns match any whitespace run (e.g. "Read   more").
_BOILERPLATE = [re.compile(p.replace(" ", r"\s+"), re.IGNORECASE) for p in DEFAULT_BOILERPLATE_PATTERNS]


def is_missing(value: object) -> bool:
    """True for None / NaN / pandas NA / blank strings."""
    if value is None:
        return True
    try:
        if value != value:  # NaN check without importing numpy/pandas
            return True
    except Exception:  # pragma: no cover - exotic types
        pass
    return isinstance(value, str) and value.strip() == ""


def to_text(value: object) -> str:
    return "" if is_missing(value) else str(value)


def strip_html(text: str) -> str:
    text = _BR_TAG.sub("\n", text)
    text = _HTML_TAG.sub(" ", text)
    return html.unescape(text)


def remove_boilerplate(text: str) -> str:
    for pattern in _BOILERPLATE:
        text = pattern.sub(" ", text)
    return text


def normalize_unicode(text: str) -> str:
    # NFKC folds full-width letters, ligatures, non-breaking spaces, etc.
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = text.translate(_QUOTE_MAP)
    # Drop control characters except newline/tab.
    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_SPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_NEWLINE.sub("\n", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return text.strip()


def normalize_text(value: object, lowercase: bool = False) -> str:
    """Conservative cleaning normalization for retrieval documents.

    Order matters: HTML entities are decoded before Unicode folding, and
    boilerplate is stripped before whitespace is collapsed.
    """
    text = to_text(value)
    if not text:
        return ""
    text = html.unescape(text)  # decode entities first so "&lt;b&gt;" tags are seen
    text = strip_html(text)
    text = normalize_unicode(text)
    text = normalize_whitespace(text)  # before boilerplate, so multi-space variants still match
    text = remove_boilerplate(text)
    text = normalize_whitespace(text)
    # Strip stray leading/trailing separator characters left behind by removals.
    text = text.strip(" -|•·")
    if lowercase:
        text = text.lower()
    return text


def matching_key(value: object) -> str:
    """Aggressive canonical key: lowercase alphanumerics separated by single spaces.

    Two strings with the same key differ only in case, punctuation, whitespace,
    HTML markup, boilerplate or Unicode presentation.
    """
    text = normalize_text(value).lower()
    return _NON_ALNUM.sub(" ", text).strip()
