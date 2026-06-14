from __future__ import annotations

import re
import urllib.parse

URL_RE = re.compile(r"https?://[^\s)>\]}\"']+", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]{1,}")


def parse_csvish(value: str | None) -> list[str]:
    if not value:
        return []
    parts = re.split(r"[\n,]+", value)
    return [part.strip() for part in parts if part.strip()]


def extract_urls(text: str) -> list[str]:
    return [canonicalize_url(match.group(0)) for match in URL_RE.finditer(text or "")]


def canonicalize_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url.rstrip(".,;:"))
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=False)
    kept = [
        (k, v)
        for k, v in query
        if not k.lower().startswith("utm_")
        and k.lower() not in {"fbclid", "gclid", "mc_cid", "mc_eid"}
    ]
    return urllib.parse.urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/") or "/",
            urllib.parse.urlencode(kept),
            "",
        )
    )


def tokenize(text: str) -> set[str]:
    stop = {
        "about", "after", "again", "also", "because", "been", "being", "from",
        "have", "here", "into", "just", "more", "over", "such", "than", "that",
        "their", "then", "there", "these", "they", "this", "with", "would",
    }
    return {
        token.lower()
        for token in TOKEN_RE.findall(text or "")
        if token.lower() not in stop and len(token) > 2
    }


def short_title(text: str, max_words: int = 12) -> str:
    words = re.sub(r"\s+", " ", text or "").strip().split(" ")
    title = " ".join(words[:max_words]).strip()
    if len(words) > max_words:
        title += "..."
    return title or "Untitled story"
