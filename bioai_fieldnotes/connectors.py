from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import timedelta, timezone
from typing import Any

from .models import NormalizedPost, ScanConfig, SourceResult, utc_now
from .settings import env
from .text import extract_urls


class HttpJsonClient:
    def get_json(self, url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
        request_headers = {"User-Agent": "BioAIFieldNotes/0.1"}
        request_headers.update(headers or {})
        req = urllib.request.Request(url, headers=request_headers)
        with urllib.request.urlopen(req, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))


def _iso_z(dt) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


class XConnector:
    endpoint = "https://api.x.com/2/tweets/search/recent"
    story_fallback_cues = [
        "I used",
        "we used",
        "my workflow",
        "our workflow",
        "I built",
        "we built",
        "I tried",
        "we tried",
        "I tested",
        "we tested",
    ]

    def __init__(self, token: str | None = None, client: HttpJsonClient | None = None) -> None:
        self.token = token or env("X_BEARER_TOKEN")
        self.client = client or HttpJsonClient()

    def fetch(self, config: ScanConfig, query_mode: str = "initial") -> SourceResult:
        if not self.token:
            return SourceResult(platform="x", error="X_BEARER_TOKEN is not set.", query_mode=query_mode)

        query = self._build_query(config, query_mode=query_mode)
        max_results = max(10, min(100, config.max_posts_per_source))
        params = {
            "query": query,
            "start_time": _iso_z(utc_now() - timedelta(days=6, hours=23, minutes=50)),
            "max_results": str(max_results),
            "sort_order": "relevancy",
            "tweet.fields": "attachments,author_id,created_at,entities,public_metrics,referenced_tweets,lang",
            "user.fields": "created_at,description,id,name,public_metrics,username,verified,verified_type",
            "expansions": "author_id,attachments.media_keys",
            "media.fields": "alt_text,duration_ms,media_key,preview_image_url,public_metrics,type,url,width,height",
        }
        url = f"{self.endpoint}?{urllib.parse.urlencode(params)}"
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            payload = self.client.get_json(url, headers=headers)
        except urllib.error.HTTPError as exc:
            return SourceResult(platform="x", error=_http_error("X API", exc), query_mode=query_mode, query=query)
        except Exception as exc:
            return SourceResult(platform="x", error=f"X API error: {exc}", query_mode=query_mode, query=query)

        users = {
            user.get("id"): user
            for user in payload.get("includes", {}).get("users", [])
            if isinstance(user, dict)
        }
        media_by_key = {
            media.get("media_key"): media
            for media in payload.get("includes", {}).get("media", [])
            if isinstance(media, dict)
        }
        posts: list[NormalizedPost] = []
        for item in payload.get("data", []) or []:
            author = users.get(item.get("author_id"), {})
            metrics = item.get("public_metrics") or {}
            handle = author.get("username") or item.get("author_id") or "unknown"
            text = item.get("text") or ""
            post_id = str(item.get("id", ""))
            created_at = item.get("created_at")
            media = _x_media(item, media_by_key)
            raw = dict(item)
            raw["media"] = media
            posts.append(
                NormalizedPost(
                    platform="x",
                    platform_post_id=post_id,
                    url=f"https://x.com/{handle}/status/{post_id}" if post_id else "",
                    text=text,
                    author_handle=handle,
                    author_display_name=author.get("name") or handle,
                    author_followers=_int((author.get("public_metrics") or {}).get("followers_count")),
                    created_at=_parse_datetime(created_at),
                    like_count=_int(metrics.get("like_count")),
                    repost_count=_int(metrics.get("retweet_count")),
                    reply_count=_int(metrics.get("reply_count")),
                    quote_count=_int(metrics.get("quote_count")),
                    extracted_urls=_x_urls(item, text, media),
                    raw=raw,
                )
            )
        return SourceResult(
            platform="x",
            posts=posts,
            fetched_count=len(posts),
            raw_request_count=1,
            query_mode=query_mode,
            query=query,
        )

    def _build_query(self, config: ScanConfig, query_mode: str = "initial") -> str:
        target_terms = config.target_terms or config.keywords
        target_group = _or_group(target_terms[:8])
        usage_terms = config.usage_cues[:8]
        if query_mode == "story_fallback":
            usage_terms = self.story_fallback_cues[:8]
        usage_group = _or_group(usage_terms)
        context_group = ""
        if query_mode == "initial":
            context_group = _or_group(_specific_context_terms(config.context_terms)[:6])
        if target_group and usage_group:
            base = f"{target_group} {usage_group}"
        else:
            base = target_group or _or_group(config.keywords[:10])
        if context_group:
            base = f"{base} {context_group}"
        exclusions = " ".join(f"-{term.strip()}" for term in config.exclusions if term.strip())
        return f"{base} lang:en -is:retweet {exclusions}".strip()


def _or_group(values: list[str]) -> str:
    terms = [_query_term(value) for value in values if value.strip()]
    terms = [term for term in terms if term]
    if not terms:
        return ""
    if len(terms) == 1:
        return terms[0]
    return f"({' OR '.join(terms)})"


def _query_term(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return ""
    if _needs_quotes(cleaned) and not (cleaned.startswith('"') and cleaned.endswith('"')):
        return f'"{cleaned}"'
    return cleaned


def _specific_context_terms(terms: list[str]) -> list[str]:
    specific = [
        term
        for term in terms
        if " " in term.strip()
        and term.strip().lower() not in {"deep research", "scientific research"}
    ]
    return specific or terms


def _x_media(item: dict[str, Any], media_by_key: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ((item.get("attachments") or {}).get("media_keys") or [])
    media: list[dict[str, Any]] = []
    for key in keys:
        found = media_by_key.get(key)
        if found:
            media.append(found)
    return media


def _x_urls(item: dict[str, Any], text: str, media: list[dict[str, Any]] | None = None) -> list[str]:
    urls = []
    for url in ((item.get("entities") or {}).get("urls") or []):
        expanded = url.get("expanded_url") or url.get("url")
        if expanded:
            urls.append(expanded)
    urls.extend(extract_urls(text))
    for item_media in media or []:
        for key in ("url", "preview_image_url"):
            media_url = item_media.get(key)
            if media_url:
                urls.append(media_url)
    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def _parse_datetime(value: str | None):
    if not value:
        return utc_now()
    cleaned = value.replace("Z", "+00:00")
    try:
        from datetime import datetime

        return datetime.fromisoformat(cleaned)
    except ValueError:
        return utc_now()


def _needs_quotes(term: str) -> bool:
    return any(not (char.isalnum() or char == "_") for char in term)


def _http_error(label: str, exc: urllib.error.HTTPError) -> str:
    body = ""
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""
    if body:
        body = body[:500].replace("\n", " ")
        return f"{label} HTTP {exc.code}: {exc.reason} - {body}"
    return f"{label} HTTP {exc.code}: {exc.reason}"
