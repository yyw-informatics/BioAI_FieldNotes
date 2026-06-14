from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from urllib.parse import urlsplit

from .models import NormalizedPost, StoryCluster
from .story_selection import cluster_story_metrics
from .text import canonicalize_url, short_title, tokenize


def cluster_posts(posts: list[NormalizedPost]) -> list[StoryCluster]:
    groups = _merge_groups_by_story_text(_group_by_shared_urls(posts))
    clusters: list[StoryCluster] = []
    for key, group_posts in groups:
        group_posts = sorted(group_posts, key=lambda p: p.score, reverse=True)
        score = _cluster_score(group_posts)
        metrics = _cluster_metrics(group_posts)
        clusters.append(
            StoryCluster(
                title=short_title(group_posts[0].text),
                posts=group_posts,
                score=round(score, 4),
                key=key,
                metrics=metrics,
            )
        )
    return sorted(clusters, key=lambda cluster: cluster.score, reverse=True)


def _group_by_shared_urls(posts: list[NormalizedPost]) -> list[tuple[str, list[NormalizedPost]]]:
    if not posts:
        return []
    parent = list(range(len(posts)))
    url_owner: dict[str, int] = {}

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for idx, post in enumerate(posts):
        for url in _post_urls(post):
            if url in url_owner:
                union(idx, url_owner[url])
            else:
                url_owner[url] = idx

    grouped: dict[int, list[NormalizedPost]] = defaultdict(list)
    for idx, post in enumerate(posts):
        grouped[find(idx)].append(post)

    output: list[tuple[str, list[NormalizedPost]]] = []
    for idx, group_posts in grouped.items():
        urls = sorted({url for post in group_posts for url in _post_urls(post)})
        if urls:
            key_source = " ".join(urls)
            key = f"url:{hashlib.sha1(key_source.encode('utf-8')).hexdigest()[:12]}"
        else:
            key = _text_group_key(group_posts)
        output.append((key, group_posts))
    return output


def _merge_groups_by_story_text(
    groups: list[tuple[str, list[NormalizedPost]]],
) -> list[tuple[str, list[NormalizedPost]]]:
    merged: list[tuple[str, set[str], list[NormalizedPost]]] = []
    for key, group_posts in sorted(
        groups,
        key=lambda item: max((post.score for post in item[1]), default=0.0),
        reverse=True,
    ):
        tokens = _group_story_tokens(group_posts)
        matched = False
        for idx, (existing_key, existing_tokens, existing_posts) in enumerate(merged):
            if _same_story(tokens, existing_tokens):
                existing_posts.extend(group_posts)
                merged[idx] = (existing_key, existing_tokens, existing_posts)
                matched = True
                break
        if not matched:
            merged.append((key, set(tokens), list(group_posts)))
    return [(key, posts) for key, _, posts in merged]


def _cluster_by_text(posts: list[NormalizedPost]) -> list[tuple[str, list[NormalizedPost]]]:
    groups: list[tuple[set[str], list[NormalizedPost]]] = []
    for post in posts:
        tokens = tokenize(post.text)
        matched = False
        for group_tokens, group_posts in groups:
            if _jaccard(tokens, group_tokens) >= 0.68:
                group_posts.append(post)
                group_tokens.update(tokens)
                matched = True
                break
        if not matched:
            groups.append((set(tokens), [post]))
    output = []
    for tokens, group_posts in groups:
        digest = hashlib.sha1(" ".join(sorted(tokens)).encode("utf-8")).hexdigest()[:12]
        output.append((f"text:{digest}", group_posts))
    return output


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _same_story(a: set[str], b: set[str]) -> bool:
    if not a or not b:
        return False
    intersection = len(a & b)
    containment = intersection / min(len(a), len(b))
    if min(len(a), len(b)) <= 6:
        return intersection >= 3 and containment >= 0.75
    if intersection < 5:
        return False
    return _jaccard(a, b) >= 0.62 or containment >= 0.78


def _cluster_score(posts: list[NormalizedPost]) -> float:
    top_scores = [post.score for post in posts[:3]]
    top = max(top_scores) if top_scores else 0.0
    avg_top = sum(top_scores) / len(top_scores) if top_scores else 0.0
    size_boost = min(0.12, math.log1p(len(posts)) / 20)
    return min(1.0, 0.70 * top + 0.30 * avg_top + size_boost)


def _cluster_metrics(posts: list[NormalizedPost]) -> dict:
    platform_mix = Counter(post.platform for post in posts)
    total_engagement = sum(post.engagement for post in posts)
    author_followers = [post.author_followers for post in posts]
    metrics = {
        "post_count": len(posts),
        "platform_mix": dict(platform_mix),
        "total_engagement": round(total_engagement, 2),
        "max_author_followers": max(author_followers or [0]),
        "representative_url": posts[0].url if posts else "",
        "unique_author_count": len({post.author_handle.lower() for post in posts}),
    }
    metrics.update(cluster_story_metrics(posts))
    return metrics


def _post_urls(post: NormalizedPost) -> list[str]:
    urls = [
        canonicalize_url(url)
        for url in post.extracted_urls
        if url and _is_clusterable_url(canonicalize_url(url))
    ]
    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        unique.append(url)
    return unique


def _is_clusterable_url(url: str) -> bool:
    parsed = urlsplit(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if host in {"x.com", "twitter.com", "www.x.com", "www.twitter.com"}:
        return "/video/" in path or "/photo/" in path
    return True


def _group_story_tokens(posts: list[NormalizedPost]) -> set[str]:
    tokens: set[str] = set()
    for post in posts[:5]:
        tokens.update(_story_tokens(post.text))
    return tokens


def _story_tokens(text: str) -> set[str]:
    cleaned = URL_INLINE_RE.sub(" ", text or "")
    cleaned = HANDLE_RE.sub(" ", cleaned)
    generic = {
        "ai", "amp", "anthropic", "claude", "fable", "fable5", "mythos",
        "believe", "can", "cant", "model", "models", "new", "one",
        "prompt", "prompts", "the", "using", "used", "use", "uses",
        "built", "build", "tried", "try", "video", "watch",
    }
    return {token for token in tokenize(cleaned) if token not in generic}


def _text_group_key(posts: list[NormalizedPost]) -> str:
    tokens = _group_story_tokens(posts)
    digest = hashlib.sha1(" ".join(sorted(tokens)).encode("utf-8")).hexdigest()[:12]
    return f"text:{digest}"


URL_INLINE_RE = re.compile(r"https?://\S+", re.IGNORECASE)
HANDLE_RE = re.compile(r"@\w+")
