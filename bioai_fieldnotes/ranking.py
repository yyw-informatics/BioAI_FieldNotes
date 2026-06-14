from __future__ import annotations

import math
from datetime import timezone

from .models import NormalizedPost, utc_now


def rank_posts(posts: list[NormalizedPost], trusted_handles: list[str] | None = None) -> list[NormalizedPost]:
    trusted = {handle.lower().lstrip("@") for handle in (trusted_handles or [])}
    engagement = [post.engagement for post in posts]
    velocity = [post.engagement / max(_age_hours(post), 1.0) for post in posts]
    followers = [math.log10(max(post.author_followers, 0) + 1) for post in posts]
    adjusted = [
        post.engagement / math.sqrt(max(post.author_followers, 0) + 1)
        for post in posts
    ]

    engagement_pct = _relative_log_signal(engagement)
    velocity_pct = _relative_log_signal(velocity)
    follower_pct = _relative_signal(followers)
    adjusted_pct = _relative_log_signal(adjusted)

    ranked: list[NormalizedPost] = []
    for idx, post in enumerate(posts):
        handle = post.author_handle.lower().lstrip("@")
        is_trusted = handle in trusted
        author_signal = max(follower_pct[idx], 0.90 if is_trusted else 0.0)
        domain_bonus = 0.12 if is_trusted else 0.0
        recency = max(0.0, 1.0 - (_age_hours(post) / (7 * 24)))
        score = min(
            1.0,
            0.35 * engagement_pct[idx]
            + 0.20 * velocity_pct[idx]
            + 0.25 * author_signal
            + 0.15 * adjusted_pct[idx]
            + 0.05 * recency
            + domain_bonus,
        )
        post.score = round(score, 4)
        post.score_parts = {
            "engagement_percentile": round(engagement_pct[idx], 4),
            "velocity_percentile": round(velocity_pct[idx], 4),
            "author_signal": round(author_signal, 4),
            "follower_adjusted_engagement": round(adjusted_pct[idx], 4),
            "recency": round(recency, 4),
        }
        ranked.append(post)
    return sorted(ranked, key=lambda p: p.score, reverse=True)


def _age_hours(post: NormalizedPost) -> float:
    created = post.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return max((utc_now() - created.astimezone(timezone.utc)).total_seconds() / 3600, 0.1)


def _relative_signal(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    if max_value <= 0:
        return [0.0 for _ in values]
    return [max(0.0, min(1.0, value / max_value)) for value in values]


def _relative_log_signal(values: list[float]) -> list[float]:
    logged = [math.log1p(max(0.0, value)) for value in values]
    return _relative_signal(logged)
