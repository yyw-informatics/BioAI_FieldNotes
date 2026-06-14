from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


SUPPORTED_PLATFORMS = {"x"}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ScanConfig:
    topic: str
    keywords: list[str]
    prompt: str = ""
    exclusions: list[str] = field(default_factory=list)
    target_terms: list[str] = field(default_factory=list)
    context_terms: list[str] = field(default_factory=list)
    usage_cues: list[str] = field(default_factory=list)
    negative_cues: list[str] = field(default_factory=list)
    platforms: list[str] = field(default_factory=lambda: ["x"])
    trusted_handles: list[str] = field(default_factory=list)
    days: int = 7
    max_posts_per_source: int = 100
    max_summaries: int = 15
    max_estimated_cost_usd: float = 1.0
    allow_fallback_pricing: bool = False

    def __post_init__(self) -> None:
        if self.days != 7:
            raise ValueError("M1 supports only 7-day scans.")
        if not self.topic.strip():
            raise ValueError("Topic is required.")
        if not self.keywords and not self.prompt.strip():
            raise ValueError("Provide a research request or at least one keyword.")
        unknown = set(self.platforms) - SUPPORTED_PLATFORMS
        if unknown:
            raise ValueError(f"Unsupported platform(s): {', '.join(sorted(unknown))}")
        if self.max_posts_per_source < 10:
            raise ValueError("max_posts_per_source must be at least 10.")
        if self.max_summaries < 0:
            raise ValueError("max_summaries cannot be negative.")
        if self.max_estimated_cost_usd < 0:
            raise ValueError("max_estimated_cost_usd cannot be negative.")


@dataclass
class NormalizedPost:
    platform: str
    platform_post_id: str
    url: str
    text: str
    author_handle: str
    author_display_name: str = ""
    author_followers: int = 0
    created_at: datetime = field(default_factory=utc_now)
    like_count: int = 0
    repost_count: int = 0
    reply_count: int = 0
    quote_count: int = 0
    extracted_urls: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    score_parts: dict[str, float] = field(default_factory=dict)

    @property
    def engagement(self) -> float:
        return (
            self.like_count
            + 2.0 * self.repost_count
            + 1.5 * self.quote_count
            + self.reply_count
        )


@dataclass
class SourceResult:
    platform: str
    posts: list[NormalizedPost] = field(default_factory=list)
    error: str | None = None
    fetched_count: int = 0
    raw_request_count: int = 0
    query_mode: str = "initial"
    query: str = ""


@dataclass
class StoryCluster:
    title: str
    posts: list[NormalizedPost]
    score: float
    key: str
    metrics: dict[str, Any] = field(default_factory=dict)
    summary_status: str = "pending"
    summary: str = ""
    caveats: str = ""


@dataclass
class SummaryResult:
    status: str
    summary: str = ""
    caveats: str = ""
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float | None = None
    error: str | None = None


@dataclass
class QueryPlan:
    topic: str
    keywords: list[str]
    exclusions: list[str] = field(default_factory=list)
    trusted_handles: list[str] = field(default_factory=list)
    target_terms: list[str] = field(default_factory=list)
    context_terms: list[str] = field(default_factory=list)
    usage_cues: list[str] = field(default_factory=list)
    negative_cues: list[str] = field(default_factory=list)
    story_goal: str = ""
    notes: str = ""
    status: str = "heuristic"


@dataclass
class QueryPlanResult:
    plan: QueryPlan
    usage: SummaryResult
