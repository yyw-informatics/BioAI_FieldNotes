from __future__ import annotations

import re
from collections import Counter

from .models import NormalizedPost, QueryPlan


USE_VERBS = [
    "use",
    "used",
    "using",
    "try",
    "tried",
    "test",
    "tested",
    "build",
    "built",
    "deploy",
    "deployed",
    "integrate",
    "integrated",
    "apply",
    "applied",
    "prompt",
    "prompted",
    "ask",
    "asked",
    "analyze",
    "analyzed",
    "summarize",
    "summarized",
    "debug",
    "debugged",
    "write",
    "wrote",
]

OUTCOME_TERMS = [
    "workflow",
    "pipeline",
    "analysis",
    "dataset",
    "paper",
    "manuscript",
    "grant",
    "figure",
    "code",
    "notebook",
    "protocol",
    "experiment",
    "screen",
    "saved",
    "helped",
    "found",
    "automated",
    "improved",
    "reduced",
    "accelerated",
    "generated",
]

DOMAIN_TERMS = [
    "biomedical",
    "public health",
    "bioinformatics",
    "biology",
    "medicine",
    "clinical",
    "patient",
    "patients",
    "epidemiology",
    "genomics",
    "single cell",
    "single-cell",
    "omics",
    "lab",
    "research",
    "science",
]

DEFAULT_NEGATIVE_CUES = [
    "announcement",
    "announced",
    "announces",
    "launch",
    "launched",
    "launches",
    "releasing",
    "released",
    "release notes",
    "benchmark",
    "leaderboard",
    "pricing",
    "subscription",
    "job",
    "hiring",
    "course",
    "webinar",
    "podcast",
    "newsletter",
    "rumor",
]

POSITIVE_SENTIMENT_TERMS = [
    "amazing",
    "awesome",
    "best",
    "changed how we work",
    "can't believe i built",
    "cute",
    "dangerously cute",
    "excited",
    "faster",
    "good",
    "great",
    "helped",
    "high quality",
    "impressive",
    "improved",
    "incredible",
    "insane",
    "love",
    "loved",
    "one prompt",
    "one-shot",
    "recommend",
    "saved",
    "smart",
    "smooth",
    "success",
    "useful",
    "works",
    "worked",
    "yay",
]

NEGATIVE_SENTIMENT_TERMS = [
    "bad",
    "banned",
    "blocked",
    "burned",
    "burnt",
    "can't be used",
    "can't respond",
    "can't solve",
    "can't use",
    "cannot",
    "chill bro",
    "couldn't",
    "dangerous",
    "disappoint",
    "error",
    "expensive",
    "fail",
    "failed",
    "flagged",
    "frustrating",
    "harmful",
    "issue",
    "limit",
    "nooo",
    "out of messages",
    "problem",
    "quota",
    "risk",
    "sad",
    "scary",
    "slow",
    "terrified",
    "terrifying",
    "unsafe",
    "worse",
]

MEDIA_DEMO_TERMS = [
    "built this",
    "built a",
    "built an",
    "made this",
    "created this",
    "one prompt",
    "in 3 hours",
    "in three hours",
    "watch",
    "demo",
    "video",
    "game",
    "app",
    "website",
    "landing page",
    "prototype",
    "dashboard",
    "interface",
    "command center",
    "showcase",
    "look what i built",
    "look what we built",
    "vibe coding",
    "vibe-coded",
]

MEDIA_ALT_TERMS = [
    "screen recording",
    "screenshot",
    "gameplay",
    "playable",
    "application",
    "web app",
    "user interface",
    "demo",
    "prototype",
    "dashboard",
    "visualization",
]

PERSONAL_USE_RE = re.compile(
    r"\b(i|we|team|lab|clinic|group|students|colleagues)\b"
    r"\s+(?:have\s+|had\s+|just\s+|also\s+|actually\s+|finally\s+|recently\s+|am\s+|are\s+|was\s+|were\s+|started\s+)?("
    + "|".join(re.escape(verb) for verb in USE_VERBS)
    + r")\b",
    re.IGNORECASE | re.DOTALL,
)
OWNED_WORKFLOW_RE = re.compile(
    r"\b(my|our)\b.{0,50}\b(workflow|project|pipeline|analysis|paper|manuscript|code|notebook|product|lab|clinic)\b",
    re.IGNORECASE | re.DOTALL,
)
THIRD_PARTY_USE_RE = re.compile(
    r"\b(developer|researcher|scientist|physician|clinician|student|founder|team|lab|company|user|users|people)\b"
    r".{0,80}\b(using|used|built|tested|tried|deployed|applied)\b",
    re.IGNORECASE | re.DOTALL,
)
TASK_USE_RE = re.compile(
    r"\b("
    + "|".join(re.escape(verb) for verb in USE_VERBS)
    + r")\b.{0,80}\b(for|to|on|with|in)\b",
    re.IGNORECASE | re.DOTALL,
)
PASSIVE_NONSTORY_RE = re.compile(
    r"\b(can'?t|cannot|can not|couldn'?t|shouldn'?t|won'?t|not)\s+be\s+used\b",
    re.IGNORECASE,
)
VISUAL_BUILD_RE = re.compile(
    r"\b(?:built|made|created|generated|shipped|coded|vibe[-\s]?coded)\b"
    r".{0,90}\b(?:game|app|site|website|landing page|demo|prototype|tool|dashboard|interface|ui|video|command center)\b"
    r"|"
    r"\b(?:game|app|site|website|landing page|demo|prototype|tool|dashboard|interface|ui|video|command center)\b"
    r".{0,90}\b(?:built|made|created|generated|shipped|coded)\b",
    re.IGNORECASE | re.DOTALL,
)
EXPLICIT_NEGATIVE_SENTIMENT_RE = re.compile(
    r"\b(?:bad|banned|blocked|burned|burnt|cannot|couldn'?t|dangerous|disappoint(?:ed|ing)?|"
    r"error|expensive|fail(?:ed)?|flagged|frustrating|harmful|issue|limit|nooo|"
    r"out of messages|problem|quota|risk|sad|scary|slow|terrified|terrifying|unsafe|worse)\b"
    r"|can'?t\s+(?:be used|respond|solve|use)",
    re.IGNORECASE,
)


def rerank_for_story_fit(
    posts: list[NormalizedPost],
    plan: QueryPlan,
    domain_weight: float = 0.16,
) -> list[NormalizedPost]:
    for post in posts:
        story_fit, parts = score_post_story_fit(post, plan, domain_weight=domain_weight)
        base_score = post.score
        post.score = round(min(1.0, 0.45 * base_score + 0.55 * story_fit), 4)
        post.score_parts.update(parts)
    return sorted(posts, key=lambda p: p.score, reverse=True)


def score_post_story_fit(
    post: NormalizedPost,
    plan: QueryPlan,
    domain_weight: float = 0.16,
) -> tuple[float, dict]:
    text = post.text or ""
    lower = text.lower()
    target_terms = plan.target_terms or plan.keywords
    context_terms = plan.context_terms or DOMAIN_TERMS
    usage_cues = plan.usage_cues or USE_VERBS
    negative_cues = plan.negative_cues or DEFAULT_NEGATIVE_CUES

    target_signal = _term_signal(lower, target_terms)
    first_person_signal = 1.0 if re.search(r"\b(i|we|my|our)\b", lower) else 0.0
    media_parts = _media_parts(post, lower)
    media_signal = float(media_parts["media_signal"])
    visual_demo_signal = float(media_parts["visual_demo_signal"])
    usage_signal = max(
        min(_term_signal(lower, usage_cues), 0.45),
        1.0 if PERSONAL_USE_RE.search(text) else 0.0,
        0.85 if OWNED_WORKFLOW_RE.search(text) else 0.0,
        0.85 if THIRD_PARTY_USE_RE.search(text) else 0.0,
        0.55 if TASK_USE_RE.search(text) else 0.0,
    )
    domain_signal = _term_signal(lower, context_terms)
    outcome_signal = _term_signal(lower, OUTCOME_TERMS)
    negative_signal = _term_signal(lower, negative_cues)
    if visual_demo_signal >= 0.75:
        usage_signal = max(usage_signal, 0.65 if first_person_signal else 0.55)
        outcome_signal = max(outcome_signal, 0.70)
    elif media_signal >= 0.45:
        outcome_signal = max(outcome_signal, 0.45)
    if PASSIVE_NONSTORY_RE.search(text):
        usage_signal = min(usage_signal, 0.2)
        negative_signal = max(negative_signal, 0.75)

    raw = (
        0.30 * target_signal
        + 0.34 * usage_signal
        + domain_weight * domain_signal
        + 0.12 * outcome_signal
        + 0.08 * first_person_signal
        + 0.08 * media_signal
        - 0.22 * negative_signal
    )
    if usage_signal < 0.25:
        raw *= 0.45
    if target_signal < 0.25:
        raw *= 0.65
    story_fit = _clamp(raw)
    label = _selection_label(story_fit, usage_signal, negative_signal)
    sentiment = _sentiment_parts(text, media_parts)
    return (
        round(story_fit, 4),
        {
            "story_fit": round(story_fit, 4),
            "target_signal": round(target_signal, 4),
            "usage_signal": round(usage_signal, 4),
            "domain_context_signal": round(domain_signal, 4),
            "domain_weight": round(domain_weight, 4),
            "outcome_signal": round(outcome_signal, 4),
            "negative_meta_signal": round(negative_signal, 4),
            "selection_label": label,
            **media_parts,
            **sentiment,
        },
    )


def needs_story_fallback(clusters, max_summaries: int) -> bool:
    desired = max(3, min(max_summaries or 0, 15) // 3)
    good = 0
    for cluster in clusters[: max(max_summaries, 10)]:
        metrics = getattr(cluster, "metrics", {}) or {}
        story_fit = float(metrics.get("max_story_fit", 0.0) or 0.0)
        usage_signal = float(metrics.get("usage_signal", 0.0) or 0.0)
        label = str(metrics.get("selection_label", ""))
        if (
            label in {"likely_use_story", "possible_use_case"}
            or (story_fit >= 0.52 and usage_signal >= 0.55)
        ):
            good += 1
    return good < desired


def cluster_story_metrics(posts: list[NormalizedPost]) -> dict:
    story_fits = [_float_part(post, "story_fit") for post in posts]
    usage = [_float_part(post, "usage_signal") for post in posts]
    domain = [_float_part(post, "domain_context_signal") for post in posts]
    media_signals = [_float_part(post, "media_signal") for post in posts]
    media_counts = [_float_part(post, "media_count") for post in posts]
    labels = Counter(str(post.score_parts.get("selection_label", "unknown")) for post in posts)
    sentiments = Counter(str(post.score_parts.get("sentiment_label", "neutral")) for post in posts)
    sentiment_scores = [_float_part(post, "sentiment_score") for post in posts]
    sentiment_negative_hits = sum(_float_part(post, "sentiment_negative_hits") for post in posts)
    media_types = sorted({
        media_type
        for post in posts
        for media_type in str(post.score_parts.get("media_types", "")).split(",")
        if media_type
    })
    top_label = labels.most_common(1)[0][0] if labels else "unknown"
    top_sentiment = sentiments.most_common(1)[0][0] if sentiments else "neutral"
    visual_evidence = max(media_signals or [0.0]) >= 0.75
    if visual_evidence and top_sentiment == "neutral" and sentiment_negative_hits == 0:
        top_sentiment = "yay"
    elif visual_evidence and top_sentiment == "mixed" and sentiment_negative_hits == 0:
        top_sentiment = "yay"
    return {
        "max_story_fit": round(max(story_fits or [0.0]), 4),
        "avg_story_fit": round(sum(story_fits) / len(story_fits), 4) if story_fits else 0.0,
        "usage_signal": round(max(usage or [0.0]), 4),
        "domain_context_signal": round(max(domain or [0.0]), 4),
        "media_signal": round(max(media_signals or [0.0]), 4),
        "media_count": int(sum(media_counts)),
        "media_types": media_types,
        "visual_evidence": visual_evidence,
        "selection_label": top_label,
        "sentiment_label": top_sentiment,
        "sentiment_score": round(sum(sentiment_scores) / len(sentiment_scores), 4) if sentiment_scores else 0.0,
        "sentiment_mix": dict(sentiments),
    }


def _media_parts(post: NormalizedPost, lower_text: str) -> dict:
    media = _post_media(post)
    media_count = len(media)
    media_types = sorted({
        str(item.get("type", "")).strip().lower()
        for item in media
        if str(item.get("type", "")).strip()
    })
    has_visual = any(media_type in {"photo", "video", "animated_gif"} for media_type in media_types)
    alt_text = " ".join(str(item.get("alt_text") or "") for item in media).lower()
    demo_text_signal = max(
        _term_signal(lower_text, MEDIA_DEMO_TERMS),
        1.0 if VISUAL_BUILD_RE.search(lower_text) else 0.0,
    )
    alt_signal = max(
        _term_signal(alt_text, MEDIA_DEMO_TERMS + MEDIA_ALT_TERMS + OUTCOME_TERMS),
        1.0 if VISUAL_BUILD_RE.search(alt_text) else 0.0,
    )

    media_signal = 0.0
    if has_visual:
        media_signal = 0.45
    if has_visual and re.search(r"\b(this|watch|video|demo|screenshot|clip)\b", lower_text):
        media_signal = max(media_signal, 0.65)
    if has_visual and demo_text_signal:
        media_signal = max(media_signal, 0.85)
    if has_visual and alt_signal:
        media_signal = max(media_signal, min(1.0, 0.60 + 0.35 * alt_signal))

    return {
        "media_signal": round(_clamp(media_signal), 4),
        "media_count": media_count,
        "media_types": ",".join(media_types),
        "media_alt_signal": round(alt_signal, 4),
        "visual_demo_signal": round(_clamp(media_signal if demo_text_signal or alt_signal else 0.0), 4),
    }


def _post_media(post: NormalizedPost) -> list[dict]:
    raw = post.raw or {}
    media = raw.get("media") or raw.get("includes_media") or []
    if not isinstance(media, list):
        return []
    return [item for item in media if isinstance(item, dict)]


def _term_signal(text: str, terms: list[str]) -> float:
    cleaned_terms = [term.strip().lower() for term in terms if term.strip()]
    if not cleaned_terms:
        return 0.0
    matches = 0
    for term in cleaned_terms:
        if " " in term:
            matches += 1 if term in text else 0
        else:
            matches += 1 if re.search(rf"\b{re.escape(term)}\b", text) else 0
    return min(1.0, matches / min(3, max(1, len(cleaned_terms))))


def _selection_label(story_fit: float, usage_signal: float, negative_signal: float) -> str:
    if negative_signal >= 0.45 and usage_signal < 0.7:
        return "announcement_or_meta"
    if story_fit >= 0.72 and usage_signal >= 0.75:
        return "likely_use_story"
    if story_fit >= 0.52 and usage_signal >= 0.65 and negative_signal < 0.45:
        return "possible_use_case"
    if story_fit >= 0.32:
        return "weak_story_signal"
    return "keyword_mention"


def _sentiment_parts(text: str, media_parts: dict | None = None) -> dict:
    normalized = _normalize_sentiment_text(text)
    positive_hits = _count_terms(normalized, POSITIVE_SENTIMENT_TERMS)
    negative_hits = _count_terms(normalized, NEGATIVE_SENTIMENT_TERMS)
    visual_positive = (
        1
        if media_parts
        and float(media_parts.get("visual_demo_signal", 0.0) or 0.0) >= 0.75
        and not EXPLICIT_NEGATIVE_SENTIMENT_RE.search(normalized)
        else 0
    )
    positive_hits += visual_positive
    if positive_hits == 0 and negative_hits == 0:
        score = 0.0
        label = "neutral"
    else:
        score = (positive_hits - negative_hits) / max(positive_hits + negative_hits, 1)
        if positive_hits and negative_hits and abs(score) < 0.45:
            label = "mixed"
        elif score >= 0.2:
            label = "yay"
        elif score <= -0.2:
            label = "nooo"
        else:
            label = "neutral"
    return {
        "sentiment_label": label,
        "sentiment_score": round(score, 4),
        "sentiment_positive_hits": positive_hits,
        "sentiment_negative_hits": negative_hits,
        "sentiment_visual_positive": visual_positive,
    }


def _normalize_sentiment_text(text: str) -> str:
    return (
        (text or "")
        .lower()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )


def _count_terms(text: str, terms: list[str]) -> int:
    hits = 0
    for term in terms:
        cleaned = term.lower()
        if " " in cleaned or "'" in cleaned:
            hits += text.count(cleaned)
        else:
            hits += len(re.findall(rf"\b{re.escape(cleaned)}\b", text))
    return hits


def _float_part(post: NormalizedPost, key: str) -> float:
    try:
        return float(post.score_parts.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
