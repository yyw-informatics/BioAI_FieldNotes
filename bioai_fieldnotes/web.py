from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .models import ScanConfig
from .pipeline import run_scan
from .query_planner import topic_from_prompt
from .settings import load_dotenv, runtime_status
from .storage import Database, row_json
from .text import parse_csvish


PACKAGE_DIR = Path(__file__).resolve().parent

load_dotenv()

app = FastAPI(title="BioAI FieldNotes")
app.mount("/static", StaticFiles(directory=PACKAGE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=PACKAGE_DIR / "templates")


SENTIMENT_META = {
    "yay": {
        "emoji": "🎉",
        "label": "Positive",
        "brief": "Positive",
        "section": "Top Positive Stories",
        "tone": "is-yay",
    },
    "nooo": {
        "emoji": "🚨",
        "label": "Negative / friction",
        "brief": "Negative",
        "section": "Friction Stories",
        "tone": "is-nooo",
    },
    "mixed": {
        "emoji": "⚠️",
        "label": "Mixed / caution",
        "brief": "Mixed",
        "section": "Mixed Signals",
        "tone": "is-mixed",
    },
    "neutral": {
        "emoji": "⚪",
        "label": "Neutral / context",
        "brief": "Neutral",
        "section": "Context Notes",
        "tone": "is-neutral",
    },
}

PUBLIC_REPORT_META = {
    "experience": {
        "emoji": "",
        "label": "User Experience",
        "brief": "User Experience",
        "section": "User Experience",
        "tone": "is-yay",
    },
}

LOCKED_REPORT_STORY_IDS = {
    12: {
        "experience": [
            630, 635, 609, 610, 612, 613, 614, 615, 617, 619,
            608, 618, 629, 632, 664, 669, 677,
        ],
    },
}

LOCKED_REPORT_SUMMARIES = {
    12: {
        "heading": "What users experienced",
        "body": (
            "The selected posts show Claude Fable/Mythos being tested less as a chatbot and more as a "
            "hands-on build partner. Users highlight one-prompt prototypes, game design, landing pages, "
            "command centers, debugging workflows, shell speedups, 3D product sites, and a World Cup "
            "predictor. The strongest positive signal is speed: people describe moving from idea to "
            "working interface or workflow in minutes."
        ),
        "bullets": [
            "Creative and developer workflows dominate: visual apps, games, UI demos, code debugging, and lightweight automation.",
            "Scientific-adjacent friction remains important: biomedical prompts and a biology-economics thesis triggered safety filters.",
            "Reliability and cost are still weak spots: users reported quota burn, expensive review workflows, data-viewer errors, and limits on ambitious reasoning tasks.",
            "Mythos appears in higher-risk evaluation contexts, including sabotage-risk testing, while Fable appears most often in hands-on building stories.",
        ],
    },
}

REPORT_REMOVE_PATTERNS = [
    r"vibe-coded build demo",
    r"claude code team perspective",
    r"crypto workflow test with mytho",
    r"typing is slow",
    r"launch video highlights",
    r"issue triage and feature implementation",
    r"anthropic launched a",
    r"anthropic announces|released its latest ai models",
    r"last week is obsolete",
    r"fable launch context",
    r"claude code workflow|code team day to day",
    r"safety concerns around mythos access|general public.*trusted|mythos \+ restricted access",
]
REPORT_POSITIVE_PATTERNS = [
    r"macos app debugging|macos debugging|macos app right now|complex issue",
    r"6 months ago i one-shotted a game",
]
REPORT_FRICTION_PATTERNS = [
    r"theory-of-everything stress test",
    r"coco-to-vqa",
    r"buy extra credits|looks unfair|looked at the bill|haiku-level cost",
]
REPORT_TITLE_OVERRIDES = [
    (r"macos app debugging|macos app right now|complex issue", "Fable + macOS debugging"),
    (r"one-shotted a game|cooked this game", "Fable + game design"),
    (r"claude code workflow|code team day to day", "Fable + code workflow"),
    (r"e-commerce landing|ecom landing|fully branded ecom", "Mythos + landing pages"),
    (r"one-prompt build demo|built this from one prompt|not a mockup", "Fable + one-prompt prototypes"),
    (r"jarvis|command center", "Fable + command centers"),
    (r"backrooms escape game", "Fable + horror games"),
    (r"world cup predictor", "Fable + World Cup"),
    (r"prediction market.*world cup|best price.*world cup", "Fable + World Cup pricing"),
    (r"zsh shell startup|startup time", "Fable + shell speedups"),
    (r"3d product website|3d model", "Fable + 3D product sites"),
    (r"literally nothing about coding|nothing about coding|built this app in 1s", "Fable + no-code apps"),
    (r"fake camera stream|ios simulators|avcam", "Fable + iOS simulator testing"),
    (r"fable 5 for ecom|game-changer", "Fable + e-commerce testing"),
    (r"macos notch app|track it", "Fable + usage tracking"),
    (r"ui/ux concepts|endless inspiration", "Fable + UI concepts"),
    (r"ocean wave simulation|live physics sim", "Fable + physics simulation"),
    (r"creation tools|detailed prompt", "Fable + creation tools"),
    (r"dynamic login page|live time display", "Fable + animated login pages"),
    (r"starlink|satellite constellation", "Fable + satellite tracking"),
    (r"asteroid simulation", "Fable + asteroid simulation"),
    (r"one single prompt", "Fable + one-prompt prototypes"),
    (r"brotato|penguins", "Fable + game cloning"),
    (r"defensive security review|owasp", "Fable + security reviews"),
    (r"g1000|pixel game", "Fable + flight-sim UI"),
    (r"build the new @fable|in 1 prompt", "Fable + app cloning"),
    (r"solidity smart contract", "Fable + smart-contract review"),
    (r"terrain simulation|from dust", "Fable + terrain simulation"),
    (r"buy extra credits|looks unfair", "Fable + paid-credit concerns"),
    (r"looked at the bill|haiku-level cost", "Fable + cost concerns"),
    (r"security.*defensive rules|best-in-class at security", "Mythos + security testing"),
    (r"push directly to prod|test it in dev", "Fable + safer deployment"),
    (r"broken my workflow|best way possible", "Fable/Mythos + workflow shift"),
    (r"theory-of-everything|theory of everything|relativity", "Fable + physics limits"),
    (r"coco-to-vqa|coco json|data viewer", "Fable + data-viewer errors"),
    (r"biology security risk|flagged as a biology", "Fable + biosecurity filters"),
    (r"quota|weekly quota|usage-limit", "Fable + usage limits"),
    (r"controlarena|research sabotage|sabotage", "Mythos + higher sabotage risk"),
    (r"skill we built together|would have cost", "Fable + review costs"),
    (r"biology-economics thesis", "Fable + false safety flags"),
]


def db() -> Database:
    return Database()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    database = db()
    return templates.TemplateResponse(
        request,
        "index.html",
        {"runs": database.list_runs(limit=8), "row_json": row_json},
    )


@app.get("/scan", response_class=HTMLResponse)
def scan_form(request: Request):
    return templates.TemplateResponse(
        request,
        "scan.html",
        {"error": None, "runtime_status": runtime_status()},
    )


@app.post("/scan")
def start_scan(
    request: Request,
    prompt: str = Form(...),
    topic: str = Form(""),
    keywords: str = Form(""),
    exclusions: str = Form(""),
    platforms: list[str] = Form(["x"]),
    trusted_handles: str = Form(""),
    max_posts_per_source: int = Form(100),
    max_summaries: int = Form(15),
    max_estimated_cost_usd: float = Form(1.0),
    allow_fallback_pricing: str | None = Form(None),
):
    try:
        config = ScanConfig(
            topic=topic.strip() or topic_from_prompt(prompt),
            keywords=parse_csvish(keywords),
            prompt=prompt,
            exclusions=parse_csvish(exclusions),
            platforms=platforms,
            trusted_handles=parse_csvish(trusted_handles),
            days=7,
            max_posts_per_source=max_posts_per_source,
            max_summaries=max_summaries,
            max_estimated_cost_usd=max_estimated_cost_usd,
            allow_fallback_pricing=allow_fallback_pricing == "1",
        )
        run_id = run_scan(config, db())
    except Exception as exc:
        return templates.TemplateResponse(
            request,
            "scan.html",
            {"error": str(exc), "runtime_status": runtime_status()},
            status_code=400,
        )
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


@app.get("/runs", response_class=HTMLResponse)
def runs(request: Request):
    database = db()
    return templates.TemplateResponse(
        request,
        "runs.html",
        {"runs": database.list_runs(limit=50), "row_json": row_json},
    )


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail(request: Request, run_id: int):
    database = db()
    run = database.get_run(run_id)
    if not run:
        return templates.TemplateResponse(
            request,
            "message.html",
            {"title": "Run not found", "message": f"No run {run_id} exists."},
            status_code=404,
        )
    return templates.TemplateResponse(
        request,
        "run_detail.html",
        {
            "run": run,
            "clusters": database.list_clusters(run_id),
            "row_json": row_json,
        },
    )


@app.get("/reports/{run_id}", response_class=HTMLResponse)
def report(request: Request, run_id: int):
    database = db()
    run = database.get_run(run_id)
    if not run:
        return templates.TemplateResponse(
            request,
            "message.html",
            {"title": "Run not found", "message": f"No run {run_id} exists."},
            status_code=404,
        )
    return templates.TemplateResponse(
        request,
        "report.html",
        _report_context(request, database, run),
    )


@app.get("/clusters/{cluster_id}", response_class=HTMLResponse)
def cluster_detail(request: Request, cluster_id: int):
    database = db()
    cluster = database.get_cluster(cluster_id)
    if not cluster:
        return templates.TemplateResponse(
            request,
            "message.html",
            {"title": "Story not found", "message": f"No story {cluster_id} exists."},
            status_code=404,
        )
    return templates.TemplateResponse(
        request,
        "cluster_detail.html",
        {
            "cluster": cluster,
            "posts": database.cluster_posts(cluster_id),
            "metrics": row_json(cluster, "metrics_json", {}),
            "row_json": row_json,
        },
    )


@app.get("/costs", response_class=HTMLResponse)
def costs(request: Request):
    database = db()
    events = database.list_cost_events(limit=100)
    total_estimated = sum(float(event["estimated_cost_usd"] or 0) for event in events)
    return templates.TemplateResponse(
        request,
        "costs.html",
        {"events": events, "total_estimated": total_estimated},
    )


def _report_context(request: Request, database: Database, run) -> dict[str, Any]:
    clusters = database.list_clusters(run["id"])
    posts = database.run_posts(run["id"])
    cost_events = database.list_cost_events_for_run(run["id"])
    source_status = row_json(run, "source_status_json", {})
    query_plan = row_json(run, "query_plan_json", {})
    errors = row_json(run, "errors_json", [])
    story_cards = [_story_card(database, cluster) for cluster in clusters]
    total_estimated_cost = sum(float(event["estimated_cost_usd"] or 0.0) for event in cost_events)
    total_actual_cost = sum(float(event["actual_cost_usd"] or 0.0) for event in cost_events)
    total_input_tokens = sum(int(event["input_tokens"] or 0) for event in cost_events)
    total_output_tokens = sum(int(event["output_tokens"] or 0) for event in cost_events)
    models_used = sorted({
        f"{event['provider']} / {event['model']}"
        for event in cost_events
        if event["provider"] or event["model"]
    })
    posts_scanned = sum(
        int(status.get("fetched_count") or 0)
        for status in source_status.values()
        if isinstance(status, dict)
    )
    request_count = sum(
        int(status.get("request_count") or 0)
        for status in source_status.values()
        if isinstance(status, dict)
    )
    fallback_statuses = [
        status for status in source_status.values()
        if isinstance(status, dict) and status.get("fallback_triggered")
    ]
    sections = _public_report_sections(story_cards, run["id"])
    return {
        "request": request,
        "run": run,
        "report_title": _report_title(run, query_plan),
        "query_plan": query_plan,
        "errors": errors,
        "source_status": source_status,
        "report_summary": LOCKED_REPORT_SUMMARIES.get(int(run["id"])),
        "sections": sections,
        "posts_scanned": posts_scanned,
        "request_count": request_count,
        "unique_stories": len(clusters),
        "unique_authors": len({post["author_handle"].lower() for post in posts}),
        "fallback_triggered": bool(fallback_statuses),
        "fallback_reason": fallback_statuses[0].get("fallback_reason", "") if fallback_statuses else "",
        "scan_window": "Last 7 days",
        "platforms": ", ".join(row_json(run, "platforms_json", ["x"])),
        "platform_label": ", ".join(row_json(run, "platforms_json", ["x"])).upper(),
        "report_date": (run["completed_at"] or run["created_at"])[:19].replace("T", " "),
        "report_day": (run["completed_at"] or run["created_at"])[:10],
        "tool_name": "BioAI FieldNotes",
        "cost_event_count": len(cost_events),
        "total_estimated_cost": total_estimated_cost,
        "total_actual_cost": total_actual_cost,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "models_used": models_used,
    }


def _public_report_sections(story_cards: list[dict[str, Any]], run_id: int | None = None) -> list[dict[str, Any]]:
    locked = LOCKED_REPORT_STORY_IDS.get(int(run_id or 0))
    if locked:
        return _locked_report_sections(story_cards, locked)

    buckets: dict[str, list[dict[str, Any]]] = {"experience": []}
    for index, card in enumerate(story_cards):
        bucket = _public_report_bucket(card)
        if not bucket:
            continue
        curated = dict(card)
        curated["one_liner"] = _public_report_title(curated)
        curated["sentiment_meta"] = PUBLIC_REPORT_META[bucket]
        curated["_report_order"] = _public_report_priority(curated, bucket, index)
        buckets[bucket].append(curated)

    sections = []
    for bucket in ["experience"]:
        stories = sorted(buckets[bucket], key=lambda card: card["_report_order"])
        if stories:
            sections.append({
                "meta": PUBLIC_REPORT_META[bucket],
                "count": len(stories),
                "stories": stories,
            })
    return sections


def _locked_report_sections(
    story_cards: list[dict[str, Any]],
    locked: dict[str, list[int]],
) -> list[dict[str, Any]]:
    cards_by_id = {int(card["id"]): card for card in story_cards}
    sections = []
    for bucket in ["experience"]:
        stories = []
        for card_id in locked.get(bucket, []):
            card = cards_by_id.get(card_id)
            if not card:
                continue
            curated = dict(card)
            curated["one_liner"] = _public_report_title(curated)
            curated["sentiment_meta"] = PUBLIC_REPORT_META[bucket]
            stories.append(curated)
        if stories:
            sections.append({
                "meta": PUBLIC_REPORT_META[bucket],
                "count": len(stories),
                "stories": stories,
            })
    return sections


def _public_report_bucket(card: dict[str, Any]) -> str:
    text = _public_report_match_text(card)
    if _matches_any(text, REPORT_REMOVE_PATTERNS):
        return ""
    if _matches_any(text, REPORT_POSITIVE_PATTERNS + REPORT_FRICTION_PATTERNS):
        return "experience"
    if card["sentiment"] in {"yay", "nooo", "mixed"}:
        return "experience"
    return ""


def _public_report_title(card: dict[str, Any]) -> str:
    text = _public_report_match_text(card)
    for pattern, title in REPORT_TITLE_OVERRIDES:
        if re.search(pattern, text):
            return title
    return card["one_liner"]


def _public_report_priority(card: dict[str, Any], bucket: str, index: int) -> tuple[int, int]:
    text = _public_report_match_text(card)
    if _matches_any(text, REPORT_POSITIVE_PATTERNS + REPORT_FRICTION_PATTERNS):
        return (0, index)
    return (1, index)


def _public_report_match_text(card: dict[str, Any]) -> str:
    return " ".join(
        str(card.get(key, ""))
        for key in ["one_liner", "excerpt", "source_url", "author"]
    ).lower()


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _story_card(database: Database, cluster) -> dict[str, Any]:
    metrics = row_json(cluster, "metrics_json", {})
    posts = database.cluster_posts(cluster["id"])
    top_post = posts[0] if posts else None
    sentiment = str(metrics.get("sentiment_label") or "neutral")
    if sentiment not in SENTIMENT_META:
        sentiment = "neutral"
    post_count = int(metrics.get("post_count") or len(posts))
    author_count = int(metrics.get("unique_author_count") or post_count)
    story_fit = float(metrics.get("max_story_fit") or 0.0)
    domain_signal = float(metrics.get("domain_context_signal") or 0.0)
    media_signal = float(metrics.get("media_signal") or 0.0)
    engagement = float(metrics.get("total_engagement") or 0.0)
    excerpt = _clean_text(top_post["text"] if top_post else cluster["title"])
    return {
        "id": cluster["id"],
        "one_liner": _story_title(cluster, excerpt),
        "href": f"/clusters/{cluster['id']}",
        "source_url": top_post["url"] if top_post else "",
        "author": top_post["author_handle"] if top_post else "",
        "excerpt": _truncate_words(excerpt, 34),
        "sentiment": sentiment,
        "sentiment_meta": SENTIMENT_META[sentiment],
        "score": float(cluster["score"] or 0.0),
        "story_fit": story_fit,
        "post_count": post_count,
        "author_count": author_count,
        "engagement": engagement,
        "engagement_label": _compact_number(engagement),
        "visual_evidence": bool(metrics.get("visual_evidence")),
        "media_signal": media_signal,
        "tag": "Bio/public health match" if domain_signal >= 0.3 else "General user story",
        "summary_status": cluster["summary_status"],
        "bullets": _story_bullets(cluster, top_post, metrics, sentiment, excerpt),
    }


def _report_title(run, query_plan: dict[str, Any]) -> str:
    targets = [
        str(term).strip()
        for term in query_plan.get("target_terms", [])
        if str(term).strip()
    ]
    target_text = " ".join(targets).lower()
    if "mythos" in target_text and "fable" in target_text:
        return "Claude Mythos, Fable, User Stories"
    compact: list[str] = []
    lowered_targets = [term.lower() for term in targets]
    for term in targets:
        lower = term.lower()
        if any(other != lower and lower in other for other in lowered_targets):
            continue
        if lower not in {item.lower() for item in compact}:
            compact.append(term)
    if compact:
        return f"Field Brief: {' + '.join(compact[:3])} User Stories"
    return f"Field Brief: {run['topic']}"


def _story_bullets(cluster, top_post, metrics: dict[str, Any], sentiment: str, excerpt: str) -> list[str]:
    post_count = int(metrics.get("post_count") or 1)
    author_count = int(metrics.get("unique_author_count") or post_count)
    engagement = float(metrics.get("total_engagement") or 0.0)
    story_fit = float(metrics.get("max_story_fit") or 0.0)
    mood = SENTIMENT_META.get(sentiment, SENTIMENT_META["neutral"])["label"].lower()
    summary_source = _clean_text(cluster["summary"]) if cluster["summary"] else excerpt
    summary = _truncate_words(summary_source, 26)
    bullets = [
        f"Summary: {summary}",
        f"Signal: {post_count} merged version(s) from {author_count} author(s); engagement {_compact_number(engagement)}.",
        f"Mood: {mood}; story fit {story_fit * 100:.0f}%.",
    ]
    if top_post:
        bullets.append(f"Representative: @{top_post['author_handle']}.")
    return bullets


def _story_title(cluster, excerpt: str) -> str:
    summary = _clean_text(cluster["summary"]) if cluster["summary"] else ""
    title_text = _clean_text(cluster["title"])
    raw_source = summary or excerpt or title_text
    source = raw_source
    source = re.sub(r"^(Title|Summary|Why it matters|Evidence|Caveats):\s*", "", source)
    source = _strip_redundant_story_prefix(source)
    combined_source = f"{summary} {title_text} {excerpt} {source}"
    lower = combined_source.lower()
    model = _story_model_label(combined_source)
    title_rules = [
        (r"changed how we work|code team day to day", "Claude Code workflow changed by Fable"),
        (r"biology security risk|biosecurity|flagged.*security risk", "Biology prompt flagged as security risk"),
        (r"flagged as", "Prompt flagged as a safety risk"),
        (r"controlarena|research sabotage|sabotage", "ControlArena sabotage test"),
        (r"too dangerous|general public", "Safety concerns around Mythos access"),
        (r"weekly quota|usage limit|quota", "Fable quota and usage-limit friction"),
        (r"zsh shell startup|startup time", "Shell startup optimized with Fable"),
        (r"backrooms escape game|escape game", "Backrooms game built with Fable"),
        (r"coco json|vqa", "COCO-to-VQA converter experiment"),
        (r"macos app", "macOS app debugging with Fable"),
        (r"ecom landing page|e-commerce landing|landing page", "E-commerce landing page from one prompt"),
        (r"world cup predictor", "World Cup predictor built with Fable"),
        (r"email and calendar", "Email and calendar automation test"),
        (r"crypto live|wallet|buyback|tokens", "Crypto workflow test with Mythos"),
        (r"launch video", "Launch video highlights model behavior"),
        (r"out today|first mythos-class", "Fable launch context"),
        (r"theory of everything|relativity", "Theory-of-Everything stress test"),
        (r"working copy|code review", "Working-copy code review with Fable"),
        (r"issue and feature|feature we can implement|issues and features", "Issue triage and feature implementation"),
        (r"engineer on.*code team|code team", "Claude Code team perspective"),
        (r"jarvis|command center", "Jarvis-style command center built with Fable"),
        (r"replaced my entire|completely replaced", "Workflow replacement story"),
        (r"briefing site", "Fable briefing site overview"),
        (r"nothing about coding|zero.*coding", "No-code app build with Fable"),
        (r"vibe coding|look what i built", "Vibe-coded build demo"),
        (r"one prompt", "One-prompt build demo"),
    ]
    for pattern, title in title_rules:
        if re.search(pattern, lower):
            return title
    task = _extract_story_task(source)
    if task:
        return f"{task} with {model}" if model else task
    return _truncate_words(source, 8)


def _story_model_label(text: str) -> str:
    lower = text.lower()
    has_fable = "fable" in lower
    has_mythos = "mythos" in lower
    if has_fable and has_mythos:
        return "Fable/Mythos"
    if has_fable:
        return "Fable"
    if has_mythos:
        return "Mythos"
    return "Claude"


def _extract_story_task(text: str) -> str:
    patterns = [
        r"\b(?:i|we)\s+(?:used|tried|asked)\b.*?\bto\s+(?P<task>[^.!?;:]{8,110})",
        r"\b(?:using|with)\b.*?\bto\s+(?P<task>[^.!?;:]{8,110})",
        r"\b(?:i|we)\s+built\s+(?P<task>[^.!?;:]{8,90})",
        r"\bbuilt\s+(?P<task>[^.!?;:]{8,90})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            task = _clean_story_task(match.group("task"))
            if task:
                return task
    return ""


def _clean_story_task(text: str) -> str:
    cleaned = _clean_text(text)
    cleaned = re.sub(r"\b(?:claude|fable|mythos|fable-5|max|model|new)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:this|that|the|a|an)\s+", "", cleaned, count=1, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:with|using|from|for)\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;,.()[]{}")
    if not cleaned:
        return ""
    cleaned = _truncate_words(cleaned, 8)
    return cleaned[:1].upper() + cleaned[1:]


def _strip_redundant_story_prefix(text: str) -> str:
    cleaned = re.sub(r"\bClaude\s+(Fable|Mythos)\s*\d*\b", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bClaude'?s?\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;,.")
    return cleaned or text


def _clean_text(value: str | None) -> str:
    text = re.sub(r"https?://\S+", "", value or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _truncate_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]).rstrip(".,;:") + "..."


def _compact_number(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}m".replace(".0m", "m")
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}k".replace(".0k", "k")
    return str(int(value))
