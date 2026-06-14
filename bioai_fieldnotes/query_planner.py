from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import replace

from .models import QueryPlan, QueryPlanResult, ScanConfig, SummaryResult
from .pricing import estimate_tokens, load_price_catalog
from .settings import env
from .text import parse_csvish


STOP_WORDS = {
    "about", "across", "after", "anecdotes", "and", "around", "because",
    "find", "for", "from", "interesting", "into", "latest", "models",
    "most", "people", "prioritize", "recent", "related", "stories", "that",
    "the", "their", "this", "using", "want", "with",
}

DEFAULT_USAGE_CUES = [
    "I used",
    "we used",
    "using",
    "used",
    "built",
    "tried",
    "tested",
    "deployed",
    "workflow",
    "case study",
]

DEFAULT_NEGATIVE_CUES = [
    "announcement",
    "launch",
    "released",
    "benchmark",
    "leaderboard",
    "pricing",
    "job",
    "hiring",
    "course",
    "webinar",
]


def plan_scan_query(config: ScanConfig) -> QueryPlanResult:
    prompt = config.prompt.strip()
    if not prompt:
        plan = QueryPlan(
            topic=config.topic,
            keywords=config.keywords,
            exclusions=config.exclusions,
            trusted_handles=config.trusted_handles,
            target_terms=config.target_terms or config.keywords,
            context_terms=config.context_terms,
            usage_cues=config.usage_cues,
            negative_cues=config.negative_cues,
            story_goal="Manual keyword mode.",
            notes="Manual keyword mode.",
            status="manual",
        )
        return QueryPlanResult(plan=plan, usage=SummaryResult(status="manual_query"))

    llm_result = _llm_plan(prompt, config)
    if llm_result:
        return llm_result
    return _heuristic_plan(prompt, config, status="heuristic_fallback")


def apply_query_plan(config: ScanConfig, plan: QueryPlan) -> ScanConfig:
    return replace(
        config,
        topic=plan.topic or config.topic,
        keywords=_dedupe(plan.keywords + config.keywords)[:10],
        exclusions=_dedupe(plan.exclusions + config.exclusions)[:12],
        target_terms=_compact_target_terms(plan.target_terms + config.target_terms)[:8],
        context_terms=_dedupe(plan.context_terms + config.context_terms)[:10],
        usage_cues=_dedupe(plan.usage_cues + config.usage_cues)[:10],
        negative_cues=_dedupe(plan.negative_cues + config.negative_cues)[:12],
        trusted_handles=_dedupe(plan.trusted_handles + config.trusted_handles),
    )


def topic_from_prompt(prompt: str) -> str:
    cleaned = re.sub(r"\s+", " ", prompt).strip()
    if not cleaned:
        return "7-day scan"
    return cleaned[:82] + ("..." if len(cleaned) > 82 else "")


def _llm_plan(prompt: str, config: ScanConfig) -> QueryPlanResult | None:
    api_key = env("OPENAI_API_KEY")
    if not api_key:
        return None
    provider = "openai"
    model = env("BIOAI_QUERY_PLANNER_MODEL", env("BIOAI_LLM_MODEL", "gpt-5-mini") or "gpt-5-mini") or "gpt-5-mini"
    planner_prompt = _planner_prompt(prompt, config)
    input_tokens = estimate_tokens(planner_prompt)
    expected_output_tokens = 300
    price = load_price_catalog(allow_fallback=True).get(provider, model)
    estimated_cost = price.estimate(input_tokens, expected_output_tokens) if price else 0.0
    if estimated_cost > 0.05:
        return None
    payload = {
        "model": model,
        "input": planner_prompt,
        "max_output_tokens": 500,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=35) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return None
    text = data.get("output_text") or _collect_output_text(data)
    try:
        parsed = _parse_json_text(text)
    except ValueError:
        return None
    plan = QueryPlan(
        topic=str(parsed.get("topic") or config.topic or topic_from_prompt(prompt)),
        keywords=_coerce_list(parsed.get("keywords"))[:10],
        exclusions=_coerce_list(parsed.get("exclusions"))[:12],
        trusted_handles=_clean_handles(_coerce_list(parsed.get("trusted_handles"))),
        target_terms=_coerce_list(parsed.get("target_terms"))[:8],
        context_terms=_coerce_list(parsed.get("context_terms"))[:10],
        usage_cues=_coerce_list(parsed.get("usage_cues"))[:10],
        negative_cues=_coerce_list(parsed.get("negative_cues"))[:12],
        story_goal=str(parsed.get("story_goal") or ""),
        notes=str(parsed.get("notes") or ""),
        status="llm",
    )
    if not plan.target_terms:
        plan.target_terms = _target_terms(prompt, plan.keywords)
    plan.target_terms = _compact_target_terms(plan.target_terms)
    if not plan.context_terms:
        plan.context_terms = _context_terms(prompt)
    if not plan.usage_cues and _wants_usage_stories(prompt):
        plan.usage_cues = DEFAULT_USAGE_CUES[:]
    if not plan.negative_cues:
        plan.negative_cues = DEFAULT_NEGATIVE_CUES[:]
    if not plan.keywords:
        return None
    usage = data.get("usage") or {}
    output_tokens = int(usage.get("output_tokens") or estimate_tokens(text))
    input_tokens = int(usage.get("input_tokens") or input_tokens)
    actual_cost = price.estimate(input_tokens, output_tokens) if price else estimated_cost
    return QueryPlanResult(
        plan=plan,
        usage=SummaryResult(
            status="query_plan_complete",
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost,
            actual_cost_usd=actual_cost,
        ),
    )


def _heuristic_plan(prompt: str, config: ScanConfig, status: str) -> QueryPlanResult:
    phrases = _quoted_phrases(prompt)
    title_terms = _capitalized_terms(prompt)
    context_terms = _context_terms(prompt)
    target_terms = _target_terms(prompt, phrases + _model_combinations(title_terms) + title_terms)
    usage_cues = DEFAULT_USAGE_CUES[:] if _wants_usage_stories(prompt) else []
    words = [
        word
        for word in re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", prompt)
        if word.lower() not in STOP_WORDS
    ]
    keywords = _dedupe(
        target_terms
        + phrases
        + context_terms
        + words
        + config.keywords
    )[:10]
    plan = QueryPlan(
        topic=config.topic or topic_from_prompt(prompt),
        keywords=keywords,
        exclusions=config.exclusions,
        target_terms=_compact_target_terms(target_terms or keywords[:6]),
        context_terms=context_terms,
        usage_cues=usage_cues,
        negative_cues=DEFAULT_NEGATIVE_CUES[:],
        trusted_handles=config.trusted_handles,
        story_goal=_story_goal(prompt),
        notes="Heuristic query plan because the LLM planner was unavailable or skipped.",
        status=status,
    )
    return QueryPlanResult(plan=plan, usage=SummaryResult(status=status))


def _planner_prompt(prompt: str, config: ScanConfig) -> str:
    seed_keywords = ", ".join(config.keywords) or "(none)"
    seed_exclusions = ", ".join(config.exclusions) or "(none)"
    return f"""
You convert a user's natural-language social-search request into a compact query plan for X recent search.

Return strict JSON only:
{{
  "topic": "short dashboard label",
  "keywords": ["6-10 exact phrases or terms for display and fallback search"],
  "target_terms": ["2-8 names of the model/product/lab/project that must anchor retrieval"],
  "context_terms": ["0-10 domain terms to prioritize but not necessarily require"],
  "usage_cues": ["0-10 words or short phrases that indicate someone actually used or applied the target"],
  "negative_cues": ["0-12 words or short phrases that indicate generic announcements, benchmarks, jobs, courses, or meta-discussion"],
  "exclusions": ["0-8 terms to filter out"],
  "trusted_handles": ["0-8 handles without @ if obvious from the request"],
  "story_goal": "one sentence describing the kind of story to select",
  "notes": "one short explanation"
}}

Guidelines:
- Preserve named models, products, labs, diseases, fields, and exact phrases.
- Separate target model/product terms from domain/context terms.
- If the user asks for stories, anecdotes, or how people used something, usage_cues should include concrete use-case language such as "I used", "we used", "built", "tried", "tested", "workflow", or "case study".
- Context terms should help rank the result later; do not use broad context words as standalone retrieval anchors.
- Prefer phrases that will work in X search, not academic database syntax.
- For biomedical/science/public-health requests, put those domain terms in context_terms unless they are part of the target name.
- Do not invent handles unless the user clearly names them.
- Keep keywords concise; avoid generic words alone unless they are essential proper names.

User request:
{prompt}

Manual seed keywords: {seed_keywords}
Manual seed exclusions: {seed_exclusions}
""".strip()


def _collect_output_text(data: dict) -> str:
    chunks: list[str] = []
    for item in data.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                chunks.append(content["text"])
    return "\n".join(chunks)


def _parse_json_text(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(parsed, dict):
        raise ValueError("query plan must be a JSON object")
    return parsed


def _coerce_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return parse_csvish(value)
    return []


def _clean_handles(handles: list[str]) -> list[str]:
    return [handle.strip().lstrip("@") for handle in handles if handle.strip()]


def _quoted_phrases(prompt: str) -> list[str]:
    matches = re.findall(r'"([^"]+)"|' + r"'([^']+)'", prompt)
    return [left or right for left, right in matches]


def _capitalized_terms(prompt: str) -> list[str]:
    raw = re.findall(r"\b[A-Z][A-Za-z0-9+-]{2,}\b", prompt)
    return _dedupe([term for term in raw if term.lower() not in STOP_WORDS])


def _model_combinations(terms: list[str]) -> list[str]:
    combos: list[str] = []
    if "Claude" in terms:
        for term in terms:
            if term != "Claude":
                combos.append(f"Claude {term}")
    return combos


def _target_terms(prompt: str, candidates: list[str]) -> list[str]:
    terms = _dedupe(candidates)
    lower_prompt = prompt.lower()
    model_markers = {"claude", "gpt", "gemini", "llama", "mistral", "mythos", "fable"}
    selected = [
        term
        for term in terms
        if term.lower() in lower_prompt
        and (
            term.lower() in model_markers
            or any(marker in term.lower() for marker in model_markers)
            or term[:1].isupper()
        )
    ]
    if "Claude" in selected:
        for term in terms:
            if (
                term != "Claude"
                and term[:1].isupper()
                and "claude" not in term.lower()
            ):
                selected.insert(0, f"Claude {term}")
    return _compact_target_terms(selected)[:8]


def _compact_target_terms(terms: list[str]) -> list[str]:
    deduped = _dedupe(terms)
    suffixes = {
        term.split(" ", 1)[1].lower()
        for term in deduped
        if " " in term and term.lower().startswith("claude ")
    }
    output = [
        term
        for term in deduped
        if term.lower() not in suffixes
    ]
    return output


def _context_terms(prompt: str) -> list[str]:
    lower = prompt.lower()
    candidates = [
        "biomedical research",
        "public health",
        "single cell",
        "bioinformatics",
        "biomedicine",
        "science",
        "research",
        "biology",
        "medicine",
        "clinical",
        "epidemiology",
    ]
    return [term for term in candidates if term in lower]


def _wants_usage_stories(prompt: str) -> bool:
    lower = prompt.lower()
    return any(
        cue in lower
        for cue in [
            "anecdote",
            "story",
            "stories",
            "used",
            "using",
            "how people use",
            "how people have used",
            "use case",
            "use-case",
        ]
    )


def _story_goal(prompt: str) -> str:
    if _wants_usage_stories(prompt):
        return (
            "Select concrete anecdotes or use cases where people report applying "
            "the target model to a real task, workflow, experiment, or decision."
        )
    return "Select posts that best match the user's research request."


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output
