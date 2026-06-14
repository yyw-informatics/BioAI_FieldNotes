from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .models import StoryCluster, SummaryResult
from .pricing import estimate_tokens, load_price_catalog
from .settings import env


class Summarizer:
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        allow_fallback_pricing: bool = False,
    ) -> None:
        self.provider = provider or env("BIOAI_LLM_PROVIDER", "openai")
        self.model = model or env("BIOAI_LLM_MODEL", "gpt-5-mini")
        self.allow_fallback_pricing = allow_fallback_pricing

    def summarize(self, cluster: StoryCluster, max_cost_usd: float) -> SummaryResult:
        if self.provider == "none":
            return SummaryResult(status="skipped_provider_none")
        prompt = self._prompt(cluster)
        input_tokens = estimate_tokens(prompt)
        expected_output_tokens = 350
        catalog = load_price_catalog(self.allow_fallback_pricing)
        price = catalog.get(self.provider, self.model)
        if not price:
            return SummaryResult(
                status="blocked_pricing",
                provider=self.provider,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=expected_output_tokens,
                error="No model price found. Set BIOAI_MODELPRICES_PATH or enable fallback pricing.",
            )
        estimated_cost = price.estimate(input_tokens, expected_output_tokens)
        if estimated_cost > max_cost_usd:
            return SummaryResult(
                status="blocked_budget",
                provider=self.provider,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=expected_output_tokens,
                estimated_cost_usd=estimated_cost,
                error="Estimated summary cost exceeds remaining scan budget.",
            )
        if self.provider != "openai":
            return SummaryResult(
                status="skipped_unsupported_provider",
                provider=self.provider,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=expected_output_tokens,
                estimated_cost_usd=estimated_cost,
                error="Only the OpenAI Responses API adapter is implemented in M1.",
            )
        api_key = env("OPENAI_API_KEY")
        if not api_key:
            return SummaryResult(
                status="skipped_no_key",
                provider=self.provider,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=expected_output_tokens,
                estimated_cost_usd=estimated_cost,
                summary=_extractive_summary(cluster),
                caveats="Set OPENAI_API_KEY to replace this extractive preview with an LLM summary.",
            )
        return self._openai_response(prompt, api_key, estimated_cost)

    def _prompt(self, cluster: StoryCluster) -> str:
        lines = [
            "Write a concise BioAI field note from these social posts.",
            "Return exactly these sections: Title, Why it matters, Evidence, Caveats.",
            "Prioritize concrete anecdotes: who used the model, what they used it for, and what happened.",
            "If the posts are only announcements or generic mentions, say that clearly in Caveats.",
            "Be careful with uncertainty and do not overclaim scientific validity.",
            "",
        ]
        for idx, post in enumerate(cluster.posts[:8], start=1):
            lines.append(
                f"Post {idx} ({post.platform}, @{post.author_handle}, "
                f"likes={post.like_count}, reposts={post.repost_count}, "
                f"replies={post.reply_count}, quotes={post.quote_count}): {post.text}"
            )
            if post.extracted_urls:
                lines.append(f"Links: {', '.join(post.extracted_urls[:3])}")
        return "\n".join(lines)

    def _openai_response(self, prompt: str, api_key: str, estimated_cost: float) -> SummaryResult:
        payload = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": 450,
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
            with urllib.request.urlopen(req, timeout=45) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return SummaryResult(
                status="error",
                provider=self.provider,
                model=self.model,
                estimated_cost_usd=estimated_cost,
                error=f"OpenAI API HTTP {exc.code}: {exc.reason}",
            )
        except Exception as exc:
            return SummaryResult(
                status="error",
                provider=self.provider,
                model=self.model,
                estimated_cost_usd=estimated_cost,
                error=f"OpenAI API error: {exc}",
            )
        usage = data.get("usage") or {}
        text = data.get("output_text") or _collect_output_text(data)
        return SummaryResult(
            status="complete",
            summary=text.strip(),
            provider=self.provider,
            model=self.model,
            input_tokens=int(usage.get("input_tokens") or estimate_tokens(prompt)),
            output_tokens=int(usage.get("output_tokens") or estimate_tokens(text)),
            estimated_cost_usd=estimated_cost,
            actual_cost_usd=estimated_cost,
        )


def _collect_output_text(data: dict) -> str:
    chunks: list[str] = []
    for item in data.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                chunks.append(content["text"])
    return "\n".join(chunks)


def _extractive_summary(cluster: StoryCluster) -> str:
    top = cluster.posts[0] if cluster.posts else None
    if not top:
        return "No representative posts were available."
    return (
        f"Title: {cluster.title}\n\n"
        f"Why it matters: This cluster is rising within the configured 7-day BioAI scan "
        f"based on relative engagement and author signal.\n\n"
        f"Evidence: The leading post is from @{top.author_handle} with "
        f"{top.like_count} likes, {top.repost_count} reposts, {top.reply_count} replies, "
        f"and {top.quote_count} quotes. The cluster contains {len(cluster.posts)} post(s).\n\n"
        "Caveats: This is an extractive preview, not an LLM synthesis."
    )
