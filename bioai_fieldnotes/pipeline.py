from __future__ import annotations

from .clustering import cluster_posts
from .connectors import XConnector
from .models import NormalizedPost, QueryPlan, ScanConfig, SourceResult
from .query_planner import apply_query_plan, plan_scan_query
from .ranking import rank_posts
from .storage import Database
from .story_selection import needs_story_fallback, rerank_for_story_fit
from .summarizer import Summarizer


def run_scan(config: ScanConfig, db: Database | None = None) -> int:
    db = db or Database()
    run_id = db.create_run(config)
    try:
        query_plan_result = plan_scan_query(config)
        config = apply_query_plan(config, query_plan_result.plan)
        db.update_run_query_plan(run_id, query_plan_result.plan)
        if query_plan_result.usage.provider or query_plan_result.usage.status.startswith("query_plan"):
            db.insert_cost_event(run_id, None, query_plan_result.usage)

        source_results: list[SourceResult] = []
        connector = XConnector()
        if "x" in config.platforms:
            source_results.append(connector.fetch(config, query_mode="initial"))

        posts = _dedupe_posts([post for result in source_results for post in result.posts])
        ranked, clusters = _rank_and_cluster(posts, config, query_plan_result.plan)
        fallback_reason = ""
        if (
            "x" in config.platforms
            and _can_fetch_fallback(source_results)
            and needs_story_fallback(clusters, config.max_summaries)
        ):
            fallback_reason = "First pass did not find enough likely user-experience stories; widened to model-focused user stories."
            source_results.append(connector.fetch(config, query_mode="story_fallback"))
            posts = _dedupe_posts([post for result in source_results for post in result.posts])
            ranked, clusters = _rank_and_cluster(
                posts,
                config,
                query_plan_result.plan,
                domain_weight=0.06,
            )

        errors = [
            f"{result.platform}: {result.error}"
            for result in source_results
            if result.error
        ]
        source_status = _source_status(source_results, fallback_reason)
        post_db_ids = db.insert_posts(run_id, ranked)
        post_identity_to_db_id = {id(post): post_db_ids[idx] for idx, post in enumerate(ranked)}

        summarizer = Summarizer(allow_fallback_pricing=config.allow_fallback_pricing)
        remaining_budget = config.max_estimated_cost_usd
        for idx, cluster in enumerate(clusters):
            cluster.summary_status = "not_selected"
            if idx < config.max_summaries:
                result = summarizer.summarize(cluster, remaining_budget)
                cluster.summary_status = result.status
                cluster.summary = result.summary
                cluster.caveats = result.caveats or result.error or ""
                remaining_budget = max(0.0, remaining_budget - result.estimated_cost_usd)
            post_ids = [post_identity_to_db_id[id(post)] for post in cluster.posts if id(post) in post_identity_to_db_id]
            cluster_id = db.insert_cluster(run_id, cluster, post_ids)
            if idx < config.max_summaries:
                db.insert_cost_event(run_id, cluster_id, result)
                db.update_cluster_summary(cluster_id, result)

        db.complete_run(run_id, source_status, errors)
        return run_id
    except Exception as exc:
        db.fail_run(run_id, str(exc))
        raise


def _rank_and_cluster(
    posts: list[NormalizedPost],
    config: ScanConfig,
    plan: QueryPlan,
    domain_weight: float = 0.16,
) -> tuple[list[NormalizedPost], list]:
    ranked = rank_posts(posts, trusted_handles=config.trusted_handles)
    ranked = rerank_for_story_fit(ranked, plan, domain_weight=domain_weight)
    return ranked, cluster_posts(ranked)


def _dedupe_posts(posts: list[NormalizedPost]) -> list[NormalizedPost]:
    seen: set[tuple[str, str]] = set()
    unique: list[NormalizedPost] = []
    for post in posts:
        key = (post.platform, post.platform_post_id or post.url or post.text[:80])
        if key in seen:
            continue
        seen.add(key)
        unique.append(post)
    return unique


def _can_fetch_fallback(results: list[SourceResult]) -> bool:
    return any(result.platform == "x" and not result.error for result in results)


def _source_status(results: list[SourceResult], fallback_reason: str = "") -> dict:
    status: dict[str, dict] = {}
    for result in results:
        entry = status.setdefault(
            result.platform,
            {
                "error": None,
                "fetched_count": 0,
                "request_count": 0,
                "fallback_triggered": False,
                "fallback_reason": "",
                "rounds": [],
            },
        )
        entry["fetched_count"] += result.fetched_count
        entry["request_count"] += result.raw_request_count
        if result.error:
            entry["error"] = result.error if not entry["error"] else f"{entry['error']} | {result.error}"
        if result.query_mode == "story_fallback":
            entry["fallback_triggered"] = True
            entry["fallback_reason"] = fallback_reason
        entry["rounds"].append(
            {
                "mode": result.query_mode,
                "fetched_count": result.fetched_count,
                "error": result.error,
                "query": result.query,
            }
        )
    return status
