from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import NormalizedPost, QueryPlan, ScanConfig, StoryCluster, SummaryResult, utc_now
from .settings import env


def default_db_path() -> Path:
    return Path(env("BIOAI_DB_PATH", "data/bioai_fieldnotes.sqlite3") or "data/bioai_fieldnotes.sqlite3")


class Database:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                create table if not exists scan_runs (
                    id integer primary key autoincrement,
                    topic text not null,
                    prompt text not null default '',
                    keywords_json text not null,
                    exclusions_json text not null,
                    query_plan_json text not null default '{}',
                    platforms_json text not null,
                    trusted_handles_json text not null,
                    max_posts_per_source integer not null,
                    max_summaries integer not null,
                    max_estimated_cost_usd real not null,
                    allow_fallback_pricing integer not null default 0,
                    status text not null,
                    source_status_json text not null default '{}',
                    errors_json text not null default '[]',
                    created_at text not null,
                    completed_at text
                );

                create table if not exists posts (
                    id integer primary key autoincrement,
                    run_id integer not null references scan_runs(id),
                    platform text not null,
                    platform_post_id text not null,
                    url text not null,
                    text text not null,
                    author_handle text not null,
                    author_display_name text not null,
                    author_followers integer not null,
                    created_at text not null,
                    like_count integer not null,
                    repost_count integer not null,
                    reply_count integer not null,
                    quote_count integer not null,
                    extracted_urls_json text not null,
                    score real not null,
                    score_parts_json text not null,
                    raw_json text not null
                );

                create table if not exists clusters (
                    id integer primary key autoincrement,
                    run_id integer not null references scan_runs(id),
                    title text not null,
                    cluster_key text not null,
                    score real not null,
                    metrics_json text not null,
                    summary_status text not null,
                    summary text not null,
                    caveats text not null,
                    created_at text not null
                );

                create table if not exists cluster_posts (
                    cluster_id integer not null references clusters(id),
                    post_id integer not null references posts(id)
                );

                create table if not exists cost_events (
                    id integer primary key autoincrement,
                    run_id integer not null references scan_runs(id),
                    cluster_id integer references clusters(id),
                    provider text not null,
                    model text not null,
                    input_tokens integer not null,
                    output_tokens integer not null,
                    estimated_cost_usd real not null,
                    actual_cost_usd real,
                    status text not null,
                    error text,
                    created_at text not null
                );
                """
            )
            self._ensure_column(conn, "scan_runs", "prompt", "text not null default ''")
            self._ensure_column(conn, "scan_runs", "query_plan_json", "text not null default '{}'")

    def create_run(self, config: ScanConfig) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                insert into scan_runs (
                    topic, prompt, keywords_json, exclusions_json, platforms_json,
                    trusted_handles_json, max_posts_per_source, max_summaries,
                    max_estimated_cost_usd, allow_fallback_pricing, status, created_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)
                """,
                (
                    config.topic,
                    config.prompt,
                    _json(config.keywords),
                    _json(config.exclusions),
                    _json(config.platforms),
                    _json(config.trusted_handles),
                    config.max_posts_per_source,
                    config.max_summaries,
                    config.max_estimated_cost_usd,
                    1 if config.allow_fallback_pricing else 0,
                    _dt(utc_now()),
                ),
            )
            return int(cur.lastrowid)

    def update_run_query_plan(self, run_id: int, plan: QueryPlan) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                update scan_runs
                set topic = ?, keywords_json = ?, exclusions_json = ?,
                    trusted_handles_json = ?, query_plan_json = ?
                where id = ?
                """,
                (
                    plan.topic,
                    _json(plan.keywords),
                    _json(plan.exclusions),
                    _json(plan.trusted_handles),
                    _json({
                        "topic": plan.topic,
                        "keywords": plan.keywords,
                        "exclusions": plan.exclusions,
                        "trusted_handles": plan.trusted_handles,
                        "target_terms": plan.target_terms,
                        "context_terms": plan.context_terms,
                        "usage_cues": plan.usage_cues,
                        "negative_cues": plan.negative_cues,
                        "story_goal": plan.story_goal,
                        "notes": plan.notes,
                        "status": plan.status,
                    }),
                    run_id,
                ),
            )

    def complete_run(self, run_id: int, source_status: dict[str, Any], errors: list[str]) -> None:
        status = "completed" if not errors else "completed_with_errors"
        with self.connect() as conn:
            conn.execute(
                """
                update scan_runs
                set status = ?, source_status_json = ?, errors_json = ?, completed_at = ?
                where id = ?
                """,
                (status, _json(source_status), _json(errors), _dt(utc_now()), run_id),
            )

    def fail_run(self, run_id: int, error: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "update scan_runs set status = 'failed', errors_json = ?, completed_at = ? where id = ?",
                (_json([error]), _dt(utc_now()), run_id),
            )

    def insert_posts(self, run_id: int, posts: list[NormalizedPost]) -> dict[int, int]:
        mapping: dict[int, int] = {}
        with self.connect() as conn:
            for idx, post in enumerate(posts):
                cur = conn.execute(
                    """
                    insert into posts (
                        run_id, platform, platform_post_id, url, text, author_handle,
                        author_display_name, author_followers, created_at, like_count,
                        repost_count, reply_count, quote_count, extracted_urls_json,
                        score, score_parts_json, raw_json
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        post.platform,
                        post.platform_post_id,
                        post.url,
                        post.text,
                        post.author_handle,
                        post.author_display_name,
                        post.author_followers,
                        _dt(post.created_at),
                        post.like_count,
                        post.repost_count,
                        post.reply_count,
                        post.quote_count,
                        _json(post.extracted_urls),
                        post.score,
                        _json(post.score_parts),
                        _json(post.raw),
                    ),
                )
                mapping[idx] = int(cur.lastrowid)
        return mapping

    def insert_cluster(
        self,
        run_id: int,
        cluster: StoryCluster,
        post_ids: list[int],
    ) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                insert into clusters (
                    run_id, title, cluster_key, score, metrics_json,
                    summary_status, summary, caveats, created_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    cluster.title,
                    cluster.key,
                    cluster.score,
                    _json(cluster.metrics),
                    cluster.summary_status,
                    cluster.summary,
                    cluster.caveats,
                    _dt(utc_now()),
                ),
            )
            cluster_id = int(cur.lastrowid)
            conn.executemany(
                "insert into cluster_posts (cluster_id, post_id) values (?, ?)",
                [(cluster_id, post_id) for post_id in post_ids],
            )
            return cluster_id

    def insert_cost_event(self, run_id: int, cluster_id: int | None, result: SummaryResult) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert into cost_events (
                    run_id, cluster_id, provider, model, input_tokens, output_tokens,
                    estimated_cost_usd, actual_cost_usd, status, error, created_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    cluster_id,
                    result.provider,
                    result.model,
                    result.input_tokens,
                    result.output_tokens,
                    result.estimated_cost_usd,
                    result.actual_cost_usd,
                    result.status,
                    result.error,
                    _dt(utc_now()),
                ),
            )

    def update_cluster_summary(self, cluster_id: int, result: SummaryResult) -> None:
        with self.connect() as conn:
            conn.execute(
                "update clusters set summary_status = ?, summary = ?, caveats = ? where id = ?",
                (result.status, result.summary, result.caveats or result.error or "", cluster_id),
            )

    def list_runs(self, limit: int = 30) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(conn.execute("select * from scan_runs order by id desc limit ?", (limit,)))

    def get_run(self, run_id: int) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute("select * from scan_runs where id = ?", (run_id,)).fetchone()

    def list_clusters(self, run_id: int) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(conn.execute("select * from clusters where run_id = ? order by score desc", (run_id,)))

    def get_cluster(self, cluster_id: int) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute("select * from clusters where id = ?", (cluster_id,)).fetchone()

    def cluster_posts(self, cluster_id: int) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(
                conn.execute(
                    """
                    select p.* from posts p
                    join cluster_posts cp on cp.post_id = p.id
                    where cp.cluster_id = ?
                    order by p.score desc
                    """,
                    (cluster_id,),
                )
            )

    def run_posts(self, run_id: int) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(
                conn.execute(
                    "select * from posts where run_id = ? order by score desc",
                    (run_id,),
                )
            )

    def list_cost_events(self, limit: int = 100) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(conn.execute("select * from cost_events order by id desc limit ?", (limit,)))

    def list_cost_events_for_run(self, run_id: int) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(
                conn.execute(
                    "select * from cost_events where run_id = ? order by id",
                    (run_id,),
                )
            )

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        columns = {row["name"] for row in conn.execute(f"pragma table_info({table})")}
        if column not in columns:
            conn.execute(f"alter table {table} add column {column} {definition}")


def row_json(row: sqlite3.Row, key: str, default: Any) -> Any:
    try:
        return json.loads(row[key] or "")
    except Exception:
        return default


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _dt(value: datetime) -> str:
    return value.isoformat()
