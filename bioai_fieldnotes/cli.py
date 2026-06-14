from __future__ import annotations

import argparse
import json

from .models import ScanConfig
from .pipeline import run_scan
from .query_planner import topic_from_prompt
from .storage import Database, row_json
from .text import parse_csvish


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bioai")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Run an on-demand 7-day social scan.")
    scan.add_argument("--prompt", default="", help="Natural-language research request.")
    scan.add_argument("--topic", default="")
    scan.add_argument("--keywords", default="", help="Optional comma-separated seed keywords.")
    scan.add_argument("--exclusions", default="")
    scan.add_argument("--platforms", default="x")
    scan.add_argument("--trusted-handles", default="")
    scan.add_argument("--days", type=int, default=7)
    scan.add_argument("--max-posts-per-source", type=int, default=100)
    scan.add_argument("--max-summaries", type=int, default=15)
    scan.add_argument("--max-estimated-cost-usd", type=float, default=1.0)
    scan.add_argument("--allow-fallback-pricing", action="store_true")
    scan.add_argument("--json", action="store_true", dest="as_json")

    args = parser.parse_args(argv)
    if args.command == "scan":
        topic = args.topic or topic_from_prompt(args.prompt)
        config = ScanConfig(
            topic=topic,
            keywords=parse_csvish(args.keywords),
            prompt=args.prompt,
            exclusions=parse_csvish(args.exclusions),
            platforms=parse_csvish(args.platforms),
            trusted_handles=parse_csvish(args.trusted_handles),
            days=args.days,
            max_posts_per_source=args.max_posts_per_source,
            max_summaries=args.max_summaries,
            max_estimated_cost_usd=args.max_estimated_cost_usd,
            allow_fallback_pricing=args.allow_fallback_pricing,
        )
        db = Database()
        run_id = run_scan(config, db)
        run = db.get_run(run_id)
        clusters = db.list_clusters(run_id)
        payload = {
            "run_id": run_id,
            "status": run["status"] if run else "unknown",
            "topic": run["topic"] if run else topic,
            "query_plan": row_json(run, "query_plan_json", {}) if run else {},
            "source_status": row_json(run, "source_status_json", {}) if run else {},
            "cluster_count": len(clusters),
        }
        if args.as_json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"Run {run_id} completed with {len(clusters)} cluster(s).")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
