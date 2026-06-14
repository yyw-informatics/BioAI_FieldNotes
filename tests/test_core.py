from __future__ import annotations

import os
import tempfile
import unittest
import urllib.parse
from datetime import timedelta
from pathlib import Path

from bioai_fieldnotes.clustering import cluster_posts
from bioai_fieldnotes.connectors import XConnector
from bioai_fieldnotes.models import NormalizedPost, QueryPlan, ScanConfig, StoryCluster, utc_now
from bioai_fieldnotes.pricing import load_price_catalog
from bioai_fieldnotes.query_planner import apply_query_plan, plan_scan_query
from bioai_fieldnotes.ranking import rank_posts
from bioai_fieldnotes.settings import load_dotenv
from bioai_fieldnotes.story_selection import needs_story_fallback, rerank_for_story_fit
from bioai_fieldnotes.summarizer import Summarizer


class FakeJsonClient:
    def __init__(self, payload):
        self.payload = payload
        self.urls = []
        self.headers = []

    def get_json(self, url, headers=None):
        self.urls.append(url)
        self.headers.append(headers or {})
        return self.payload


class ScanConfigTests(unittest.TestCase):
    def test_m1_rejects_non_7_day_scans(self):
        with self.assertRaises(ValueError):
            ScanConfig(topic="AI", keywords=["AI"], days=30)

    def test_m1_rejects_non_x_platforms(self):
        with self.assertRaises(ValueError):
            ScanConfig(topic="AI", keywords=["AI"], platforms=["x", "mastodon"])

    def test_prompt_can_replace_manual_keywords(self):
        original_key = os.environ.get("OPENAI_API_KEY")
        try:
            os.environ.pop("OPENAI_API_KEY", None)
            config = ScanConfig(
                topic="Claude anecdotes",
                keywords=[],
                prompt="Find Claude Mythos stories in public health",
            )
            result = plan_scan_query(config)
            planned = apply_query_plan(config, result.plan)
            self.assertTrue(planned.keywords)
            self.assertTrue(any("Claude" in keyword for keyword in planned.keywords))
            self.assertTrue(planned.target_terms)
            self.assertTrue(planned.usage_cues)
            self.assertIn("public health", planned.context_terms)
        finally:
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key


class XQueryTests(unittest.TestCase):
    def test_story_prompt_query_anchors_target_and_usage_cues(self):
        config = ScanConfig(
            topic="Claude use stories",
            keywords=["Claude", "biomedical research"],
            target_terms=["Claude Mythos", "Claude Fable", "Claude"],
            context_terms=["biomedical research"],
            usage_cues=["I used", "we used", "built"],
        )
        query = XConnector(token="token")._build_query(config)
        self.assertIn('"Claude Mythos"', query)
        self.assertIn('"I used"', query)
        self.assertIn(" OR ", query)
        self.assertIn('"biomedical research"', query)

    def test_fallback_query_focuses_on_user_experience_cues(self):
        config = ScanConfig(
            topic="Claude use stories",
            keywords=["Claude", "biomedical research"],
            target_terms=["Claude Mythos", "Claude Fable", "Claude"],
            context_terms=["biomedical research"],
            usage_cues=["using", "used"],
        )
        query = XConnector(token="token")._build_query(config, query_mode="story_fallback")
        self.assertIn('"Claude Fable"', query)
        self.assertIn('"my workflow"', query)
        self.assertIn('"we built"', query)
        self.assertNotIn('"biomedical research"', query)

    def test_x_fetch_requests_and_normalizes_media_metadata(self):
        payload = {
            "data": [
                {
                    "id": "1",
                    "author_id": "u1",
                    "text": "Claude Fable built this in 3 hours.",
                    "created_at": "2026-06-10T12:00:00Z",
                    "attachments": {"media_keys": ["m1"]},
                    "public_metrics": {
                        "like_count": 10,
                        "retweet_count": 2,
                        "reply_count": 1,
                        "quote_count": 0,
                    },
                }
            ],
            "includes": {
                "users": [
                    {
                        "id": "u1",
                        "username": "builder",
                        "name": "Builder",
                        "public_metrics": {"followers_count": 100},
                    }
                ],
                "media": [
                    {
                        "media_key": "m1",
                        "type": "video",
                        "preview_image_url": "https://pbs.twimg.com/media/demo.jpg",
                        "alt_text": "A polished playable game demo",
                    }
                ],
            },
        }
        client = FakeJsonClient(payload)
        config = ScanConfig(topic="Claude demos", keywords=["Claude Fable"])
        result = XConnector(token="token", client=client).fetch(config)
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(client.urls[0]).query)

        self.assertIn("attachments.media_keys", query["expansions"][0])
        self.assertIn("attachments", query["tweet.fields"][0])
        self.assertIn("preview_image_url", query["media.fields"][0])
        self.assertEqual(result.posts[0].raw["media"][0]["type"], "video")
        self.assertIn("https://pbs.twimg.com/media/demo.jpg", result.posts[0].extracted_urls)


class RankingTests(unittest.TestCase):
    def test_relative_domain_signal_can_lift_smaller_science_posts(self):
        posts = [
            NormalizedPost(
                platform="x",
                platform_post_id="fashion",
                url="https://x.com/a/status/1",
                text="AI fashion trend",
                author_handle="general",
                author_followers=900000,
                like_count=1000,
                repost_count=20,
                created_at=utc_now() - timedelta(hours=8),
            ),
            NormalizedPost(
                platform="x",
                platform_post_id="science",
                url="https://x.com/lab/status/2",
                text="single cell foundation model for perturbation biology",
                author_handle="trustedlab",
                author_followers=12000,
                like_count=80,
                repost_count=30,
                reply_count=12,
                quote_count=6,
                created_at=utc_now() - timedelta(hours=2),
            ),
        ]
        ranked = rank_posts(posts, trusted_handles=["trustedlab"])
        self.assertEqual(ranked[0].author_handle, "trustedlab")
        self.assertGreater(ranked[0].score_parts["author_signal"], 0.5)

    def test_story_fit_lifts_actual_use_case_over_announcement(self):
        plan = QueryPlan(
            topic="Claude model anecdotes",
            keywords=["Claude Fable", "biomedical research"],
            target_terms=["Claude Fable"],
            context_terms=["biomedical research", "science"],
            usage_cues=["I used", "we used", "built", "tested"],
            negative_cues=["announcement", "launch", "benchmark"],
        )
        posts = [
            NormalizedPost(
                platform="x",
                platform_post_id="announcement",
                url="https://x.com/a/status/1",
                text="Huge Claude Fable launch announcement with benchmark details.",
                author_handle="news",
                author_followers=800000,
                like_count=3000,
                repost_count=400,
                created_at=utc_now() - timedelta(hours=2),
            ),
            NormalizedPost(
                platform="x",
                platform_post_id="usecase",
                url="https://x.com/b/status/2",
                text=(
                    "We used Claude Fable in our biomedical research workflow "
                    "to triage notes and build a reproducible analysis pipeline."
                ),
                author_handle="lab",
                author_followers=5000,
                like_count=80,
                repost_count=12,
                created_at=utc_now() - timedelta(hours=3),
            ),
        ]
        ranked = rerank_for_story_fit(rank_posts(posts), plan)
        self.assertEqual(ranked[0].platform_post_id, "usecase")
        self.assertEqual(ranked[0].score_parts["selection_label"], "likely_use_story")

    def test_sentiment_labels_yay_and_nooo(self):
        plan = QueryPlan(
            topic="Claude model anecdotes",
            keywords=["Claude Fable"],
            target_terms=["Claude Fable"],
            usage_cues=["I used", "we used", "tried"],
        )
        posts = [
            NormalizedPost(
                platform="x",
                platform_post_id="yay",
                url="https://x.com/a/status/1",
                text="Claude Fable changed how we work. We used it and the result was amazing.",
                author_handle="a",
            ),
            NormalizedPost(
                platform="x",
                platform_post_id="nooo",
                url="https://x.com/b/status/2",
                text="I tried Claude Fable and the prompt was flagged as a risk. Sad and disappointing.",
                author_handle="b",
            ),
        ]
        ranked = rerank_for_story_fit(rank_posts(posts), plan)
        labels = {post.platform_post_id: post.score_parts["sentiment_label"] for post in ranked}
        self.assertEqual(labels["yay"], "yay")
        self.assertEqual(labels["nooo"], "nooo")

    def test_sentiment_handles_social_idioms(self):
        plan = QueryPlan(
            topic="Claude model anecdotes",
            keywords=["Claude Fable"],
            target_terms=["Claude Fable"],
            usage_cues=["I used"],
        )
        posts = [
            NormalizedPost(
                platform="x",
                platform_post_id="cantbelieve",
                url="https://x.com/a/status/1",
                text="Can't believe I built a game using Claude Fable.",
                author_handle="a",
            ),
            NormalizedPost(
                platform="x",
                platform_post_id="cute",
                url="https://x.com/b/status/2",
                text="I used Claude Fable to make this dangerously cute pixel art scene.",
                author_handle="b",
            ),
        ]
        ranked = rerank_for_story_fit(rank_posts(posts), plan)
        labels = {post.platform_post_id: post.score_parts["sentiment_label"] for post in ranked}
        self.assertEqual(labels["cantbelieve"], "yay")
        self.assertEqual(labels["cute"], "yay")

    def test_media_demo_lifts_neutral_caption_to_positive_story(self):
        plan = QueryPlan(
            topic="Claude model anecdotes",
            keywords=["Claude Fable"],
            target_terms=["Claude Fable"],
            usage_cues=["I used", "we used", "built"],
        )
        posts = [
            NormalizedPost(
                platform="x",
                platform_post_id="visual",
                url="https://x.com/a/status/1",
                text="Claude Fable built this in 3 hours.",
                author_handle="a",
                raw={
                    "media": [
                        {
                            "type": "video",
                            "preview_image_url": "https://pbs.twimg.com/media/demo.jpg",
                            "alt_text": "A polished playable game demo",
                        }
                    ]
                },
            )
        ]
        ranked = rerank_for_story_fit(rank_posts(posts), plan)
        self.assertGreaterEqual(ranked[0].score_parts["media_signal"], 0.75)
        self.assertEqual(ranked[0].score_parts["sentiment_label"], "yay")
        self.assertEqual(ranked[0].score_parts["sentiment_visual_positive"], 1)

    def test_media_demo_does_not_override_explicit_negative_sentiment(self):
        plan = QueryPlan(
            topic="Claude model anecdotes",
            keywords=["Claude Fable"],
            target_terms=["Claude Fable"],
            usage_cues=["I used", "we used", "built"],
        )
        posts = [
            NormalizedPost(
                platform="x",
                platform_post_id="negative",
                url="https://x.com/a/status/1",
                text="I used Claude Fable and the result was flagged as a risk.",
                author_handle="a",
                raw={
                    "media": [
                        {
                            "type": "video",
                            "preview_image_url": "https://pbs.twimg.com/media/demo.jpg",
                            "alt_text": "A polished playable game demo",
                        }
                    ]
                },
            )
        ]
        ranked = rerank_for_story_fit(rank_posts(posts), plan)
        self.assertEqual(ranked[0].score_parts["sentiment_label"], "nooo")
        self.assertEqual(ranked[0].score_parts["sentiment_visual_positive"], 0)


class ClusteringTests(unittest.TestCase):
    def test_shared_url_clusters_posts(self):
        posts = rank_posts(
            [
                NormalizedPost(
                    platform="x",
                    platform_post_id="1",
                    url="https://x.com/a/status/1",
                    text="New AI biology model https://example.org/paper?utm_source=x",
                    author_handle="a",
                    extracted_urls=["https://example.org/paper?utm_source=x"],
                    like_count=10,
                ),
                NormalizedPost(
                    platform="x",
                    platform_post_id="2",
                    url="https://x.com/b/status/2",
                    text="This AI biology model is interesting https://example.org/paper",
                    author_handle="b",
                    extracted_urls=["https://example.org/paper"],
                    like_count=7,
                ),
            ]
        )
        clusters = cluster_posts(posts)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0].posts), 2)

    def test_shared_url_clusters_even_when_not_first_link(self):
        posts = rank_posts(
            [
                NormalizedPost(
                    platform="x",
                    platform_post_id="1",
                    url="https://x.com/a/status/1",
                    text="I used Claude Fable for a demo https://example.org/context https://video.example/demo?utm_source=x",
                    author_handle="a",
                    extracted_urls=[
                        "https://example.org/context",
                        "https://video.example/demo?utm_source=x",
                    ],
                    like_count=10,
                ),
                NormalizedPost(
                    platform="x",
                    platform_post_id="2",
                    url="https://x.com/b/status/2",
                    text="Another version of the same Claude Fable demo https://video.example/demo",
                    author_handle="b",
                    extracted_urls=["https://video.example/demo"],
                    like_count=7,
                ),
            ]
        )
        clusters = cluster_posts(posts)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].metrics["unique_author_count"], 2)

    def test_near_duplicate_story_captions_merge_across_different_links(self):
        posts = rank_posts(
            [
                NormalizedPost(
                    platform="x",
                    platform_post_id="1",
                    url="https://x.com/a/status/1",
                    text="I built a backrooms escape game using Claude Fable 5. Watch the video https://x.com/a/status/1/video/1",
                    author_handle="a",
                    extracted_urls=["https://x.com/a/status/1/video/1"],
                    like_count=10,
                ),
                NormalizedPost(
                    platform="x",
                    platform_post_id="2",
                    url="https://x.com/b/status/2",
                    text="Can't believe I built a backrooms escape game using Claude Fable 5 https://x.com/b/status/2/video/1",
                    author_handle="b",
                    extracted_urls=["https://x.com/b/status/2/video/1"],
                    like_count=7,
                ),
            ]
        )
        clusters = cluster_posts(posts)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0].posts), 2)

    def test_plain_x_status_links_do_not_force_merge(self):
        posts = rank_posts(
            [
                NormalizedPost(
                    platform="x",
                    platform_post_id="1",
                    url="https://x.com/a/status/1",
                    text="I used Claude Fable to refactor my shell profile.",
                    author_handle="a",
                    extracted_urls=["https://twitter.com/claudeai/status/123"],
                    like_count=10,
                ),
                NormalizedPost(
                    platform="x",
                    platform_post_id="2",
                    url="https://x.com/b/status/2",
                    text="We built a World Cup predictor with Claude Fable.",
                    author_handle="b",
                    extracted_urls=["https://twitter.com/claudeai/status/123"],
                    like_count=7,
                ),
            ]
        )
        clusters = cluster_posts(posts)
        self.assertEqual(len(clusters), 2)

    def test_cluster_metrics_include_sentiment(self):
        plan = QueryPlan(
            topic="Claude model anecdotes",
            keywords=["Claude Fable"],
            target_terms=["Claude Fable"],
            usage_cues=["I used"],
        )
        posts = rerank_for_story_fit(
            rank_posts([
                NormalizedPost(
                    platform="x",
                    platform_post_id="1",
                    url="https://x.com/a/status/1",
                    text="I used Claude Fable and it worked great.",
                    author_handle="a",
                    like_count=3,
                ),
            ]),
            plan,
        )
        clusters = cluster_posts(posts)
        self.assertEqual(clusters[0].metrics["sentiment_label"], "yay")
        self.assertGreater(clusters[0].metrics["sentiment_score"], 0)

    def test_story_fallback_triggers_when_clusters_lack_user_experience(self):
        weak = StoryCluster(
            title="Launch mention",
            posts=[],
            score=0.4,
            key="text:weak",
            metrics={
                "max_story_fit": 0.2,
                "usage_signal": 0.0,
                "selection_label": "keyword_mention",
            },
        )
        strong = StoryCluster(
            title="Use story",
            posts=[],
            score=0.9,
            key="text:strong",
            metrics={
                "max_story_fit": 0.8,
                "usage_signal": 1.0,
                "selection_label": "likely_use_story",
            },
        )
        self.assertTrue(needs_story_fallback([weak], max_summaries=15))
        self.assertFalse(needs_story_fallback([strong, strong, strong, strong, strong], max_summaries=15))


class SummarizerTests(unittest.TestCase):
    def test_budget_gate_blocks_summary(self):
        os.environ["BIOAI_ALLOW_FALLBACK_PRICING"] = "1"
        cluster = StoryCluster(
            title="Story",
            posts=[
                NormalizedPost(
                    platform="x",
                    platform_post_id="1",
                    url="https://x.com/a/status/1",
                    text="A long discussion about AI in single cell biology.",
                    author_handle="a",
                )
            ],
            score=0.9,
            key="text:test",
        )
        result = Summarizer(allow_fallback_pricing=True).summarize(cluster, max_cost_usd=0.0)
        self.assertEqual(result.status, "blocked_budget")

    def test_fallback_price_catalog_loads(self):
        catalog = load_price_catalog(allow_fallback=True)
        self.assertIsNotNone(catalog.get("openai", "gpt-5-mini"))


class SettingsTests(unittest.TestCase):
    def test_dotenv_loads_without_overwriting_real_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("BIOAI_TEST_TOKEN=from_file\nBIOAI_TEST_KEEP=from_file\n", encoding="utf-8")
            original_token = os.environ.get("BIOAI_TEST_TOKEN")
            original_keep = os.environ.get("BIOAI_TEST_KEEP")
            try:
                os.environ["BIOAI_TEST_TOKEN"] = "from_env"
                os.environ.pop("BIOAI_TEST_KEEP", None)
                load_dotenv(path)
                self.assertEqual(os.environ["BIOAI_TEST_TOKEN"], "from_env")
                self.assertEqual(os.environ["BIOAI_TEST_KEEP"], "from_file")
            finally:
                if original_token is None:
                    os.environ.pop("BIOAI_TEST_TOKEN", None)
                else:
                    os.environ["BIOAI_TEST_TOKEN"] = original_token
                if original_keep is None:
                    os.environ.pop("BIOAI_TEST_KEEP", None)
                else:
                    os.environ["BIOAI_TEST_KEEP"] = original_keep

    def test_dotenv_explicit_path_can_load_second_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / ".env"
            second = Path(tmp) / "other.env"
            first.write_text("BIOAI_TEST_FIRST=one\n", encoding="utf-8")
            second.write_text("BIOAI_TEST_SECOND=two\n", encoding="utf-8")
            original_first = os.environ.get("BIOAI_TEST_FIRST")
            original_second = os.environ.get("BIOAI_TEST_SECOND")
            try:
                os.environ.pop("BIOAI_TEST_FIRST", None)
                os.environ.pop("BIOAI_TEST_SECOND", None)
                load_dotenv(first)
                load_dotenv(second)
                self.assertEqual(os.environ["BIOAI_TEST_FIRST"], "one")
                self.assertEqual(os.environ["BIOAI_TEST_SECOND"], "two")
            finally:
                if original_first is None:
                    os.environ.pop("BIOAI_TEST_FIRST", None)
                else:
                    os.environ["BIOAI_TEST_FIRST"] = original_first
                if original_second is None:
                    os.environ.pop("BIOAI_TEST_SECOND", None)
                else:
                    os.environ["BIOAI_TEST_SECOND"] = original_second


if __name__ == "__main__":
    unittest.main()
