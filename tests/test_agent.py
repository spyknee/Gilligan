"""Unit tests for agent.py — no chromadb, openai, or other external deps required."""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Stub out heavy optional imports before agent.py is loaded so the module
# imports cleanly even without chromadb / openai installed.
# ---------------------------------------------------------------------------
for _mod in ("chromadb", "chromadb.utils", "chromadb.utils.embedding_functions", "openai"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Make sure the repo root is on the path so `import agent` works from any cwd.
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import agent  # noqa: E402  (must come after stub setup)


# ===========================================================================
# 1. _chunk_text
# ===========================================================================
class TestChunkText(unittest.TestCase):
    def test_empty_returns_empty(self):
        self.assertEqual(agent._chunk_text("", 100, 20), [])

    def test_shorter_than_size_returns_one_chunk(self):
        chunks = agent._chunk_text("hello world", 100, 20)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "hello world")

    def test_correct_size_and_overlap(self):
        # 20-char text, size=10, overlap=3  → step=7
        # chunk 0: text[0:10]  = "abcdefghij"
        # chunk 1: text[7:17]  = "hijklmnopq"
        # chunk 2: text[14:24] = "opqrst" (clipped at len=20)
        text = "abcdefghijklmnopqrst"  # exactly 20 chars (a=0 … t=19)
        chunks = agent._chunk_text(text, 10, 3)
        self.assertEqual(chunks[0], "abcdefghij")
        self.assertEqual(chunks[1], "hijklmnopq")
        self.assertEqual(chunks[2], "opqrst")

    def test_whitespace_only_chunks_excluded(self):
        # A chunk that is all spaces should be filtered out.
        chunks = agent._chunk_text("     hello     ", 5, 0)
        for c in chunks:
            self.assertTrue(c.strip(), f"Whitespace-only chunk returned: {c!r}")


# ===========================================================================
# 2. cache_lookup / cache_store / cache_clear
# ===========================================================================
class TestCache(unittest.TestCase):
    def setUp(self):
        agent.cache_clear()
        # Ensure cache is enabled and use a very long TTL by default
        agent._profile = {
            "memory": {
                "cache_enabled": True,
                "cache_ttl_seconds": 86400,
                "cache_max_entries": 250,
            }
        }

    def tearDown(self):
        agent.cache_clear()
        agent._profile = {}

    def test_store_then_lookup_returns_value(self):
        agent.cache_store("k1", "answer one")
        result = agent.cache_lookup("k1")
        self.assertEqual(result, "answer one")

    def test_lookup_missing_key_returns_none(self):
        self.assertIsNone(agent.cache_lookup("nonexistent"))

    def test_ttl_expiry_returns_none(self):
        agent._profile["memory"]["cache_ttl_seconds"] = 1
        agent.cache_store("k2", "expiring answer")
        # Manually back-date the cached timestamp so it appears expired
        key = "k2"
        ts, ans = agent._query_cache[key]
        agent._query_cache[key] = (ts - 2, ans)  # 2 seconds in the past
        self.assertIsNone(agent.cache_lookup("k2"))

    def test_cache_clear_empties_cache(self):
        agent.cache_store("k3", "something")
        agent.cache_clear()
        self.assertIsNone(agent.cache_lookup("k3"))

    def test_max_entries_evicts_oldest(self):
        agent._profile["memory"]["cache_max_entries"] = 3
        # Seed the cache with deterministic timestamps so the oldest is predictable.
        t0 = 1_000_000.0
        agent._query_cache["a"] = (t0 + 0, "v1")
        agent._query_cache["b"] = (t0 + 1, "v2")
        agent._query_cache["c"] = (t0 + 2, "v3")
        # Adding a 4th entry should evict 'a' (oldest timestamp)
        agent.cache_store("d", "v4")
        self.assertEqual(len(agent._query_cache), 3)
        self.assertNotIn("a", agent._query_cache)

    def test_cache_disabled_returns_none(self):
        agent._profile["memory"]["cache_enabled"] = False
        agent.cache_store("k4", "should not store")
        self.assertIsNone(agent.cache_lookup("k4"))


# ===========================================================================
# 3. policy_check
# ===========================================================================
class TestPolicyCheck(unittest.TestCase):
    def setUp(self):
        self._original_blocklist = list(agent._blocklist)

    def tearDown(self):
        agent._blocklist[:] = self._original_blocklist

    def test_blocked_term_returns_rejection(self):
        agent._blocklist.append("badword")
        result = agent.policy_check("This contains badword in it")
        self.assertIsNotNone(result)
        self.assertIn("badword", result)

    def test_clean_input_returns_none(self):
        agent._blocklist.clear()
        agent._blocklist.append("badword")
        self.assertIsNone(agent.policy_check("totally clean input"))

    def test_case_insensitive(self):
        agent._blocklist.append("forbidden")
        self.assertIsNotNone(agent.policy_check("This has FORBIDDEN in caps"))

    def test_empty_blocklist_always_passes(self):
        agent._blocklist.clear()
        self.assertIsNone(agent.policy_check("anything goes"))


# ===========================================================================
# 4. run_episodic_decay — decay math (logic only, no chromadb)
# ===========================================================================
class TestEpisodicDecayMath(unittest.TestCase):
    """Test the decay arithmetic directly without touching ChromaDB."""

    def _build_meta(self, importance: float, days_ago: float) -> dict:
        """Return a metadata dict whose last_accessed is days_ago days in the past."""
        import datetime as dt
        ts = time.time() - days_ago * 86400
        iso = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).isoformat()
        return {
            "importance": str(importance),
            "last_accessed": iso,
            "source_type": "episodic",
            "timestamp": iso,
        }

    def test_decay_reduces_importance_correctly(self):
        decay_rate = 0.05
        days_since = 4.0
        importance = 0.8
        expected = importance - (decay_rate * days_since)  # 0.6
        meta = self._build_meta(importance, days_since)

        # Reproduce the exact formula used in run_episodic_decay
        last_ts = agent._ts_from_iso(meta["last_accessed"])
        now = time.time()
        actual_days = (now - last_ts) / 86400.0
        new_imp = float(meta["importance"]) - (decay_rate * actual_days)
        self.assertAlmostEqual(new_imp, expected, places=1)

    def test_below_threshold_entry_would_be_deleted(self):
        decay_rate = 0.05
        days_since = 20.0  # 0.5 - 1.0 = -0.5 → below any sane threshold
        importance = 0.5
        threshold = 0.2
        meta = self._build_meta(importance, days_since)

        last_ts = agent._ts_from_iso(meta["last_accessed"])
        now = time.time()
        actual_days = (now - last_ts) / 86400.0
        new_imp = float(meta["importance"]) - (decay_rate * actual_days)
        self.assertLess(new_imp, threshold)

    def test_above_threshold_entry_survives(self):
        decay_rate = 0.05
        days_since = 1.0  # 0.9 - 0.05 = 0.85 → well above threshold
        importance = 0.9
        threshold = 0.2
        meta = self._build_meta(importance, days_since)

        last_ts = agent._ts_from_iso(meta["last_accessed"])
        now = time.time()
        actual_days = (now - last_ts) / 86400.0
        new_imp = float(meta["importance"]) - (decay_rate * actual_days)
        self.assertGreater(new_imp, threshold)

    def test_decay_does_not_update_last_accessed(self):
        """Surviving entries must NOT have last_accessed overwritten during decay."""
        mock_col = MagicMock()
        meta = self._build_meta(0.9, 1.0)
        original_last_accessed = meta["last_accessed"]

        mock_col.get.return_value = {
            "ids": ["ep::abc"],
            "metadatas": [meta],
        }

        profile = {
            "memory": {
                "decay_rate_per_day": 0.05,
                "prune_threshold": 0.2,
            }
        }

        with patch("agent.col", return_value=mock_col):
            agent.run_episodic_decay(profile)

        mock_col.update.assert_called_once()
        updated_metas = mock_col.update.call_args[1]["metadatas"]
        self.assertEqual(len(updated_metas), 1)
        # The fix: last_accessed must NOT be changed to "now"
        self.assertEqual(updated_metas[0]["last_accessed"], original_last_accessed)

    def test_pruned_entry_is_deleted(self):
        """Entries below threshold are deleted, not updated."""
        mock_col = MagicMock()
        # importance=0.1, 10 days old → 0.1 - 0.5 = -0.4 → below threshold
        meta = self._build_meta(0.1, 10.0)

        mock_col.get.return_value = {
            "ids": ["ep::old"],
            "metadatas": [meta],
        }

        profile = {
            "memory": {
                "decay_rate_per_day": 0.05,
                "prune_threshold": 0.2,
            }
        }

        with patch("agent.col", return_value=mock_col):
            pruned = agent.run_episodic_decay(profile)

        self.assertEqual(pruned, 1)
        mock_col.delete.assert_called_once_with(ids=["ep::old"])
        mock_col.update.assert_not_called()


# ===========================================================================
# 5. _sha
# ===========================================================================
class TestSha(unittest.TestCase):
    def test_same_input_same_hash(self):
        self.assertEqual(agent._sha("hello"), agent._sha("hello"))

    def test_different_inputs_different_hashes(self):
        self.assertNotEqual(agent._sha("hello"), agent._sha("world"))

    def test_length_parameter_respected(self):
        self.assertEqual(len(agent._sha("test", length=8)), 8)
        self.assertEqual(len(agent._sha("test", length=32)), 32)

    def test_default_length_is_16(self):
        self.assertEqual(len(agent._sha("test")), 16)


# ===========================================================================
# 6. _deep_merge
# ===========================================================================
class TestDeepMerge(unittest.TestCase):
    def test_flat_override_wins(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99, "c": 3}
        result = agent._deep_merge(base, override)
        self.assertEqual(result, {"a": 1, "b": 99, "c": 3})

    def test_nested_dicts_merged_recursively(self):
        base = {"x": {"a": 1, "b": 2}}
        override = {"x": {"b": 99, "c": 3}}
        result = agent._deep_merge(base, override)
        self.assertEqual(result["x"], {"a": 1, "b": 99, "c": 3})

    def test_non_dict_override_replaces(self):
        base = {"x": {"a": 1}}
        override = {"x": "not a dict"}
        result = agent._deep_merge(base, override)
        self.assertEqual(result["x"], "not a dict")

    def test_base_not_mutated(self):
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        agent._deep_merge(base, override)
        self.assertEqual(base["a"]["b"], 1)

    def test_empty_override_returns_copy_of_base(self):
        base = {"a": 1}
        result = agent._deep_merge(base, {})
        self.assertEqual(result, base)
        self.assertIsNot(result, base)


if __name__ == "__main__":
    unittest.main()
