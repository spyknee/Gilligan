#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  Gilligan  •  Agent v69-biomemory                           ║
║  Bio-inspired persistent memory: episodic + semantic        ║
║  Decay · Consolidation · Pruning                            ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import hashlib
import re
import subprocess
import datetime
import urllib.parse
import threading
from pathlib import Path
from typing import Optional

# ── Optional third-party imports ─────────────────────────────────────────────
try:
    import chromadb
    from chromadb.utils import embedding_functions as chroma_ef
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    print("[warn] chromadb not installed — memory disabled")

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("[warn] openai not installed — LLM calls disabled")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("[warn] requests not installed — web/crawl disabled")

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    try:
        import PyPDF2
        HAS_PDF = "pypdf2"
    except ImportError:
        HAS_PDF = False

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import urllib.robotparser
    HAS_ROBOTS = True
except ImportError:
    HAS_ROBOTS = False

# ── Core constants ────────────────────────────────────────────────────────────
AGENT_VERSION   = "v70-multimodel"
ASSISTANT_NAME  = "Gilligan"
BASE_DIR        = Path(os.environ.get("GILLIGAN_BASE_DIR", r"F:\ai-agent"))
WAKE_WORDS      = ["gilligan"]

PROFILE_FILE    = BASE_DIR / "profile.json"
NOTES_FILE      = BASE_DIR / "notes.md"
BLOCKLIST_FILE  = BASE_DIR / "blocklist.txt"
CHROMA_PATH     = BASE_DIR / "chroma_db"

# ── Collection names ──────────────────────────────────────────────────────────
COL_L0       = "mem_L0_overview"
COL_L1       = "mem_L1_mid"
COL_L2       = "mem_L2_detail"
COL_L3       = "mem_L3_fine"
COL_EPISODIC = "mem_episodic"

# ── Profile defaults (used if key missing from profile.json) ──────────────────
PROFILE_DEFAULTS = {
    "user": "user",
    "assistant_name": ASSISTANT_NAME,
    "style": {
        "keep_responses_short": True,
        "prefer_step_by_step": True,
        "ask_clarifying_questions": True,
    },
    "llm": {
        "active_model": "qwen",
        "models": {
            "qwen": {
                "base_url": "http://127.0.0.1:1234",
                "model": "qwen/qwen3-coder-next",
                "api_key": "lm-studio",
                "provider": "lmstudio",
            },
            "qwen35": {
                "base_url": "http://127.0.0.1:1234",
                "model": "qwen/qwen3.5-9b",
                "api_key": "lm-studio",
                "provider": "lmstudio",
            },
            "model3": {
                "base_url": "https://models.inference.ai.azure.com",
                "model": "PLACEHOLDER",
                "api_key_env": "GITHUB_TOKEN",
                "provider": "github",
            },
        },
    },
    "web": {
        "active_engine": "google",
        "engines": {
            "google":     "https://www.google.com/search?q={q}",
            "duckduckgo": "https://duckduckgo.com/?q={q}",
        },
        "browser": {
            "use_opera_gx": True,
            "opera_gx_path": r"C:\Users\%USERNAME%\AppData\Local\Programs\Opera GX\launcher.exe",
        },
        "crawl": {
            "max_pages": 100,
            "max_depth": 2,
            "delay_seconds": 1.0,
            "respect_robots": True,
            "user_agent": "GilliganCrawler/1.0",
            "max_bytes": 1000000,
            "max_links_per_page": 200,
            "timeout_seconds": 20,
        },
        "auto_ingest_after_crawl": False,
    },
    "run": {
        "whitelist": ["dir", "type", "echo"],
    },
    "memory": {
        "chunk_size": 900,
        "chunk_overlap": 120,
        "l0_summary_max_words": 120,
        "depth_enabled": True,
        "l1_summary_max_words": 200,
        "l3_chunk_size": 300,
        "l3_chunk_overlap": 60,
        "auto_zoom_top_sources": 4,
        "auto_l2_results_per_source": 1,
        "fast_zoom_top_sources": 4,
        "fast_l2_results_per_source": 0,
        "deep_zoom_top_sources": 7,
        "deep_l2_results_per_source": 2,
        "cache_enabled": True,
        "cache_ttl_seconds": 86400,
        "cache_max_entries": 250,
        "console_silence_http_logs": True,
        "verify_summaries_with_details": True,
        "verify_min_l2_per_source": 1,
        "verify_use_l3_if_available": True,
        "verify_max_distance": None,
        "llm_max_tokens_answer": 900,
        "llm_max_tokens_summary": 200,
        "llm_max_tokens_verifier": 8,
        "conversation_history_max_turns": 20,
        "include_notes_md": True,
        "auto_learn_enabled": True,
        "auto_learn_min_evidence_chars": 60,
        "auto_learn_require_user_approval": True,
        "auto_learn_use_fact_verifier": True,
        "auto_learn_requires_verified_sources": True,
        "ingest_notes_only_approved": True,
        "episodic_retention_days": 7,
        "decay_rate_per_day": 0.05,
        "prune_threshold": 0.2,
        "consolidation_on_startup": True,
        "episodic_top_k": 3,
        "episodic_importance_boost": 0.1,
        "episodic_default_importance": 0.5,
    },
}

# ── State ─────────────────────────────────────────────────────────────────────
_profile: dict = {}
_chroma_client = None
_collections: dict = {}
_embed_fn = None
_llm_client: Optional[object] = None
_working_memory: list = []          # list of {"role": ..., "content": ...}
_query_cache: dict = {}             # {cache_key: (timestamp, answer)}
_pending_facts: list = []           # auto-learn candidates awaiting approval
_auto_learn_enabled: bool = True
_web_enabled: bool = True
_blocklist: list = []


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Profile
# ═══════════════════════════════════════════════════════════════════════════════

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_profile() -> dict:
    """Load profile.json, merging with defaults."""
    global _profile
    _profile = dict(PROFILE_DEFAULTS)
    # Check local profile.json alongside agent.py first
    local_profile = Path(__file__).parent / "profile.json"
    candidates = [local_profile, PROFILE_FILE]
    for path in candidates:
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                _profile = _deep_merge(PROFILE_DEFAULTS, data)
                print(f"[profile] loaded {path}")
                break
            except Exception as e:
                print(f"[profile] error loading {path}: {e}")
    return _profile


def save_profile():
    """Persist current profile to disk."""
    path = Path(__file__).parent / "profile.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_profile, f, indent=2)
        print(f"[profile] saved → {path}")
    except Exception as e:
        print(f"[profile] save error: {e}")


def pcfg(key_path: str, default=None):
    """Get a nested profile value using dot-path notation."""
    parts = key_path.split(".")
    node = _profile
    for p in parts:
        if not isinstance(node, dict):
            return default
        node = node.get(p, None)
        if node is None:
            return default
    return node


def mcfg(key: str, default=None):
    """Shortcut for memory config values."""
    return pcfg(f"memory.{key}", default)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LLM Client
# ═══════════════════════════════════════════════════════════════════════════════

def get_active_model_cfg() -> dict:
    """Return the active model's config dict."""
    active = pcfg("llm.active_model", "qwen")
    models = pcfg("llm.models", {})
    return models.get(active, {"base_url": "http://127.0.0.1:1234", "model": "qwen/qwen3-coder-next"})


def build_llm_client() -> Optional[object]:
    """Create and return an OpenAI client for the active model provider."""
    global _llm_client
    if not HAS_OPENAI:
        return None
    cfg = get_active_model_cfg()
    provider = cfg.get("provider", "lmstudio")
    base_url = cfg.get("base_url", "http://127.0.0.1:1234")

    if provider == "github":
        # GitHub inference endpoint — use base_url as-is, key from env
        api_key_env = cfg.get("api_key_env", "GITHUB_TOKEN")
        api_key = os.environ.get(api_key_env, "")
        _llm_client = OpenAI(base_url=base_url, api_key=api_key)
    else:
        # LM Studio (and any other OpenAI-compatible local server) — append /v1
        api_key = cfg.get("api_key", "lm-studio")
        _llm_client = OpenAI(base_url=base_url + "/v1", api_key=api_key)

    return _llm_client


def llm_chat(messages: list, max_tokens: int = 900, temperature: float = 0.7) -> str:
    """
    Send a chat completion request to LM Studio.
    Returns the assistant response text, or an error string.
    """
    if not HAS_OPENAI or _llm_client is None:
        return "[LLM unavailable]"
    cfg = get_active_model_cfg()
    try:
        resp = _llm_client.chat.completions.create(
            model=cfg["model"],
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM error: {e}]"


def llm_complete(prompt: str, max_tokens: int = 200) -> str:
    """Single-turn prompt helper."""
    return llm_chat([{"role": "user", "content": prompt}], max_tokens=max_tokens)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ChromaDB & Embeddings
# ═══════════════════════════════════════════════════════════════════════════════

def init_chroma():
    """Initialise ChromaDB client and all collections."""
    global _chroma_client, _collections, _embed_fn
    if not HAS_CHROMA:
        print("[memory] chromadb not available — running without persistent memory")
        return

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Use the default embedding function (sentence-transformers if available)
    try:
        _embed_fn = chroma_ef.DefaultEmbeddingFunction()
    except Exception:
        _embed_fn = None

    kwargs = {"embedding_function": _embed_fn} if _embed_fn else {}

    for name in [COL_L0, COL_L1, COL_L2, COL_L3, COL_EPISODIC]:
        _collections[name] = _chroma_client.get_or_create_collection(name=name, **kwargs)

    print(f"[memory] chroma ready — collections: {list(_collections.keys())}")


def col(name: str):
    """Return a ChromaDB collection by name, or None."""
    return _collections.get(name)


def _sha(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _now_ts() -> float:
    return time.time()


def _ts_from_iso(iso: str) -> float:
    try:
        dt = datetime.datetime.fromisoformat(iso)
        return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Episodic Memory
# ═══════════════════════════════════════════════════════════════════════════════

def episodic_save(question: str, answer: str):
    """Save a Q&A exchange to episodic memory."""
    c = col(COL_EPISODIC)
    if c is None:
        return
    ts = _now_ts()
    iso = _now_iso()
    doc = f"Q: {question}\nA: {answer}"
    doc_id = f"episodic::{_sha(question + answer)}::{int(ts)}"
    importance = float(mcfg("episodic_default_importance", 0.5))
    try:
        c.add(
            documents=[doc],
            ids=[doc_id],
            metadatas=[{
                "source_type":    "episodic",
                "timestamp":      iso,
                "last_accessed":  iso,
                "importance":     str(importance),
            }],
        )
    except Exception as e:
        print(f"[episodic] save error: {e}")


def episodic_retrieve(query: str) -> list:
    """Retrieve top-K episodic memories relevant to query."""
    c = col(COL_EPISODIC)
    if c is None:
        return []
    k = int(mcfg("episodic_top_k", 3))
    try:
        count = c.count()
        if count == 0:
            return []
        results = c.query(query_texts=[query], n_results=min(k, count))
        docs = results.get("documents", [[]])[0]
        ids  = results.get("ids", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        # Boost importance + update last_accessed for retrieved entries
        boost = float(mcfg("episodic_importance_boost", 0.1))
        iso_now = _now_iso()
        for doc_id, meta in zip(ids, metas):
            imp = float(meta.get("importance", 0.5))
            imp = min(1.0, imp + boost)
            c.update(ids=[doc_id], metadatas=[{
                **meta,
                "importance":    str(imp),
                "last_accessed": iso_now,
            }])

        return docs
    except Exception as e:
        print(f"[episodic] retrieve error: {e}")
        return []


def episodic_count() -> int:
    """Return number of episodic memories."""
    c = col(COL_EPISODIC)
    if c is None:
        return 0
    try:
        return c.count()
    except Exception:
        return 0


def episodic_status() -> dict:
    """Return status dict: count, avg_importance, oldest."""
    c = col(COL_EPISODIC)
    if c is None:
        return {"count": 0, "avg_importance": 0.0, "oldest": "n/a"}
    try:
        count = c.count()
        if count == 0:
            return {"count": 0, "avg_importance": 0.0, "oldest": "n/a"}
        results = c.get(include=["metadatas"])
        metas = results.get("metadatas", [])
        importances = [float(m.get("importance", 0.5)) for m in metas]
        timestamps  = [m.get("timestamp", "") for m in metas]
        avg_imp = sum(importances) / len(importances) if importances else 0.0
        oldest  = min(timestamps) if timestamps else "n/a"
        return {"count": count, "avg_importance": avg_imp, "oldest": oldest}
    except Exception as e:
        return {"count": 0, "avg_importance": 0.0, "oldest": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Bio-Memory: Decay & Consolidation
# ═══════════════════════════════════════════════════════════════════════════════

def run_episodic_decay(profile: dict) -> int:
    """
    Decay episodic memory importance scores based on time since last access.
    Prune entries whose importance falls below prune_threshold.
    Returns number of pruned entries.
    """
    c = col(COL_EPISODIC)
    if c is None:
        return 0

    mem = profile.get("memory", {})
    decay_rate  = float(mem.get("decay_rate_per_day", 0.05))
    threshold   = float(mem.get("prune_threshold", 0.2))
    now         = _now_ts()
    iso_now     = _now_iso()
    pruned      = 0

    try:
        count = c.count()
        if count == 0:
            return 0
        results = c.get(include=["metadatas", "documents"])
        ids    = results.get("ids", [])
        metas  = results.get("metadatas", [])

        to_delete = []
        to_update_ids  = []
        to_update_meta = []

        for doc_id, meta in zip(ids, metas):
            last_iso  = meta.get("last_accessed", meta.get("timestamp", iso_now))
            last_ts   = _ts_from_iso(last_iso)
            days_since = (now - last_ts) / 86400.0
            importance = float(meta.get("importance", 0.5))
            new_importance = importance - (decay_rate * days_since)

            if new_importance < threshold:
                to_delete.append(doc_id)
                pruned += 1
            else:
                to_update_ids.append(doc_id)
                to_update_meta.append({
                    **meta,
                    "importance":    str(new_importance),
                    "last_accessed": iso_now,
                })

        if to_delete:
            c.delete(ids=to_delete)
        for uid, umeta in zip(to_update_ids, to_update_meta):
            c.update(ids=[uid], metadatas=[umeta])

    except Exception as e:
        print(f"[decay] error: {e}")

    return pruned


def run_consolidation(client, model_cfg: dict, profile: dict) -> int:
    """
    Consolidate episodic memories older than episodic_retention_days into
    semantic facts stored in COL_L2. Returns number of consolidated entries.
    """
    c_ep = col(COL_EPISODIC)
    c_l2 = col(COL_L2)
    if c_ep is None or c_l2 is None:
        return 0

    mem              = profile.get("memory", {})
    retention_days   = float(mem.get("episodic_retention_days", 7))
    max_tokens_sum   = int(mem.get("llm_max_tokens_summary", 200))
    now              = _now_ts()
    cutoff           = now - (retention_days * 86400.0)
    consolidated     = 0

    try:
        count = c_ep.count()
        if count == 0:
            return 0
        results = c_ep.get(include=["documents", "metadatas"])
        ids     = results.get("ids", [])
        docs    = results.get("documents", [])
        metas   = results.get("metadatas", [])

        to_delete = []

        for doc_id, doc, meta in zip(ids, docs, metas):
            ts_str = meta.get("timestamp", "")
            ts     = _ts_from_iso(ts_str)
            if ts > cutoff:
                continue  # Not old enough yet

            # Summarise with LLM
            prompt = (
                "Extract the most important factual statement from this conversation "
                "as a single sentence. Be concise.\n\n" + doc
            )
            msgs = [{"role": "user", "content": prompt}]
            fact = ""
            if client and HAS_OPENAI:
                try:
                    resp = client.chat.completions.create(
                        model=model_cfg["model"],
                        messages=msgs,
                        max_tokens=max_tokens_sum,
                        temperature=0.3,
                    )
                    fact = resp.choices[0].message.content.strip()
                except Exception as e:
                    fact = doc[:300]  # Fall back to raw text

            if not fact:
                fact = doc[:300]

            # Save to L2 as consolidated fact
            fact_id = f"consolidated::{_sha(doc)}::{int(now)}"
            try:
                c_l2.add(
                    documents=[fact],
                    ids=[fact_id],
                    metadatas=[{
                        "source_type":  "consolidated_episode",
                        "original_id":  doc_id,
                        "timestamp":    _now_iso(),
                        "source":       "episodic_consolidation",
                    }],
                )
                to_delete.append(doc_id)
                consolidated += 1
            except Exception as e:
                print(f"[consolidate] store error: {e}")

        if to_delete:
            c_ep.delete(ids=to_delete)

    except Exception as e:
        print(f"[consolidate] error: {e}")

    return consolidated


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Semantic Memory: Ingestion helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _chunk_text(text: str, size: int, overlap: int) -> list:
    """Split text into overlapping chunks."""
    chunks = []
    start  = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
        if start >= len(text):
            break
    return [c for c in chunks if c.strip()]


def _safe_add(collection, documents: list, ids: list, metadatas: list):
    """Add documents to a ChromaDB collection, skipping duplicate IDs."""
    if not documents:
        return
    try:
        # Get existing IDs to avoid duplicates
        existing = set()
        try:
            res = collection.get(ids=ids)
            existing = set(res.get("ids", []))
        except Exception:
            pass
        new_docs  = []
        new_ids   = []
        new_metas = []
        for d, i, m in zip(documents, ids, metadatas):
            if i not in existing:
                new_docs.append(d)
                new_ids.append(i)
                new_metas.append(m)
        if new_docs:
            collection.add(documents=new_docs, ids=new_ids, metadatas=new_metas)
    except Exception as e:
        print(f"[ingest] chroma add error: {e}")


def ingest_text(text: str, source_name: str, profile: dict):
    """
    Ingest a text blob into the semantic memory layers (L0 → L3).
    L0: LLM summary of entire doc
    L1: Larger chunks with summaries
    L2: Standard chunks
    L3: Fine-grained chunks
    """
    if not text.strip():
        return

    mem      = profile.get("memory", {})
    c_size   = int(mem.get("chunk_size", 900))
    c_over   = int(mem.get("chunk_overlap", 120))
    l0_words = int(mem.get("l0_summary_max_words", 120))
    l1_words = int(mem.get("l1_summary_max_words", 200))
    l3_size  = int(mem.get("l3_chunk_size", 300))
    l3_over  = int(mem.get("l3_chunk_overlap", 60))
    max_sum  = int(mem.get("llm_max_tokens_summary", 200))

    source_id = _sha(source_name + text[:200])

    # ── L2: standard chunks
    c_l2 = col(COL_L2)
    if c_l2:
        chunks = _chunk_text(text, c_size, c_over)
        docs, ids, metas = [], [], []
        for i, chunk in enumerate(chunks):
            docs.append(chunk)
            ids.append(f"l2::{source_id}::{i}")
            metas.append({"source": source_name, "chunk_index": str(i), "layer": "L2"})
        _safe_add(c_l2, docs, ids, metas)

    # ── L3: fine-grained chunks
    c_l3 = col(COL_L3)
    if c_l3 and mem.get("depth_enabled", True):
        chunks3 = _chunk_text(text, l3_size, l3_over)
        docs3, ids3, metas3 = [], [], []
        for i, chunk in enumerate(chunks3):
            docs3.append(chunk)
            ids3.append(f"l3::{source_id}::{i}")
            metas3.append({"source": source_name, "chunk_index": str(i), "layer": "L3"})
        _safe_add(c_l3, docs3, ids3, metas3)

    # ── L0: document-level summary
    c_l0 = col(COL_L0)
    if c_l0:
        summary = _summarise(text[:4000], f"Summarise this document in up to {l0_words} words.", max_sum)
        _safe_add(
            c_l0,
            [summary],
            [f"l0::{source_id}"],
            [{"source": source_name, "layer": "L0"}],
        )

    # ── L1: mid-level chunk summaries
    c_l1 = col(COL_L1)
    if c_l1:
        large_chunks = _chunk_text(text, c_size * 3, c_over * 3)
        docs1, ids1, metas1 = [], [], []
        for i, chunk in enumerate(large_chunks):
            s = _summarise(chunk[:4000], f"Summarise in up to {l1_words} words.", max_sum)
            docs1.append(s)
            ids1.append(f"l1::{source_id}::{i}")
            metas1.append({"source": source_name, "chunk_index": str(i), "layer": "L1"})
        _safe_add(c_l1, docs1, ids1, metas1)

    print(f"[ingest] {source_name} → L0/L1/L2/L3 done")


def _summarise(text: str, instruction: str, max_tokens: int) -> str:
    """Ask LLM to summarise text. Returns text unchanged if LLM unavailable."""
    if not HAS_OPENAI or _llm_client is None:
        return text[:500]
    try:
        resp = _llm_client.chat.completions.create(
            model=get_active_model_cfg()["model"],
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user",   "content": text},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return text[:500]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Semantic Memory: Retrieval
# ═══════════════════════════════════════════════════════════════════════════════

def _query_col(collection_name: str, query: str, n: int) -> list:
    """Query a single collection. Returns list of (document, metadata) tuples."""
    c = col(collection_name)
    if c is None:
        return []
    try:
        count = c.count()
        if count == 0:
            return []
        results = c.query(query_texts=[query], n_results=min(n, count))
        docs   = results.get("documents", [[]])[0]
        metas  = results.get("metadatas", [[]])[0]
        return list(zip(docs, metas))
    except Exception as e:
        print(f"[query] {collection_name} error: {e}")
        return []


def retrieve_semantic(query: str, mode: str = "auto", profile: dict = None) -> list:
    """
    Retrieve relevant semantic memory chunks.
    mode: 'fast' | 'deep' | 'auto'
    Returns list of document strings.
    """
    if profile is None:
        profile = _profile
    mem = profile.get("memory", {})

    if mode == "fast":
        top_l0 = int(mem.get("fast_zoom_top_sources", 4))
        l2_per = int(mem.get("fast_l2_results_per_source", 0))
    elif mode == "deep":
        top_l0 = int(mem.get("deep_zoom_top_sources", 7))
        l2_per = int(mem.get("deep_l2_results_per_source", 2))
    else:  # auto
        top_l0 = int(mem.get("auto_zoom_top_sources", 4))
        l2_per = int(mem.get("auto_l2_results_per_source", 1))

    results = []

    # L0 overview pass
    l0_hits = _query_col(COL_L0, query, top_l0)
    for doc, meta in l0_hits:
        results.append(doc)

    # L2 detail zoom
    if l2_per > 0:
        l2_hits = _query_col(COL_L2, query, top_l0 * l2_per * 2)
        for doc, _ in l2_hits[:top_l0 * l2_per]:
            results.append(doc)

    # L3 fine zoom (deep mode or depth_enabled)
    if mode == "deep" and mem.get("depth_enabled", True):
        l3_hits = _query_col(COL_L3, query, top_l0 * 3)
        for doc, _ in l3_hits:
            results.append(doc)

    return results


def build_context_block(query: str, mode: str = "auto") -> str:
    """Build a memory context block to inject into the system prompt."""
    chunks = retrieve_semantic(query, mode)
    if not chunks:
        return ""
    lines = ["MEMORY CONTEXT:"]
    for i, chunk in enumerate(chunks[:10], 1):
        lines.append(f"[M-{i}] {chunk[:400]}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Working Memory
# ═══════════════════════════════════════════════════════════════════════════════

def working_memory_add(role: str, content: str):
    """Add a turn to working memory, respecting max_turns."""
    max_turns = int(mcfg("conversation_history_max_turns", 20))
    _working_memory.append({"role": role, "content": content})
    # Keep last max_turns * 2 messages (each turn = user + assistant)
    while len(_working_memory) > max_turns * 2:
        _working_memory.pop(0)


def working_memory_clear():
    _working_memory.clear()


def working_memory_as_history() -> list:
    """Return working memory as a list of message dicts."""
    return list(_working_memory)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Query Cache
# ═══════════════════════════════════════════════════════════════════════════════

def cache_lookup(key: str) -> Optional[str]:
    """Return cached answer or None."""
    if not mcfg("cache_enabled", True):
        return None
    entry = _query_cache.get(key)
    if not entry:
        return None
    ts, answer = entry
    ttl = int(mcfg("cache_ttl_seconds", 86400))
    if time.time() - ts > ttl:
        del _query_cache[key]
        return None
    return answer


def cache_store(key: str, answer: str):
    """Store an answer in the query cache."""
    if not mcfg("cache_enabled", True):
        return
    max_entries = int(mcfg("cache_max_entries", 250))
    if len(_query_cache) >= max_entries:
        # Evict oldest
        oldest = min(_query_cache, key=lambda k: _query_cache[k][0])
        del _query_cache[oldest]
    _query_cache[key] = (time.time(), answer)


def cache_clear():
    _query_cache.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Policy Precheck / Blocklist
# ═══════════════════════════════════════════════════════════════════════════════

def load_blocklist():
    """Load blocklist.txt if it exists."""
    global _blocklist
    if BLOCKLIST_FILE.exists():
        try:
            lines = BLOCKLIST_FILE.read_text(encoding="utf-8").splitlines()
            _blocklist = [l.strip().lower() for l in lines if l.strip() and not l.startswith("#")]
        except Exception:
            _blocklist = []


def policy_check(text: str) -> Optional[str]:
    """
    Returns a rejection reason string if blocked, else None.
    Checks against blocklist words/phrases.
    """
    lower = text.lower()
    for term in _blocklist:
        if term and term in lower:
            return f"Blocked: input contains restricted term '{term}'"
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — Run Sandbox
# ═══════════════════════════════════════════════════════════════════════════════

def run_sandbox(command: str) -> str:
    """
    Execute a whitelisted shell command and return output.
    Only commands whose first token is in the whitelist are permitted.
    """
    whitelist = pcfg("run.whitelist", ["dir", "type", "echo"])
    tokens = command.strip().split()
    if not tokens:
        return "[run] empty command"
    cmd_name = tokens[0].lower()
    if cmd_name not in [w.lower() for w in whitelist]:
        return f"[run] '{cmd_name}' is not in whitelist. Allowed: {whitelist}"
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = result.stdout or result.stderr or "(no output)"
        return output[:2000]
    except subprocess.TimeoutExpired:
        return "[run] command timed out"
    except Exception as e:
        return f"[run] error: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — Web Browser & Crawl
# ═══════════════════════════════════════════════════════════════════════════════

def open_web_search(query: str):
    """Open a web search in the configured browser."""
    engine_name = pcfg("web.active_engine", "google")
    engines     = pcfg("web.engines", {})
    template    = engines.get(engine_name, "https://www.google.com/search?q={q}")
    url         = template.replace("{q}", urllib.parse.quote_plus(query))

    use_opera   = pcfg("web.browser.use_opera_gx", False)
    opera_path  = pcfg("web.browser.opera_gx_path", "")

    if use_opera and opera_path:
        expanded = os.path.expandvars(opera_path)
        try:
            subprocess.Popen([expanded, url])
            print(f"[web] opened in Opera GX: {url}")
            return
        except Exception as e:
            print(f"[web] Opera GX error: {e}")

    # Fallback: default browser
    import webbrowser
    webbrowser.open(url)
    print(f"[web] opened: {url}")


def _robots_allowed(url: str, user_agent: str) -> bool:
    """Check robots.txt for the given URL."""
    if not HAS_ROBOTS:
        return True
    try:
        parsed = urllib.parse.urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True


def crawl_site(start_url: str, profile: dict) -> dict:
    """
    Crawl a website starting at start_url.
    Returns dict: {url: text_content}
    Respects crawl config from profile.
    """
    if not HAS_REQUESTS:
        print("[crawl] requests library not available")
        return {}

    cfg         = profile.get("web", {}).get("crawl", {})
    max_pages   = int(cfg.get("max_pages", 100))
    max_depth   = int(cfg.get("max_depth", 2))
    delay       = float(cfg.get("delay_seconds", 1.0))
    respect_rob = bool(cfg.get("respect_robots", True))
    user_agent  = str(cfg.get("user_agent", "GilliganCrawler/1.0"))
    max_bytes   = int(cfg.get("max_bytes", 1_000_000))
    max_links   = int(cfg.get("max_links_per_page", 200))
    timeout     = int(cfg.get("timeout_seconds", 20))

    visited  = {}
    queue    = [(start_url, 0)]
    seen_urls = {start_url}
    parsed_base = urllib.parse.urlparse(start_url)
    base_domain = parsed_base.netloc

    headers = {"User-Agent": user_agent}

    while queue and len(visited) < max_pages:
        url, depth = queue.pop(0)
        if depth > max_depth:
            continue
        if respect_rob and not _robots_allowed(url, user_agent):
            print(f"[crawl] robots.txt blocked: {url}")
            continue

        try:
            resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
            content = b""
            for chunk in resp.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > max_bytes:
                    break
            text = content.decode("utf-8", errors="replace")

            if HAS_BS4:
                soup = BeautifulSoup(text, "html.parser")
                page_text = soup.get_text(separator=" ", strip=True)
                # Extract links
                links = []
                for a in soup.find_all("a", href=True):
                    href = urllib.parse.urljoin(url, a["href"])
                    p = urllib.parse.urlparse(href)
                    if p.netloc == base_domain and href not in seen_urls:
                        links.append(href)
                        seen_urls.add(href)
                    if len(links) >= max_links:
                        break
            else:
                page_text = text
                links = []

            visited[url] = page_text
            print(f"[crawl] {len(visited)}/{max_pages} — {url}")

            if depth < max_depth:
                for link in links:
                    queue.append((link, depth + 1))

        except Exception as e:
            print(f"[crawl] error {url}: {e}")

        time.sleep(delay)

    return visited


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — File Ingestion
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text(path: Path) -> str:
    """Extract plain text from a file (PDF, DOCX, TXT, HTML, MD)."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(path)
    elif suffix == ".docx":
        return _extract_docx(path)
    elif suffix in (".html", ".htm"):
        return _extract_html(path)
    elif suffix in (".txt", ".md", ".rst", ".csv", ".log"):
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"[read error: {e}]"
    else:
        # Try as plain text
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""


def _extract_pdf(path: Path) -> str:
    if HAS_PDF is True:  # pdfplumber
        try:
            import pdfplumber
            text = []
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text.append(t)
            return "\n".join(text)
        except Exception as e:
            return f"[PDF error: {e}]"
    elif HAS_PDF == "pypdf2":
        try:
            import PyPDF2
            text = []
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text.append(t)
            return "\n".join(text)
        except Exception as e:
            return f"[PDF error: {e}]"
    else:
        return "[PDF support not available — install pdfplumber]"


def _extract_docx(path: Path) -> str:
    if not HAS_DOCX:
        return "[DOCX support not available — install python-docx]"
    try:
        doc = DocxDocument(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        return f"[DOCX error: {e}]"


def _extract_html(path: Path) -> str:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        if HAS_BS4:
            soup = BeautifulSoup(raw, "html.parser")
            return soup.get_text(separator=" ", strip=True)
        return re.sub(r"<[^>]+>", " ", raw)
    except Exception as e:
        return f"[HTML error: {e}]"


def cmd_ingest(path_str: str):
    """Ingest a file or directory into semantic memory."""
    path = Path(path_str)
    if not path.exists():
        print(f"[ingest] path not found: {path}")
        return

    files = []
    if path.is_dir():
        for ext in ["*.pdf", "*.docx", "*.txt", "*.md", "*.html", "*.htm", "*.rst", "*.csv"]:
            files.extend(path.glob(ext))
            files.extend(path.rglob(ext))
        files = list(set(files))
    else:
        files = [path]

    for f in files:
        print(f"[ingest] processing {f.name}…")
        text = extract_text(f)
        if text.strip():
            ingest_text(text, f.name, _profile)
        else:
            print(f"[ingest] no text extracted from {f.name}")

    print(f"[ingest] done — {len(files)} file(s)")


def cmd_rebuild():
    """Drop and re-create all semantic memory collections."""
    if not HAS_CHROMA or _chroma_client is None:
        print("[rebuild] chromadb not available")
        return
    confirm = input("[rebuild] This will delete all semantic memory. Type YES to confirm: ")
    if confirm.strip().upper() != "YES":
        print("[rebuild] cancelled")
        return
    for name in [COL_L0, COL_L1, COL_L2, COL_L3]:
        try:
            _chroma_client.delete_collection(name=name)
        except Exception:
            pass
    for name in [COL_L0, COL_L1, COL_L2, COL_L3]:
        kwargs = {"embedding_function": _embed_fn} if _embed_fn else {}
        _collections[name] = _chroma_client.get_or_create_collection(name=name, **kwargs)
    print("[rebuild] semantic memory cleared and collections re-created")


def cmd_reingest(path_str: str = ""):
    """Reingest all files from BASE_DIR (or a given path)."""
    target = Path(path_str) if path_str else BASE_DIR
    print(f"[reingest] scanning {target}…")
    cmd_ingest(str(target))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — Notes System
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_notes_file():
    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not NOTES_FILE.exists():
        NOTES_FILE.write_text("# Gilligan Notes\n\n", encoding="utf-8")


def notes_append(content: str, tag: str = "NOTE"):
    """Append a tagged note to notes.md."""
    _ensure_notes_file()
    ts = _now_iso()
    entry = f"\n## [{tag}] {ts}\n{content}\n"
    with open(NOTES_FILE, "a", encoding="utf-8") as f:
        f.write(entry)


def notes_read() -> str:
    """Read all notes."""
    if not NOTES_FILE.exists():
        return ""
    return NOTES_FILE.read_text(encoding="utf-8")


def parse_keep_tags(text: str) -> list:
    """
    Extract KEEP / KEEP_STEPS / KEEP_FACT blocks from LLM response.
    Returns list of (tag, content) tuples.
    """
    patterns = [
        (r"KEEP_STEPS:(.*?)(?=KEEP_|$)", "KEEP_STEPS"),
        (r"KEEP_FACT:(.*?)(?=KEEP_|$)", "KEEP_FACT"),
        (r"KEEP:(.*?)(?=KEEP_|$)", "KEEP"),
    ]
    results = []
    for pattern, tag in patterns:
        for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
            content = match.group(1).strip()
            if content:
                results.append((tag, content))
    return results


def cmd_ingest_notes():
    """Ingest notes.md into semantic memory."""
    if NOTES_FILE.exists():
        text = NOTES_FILE.read_text(encoding="utf-8")
        ingest_text(text, "notes.md", _profile)
        print("[notes] notes.md ingested into memory")
    else:
        print("[notes] notes.md not found")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — Auto-Learn
# ═══════════════════════════════════════════════════════════════════════════════

def auto_learn_check(question: str, answer: str, profile: dict):
    """
    After each Q&A, check if the answer contains learnable facts.
    If auto_learn_enabled and conditions met, queue for approval.
    """
    global _pending_facts
    if not _auto_learn_enabled:
        return
    mem = profile.get("memory", {})
    min_chars = int(mem.get("auto_learn_min_evidence_chars", 60))
    if len(answer) < min_chars:
        return

    # Simple heuristic: sentences that look factual
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    candidates = [s for s in sentences if len(s) > min_chars and not s.startswith("?")]
    for cand in candidates[:3]:
        _pending_facts.append({
            "question": question,
            "candidate": cand,
            "timestamp": _now_iso(),
            "source": "auto_learn",
        })


def cmd_review_pending():
    """Review and approve/reject pending auto-learn facts."""
    global _pending_facts
    if not _pending_facts:
        print("[auto-learn] no pending facts")
        return
    print(f"[auto-learn] {len(_pending_facts)} pending fact(s)")
    to_keep = []
    for i, item in enumerate(_pending_facts):
        print(f"\n  [{i+1}] From: {item['question'][:60]}")
        print(f"       Fact: {item['candidate'][:120]}")
        choice = input("  Approve? [y/n/q(quit)]: ").strip().lower()
        if choice == "q":
            to_keep.extend(_pending_facts[i:])
            break
        elif choice == "y":
            # Ingest the fact into L2
            c_l2 = col(COL_L2)
            if c_l2:
                fact_id = f"learned::{_sha(item['candidate'])}::{int(time.time())}"
                try:
                    c_l2.add(
                        documents=[item["candidate"]],
                        ids=[fact_id],
                        metadatas=[{
                            "source":      "auto_learn",
                            "source_type": "learned_fact",
                            "timestamp":   item["timestamp"],
                            "question":    item["question"][:200],
                        }],
                    )
                    print(f"  [auto-learn] saved: {item['candidate'][:60]}…")
                except Exception as e:
                    print(f"  [auto-learn] save error: {e}")
        else:
            print("  [auto-learn] rejected")
    _pending_facts = to_keep


def cmd_pending_count():
    print(f"[auto-learn] pending: {len(_pending_facts)}")


def cmd_pending_purge():
    global _pending_facts
    _pending_facts = []
    print("[auto-learn] pending queue cleared")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 16 — Answer Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def build_system_prompt(query: str, mode: str = "auto") -> str:
    """Build the full system prompt with memory context."""
    style      = _profile.get("style", {})
    user_name  = _profile.get("user", "user")
    asst_name  = _profile.get("assistant_name", ASSISTANT_NAME)

    lines = [
        f"You are {asst_name}, a helpful AI assistant.",
        f"You are speaking with {user_name}.",
    ]

    if style.get("keep_responses_short"):
        lines.append("Keep responses concise.")
    if style.get("prefer_step_by_step"):
        lines.append("Prefer step-by-step explanations when helpful.")
    if style.get("ask_clarifying_questions"):
        lines.append("Ask clarifying questions when the request is ambiguous.")

    # Memory context
    mem_ctx = build_context_block(query, mode)
    if mem_ctx:
        lines.append("\n" + mem_ctx)

    # Episodic context
    ep_docs = episodic_retrieve(query)
    if ep_docs:
        lines.append("\nPAST CONTEXT (from memory):")
        for i, doc in enumerate(ep_docs, 1):
            lines.append(f"[EP-{i}] {doc[:300]}")

    # Notes
    if mcfg("include_notes_md", True) and NOTES_FILE.exists():
        notes = notes_read()
        if notes.strip():
            lines.append(f"\nNOTES:\n{notes[:1000]}")

    return "\n".join(lines)


def answer(question: str, mode: str = "auto") -> str:
    """
    Full answer pipeline:
    1. Policy check
    2. Cache lookup
    3. Build system prompt (memory + episodic context)
    4. Call LLM with working memory history
    5. Save to episodic + working memory
    6. Check for KEEP tags
    7. Auto-learn
    8. Cache store
    """
    # Policy precheck
    block = policy_check(question)
    if block:
        return block

    # Cache
    cache_key = _sha(question + mode)
    cached = cache_lookup(cache_key)
    if cached:
        return f"[cached] {cached}"

    # Detect QUOTEPLUS mode
    if question.upper().startswith("QUOTEPLUS:"):
        question_clean = question[len("QUOTEPLUS:"):].strip()
        mode = "deep"
    else:
        question_clean = question

    max_tokens = int(mcfg("llm_max_tokens_answer", 900))

    # Build messages
    system_msg = build_system_prompt(question_clean, mode)
    messages   = [{"role": "system", "content": system_msg}]
    messages  += working_memory_as_history()
    messages.append({"role": "user", "content": question_clean})

    # LLM call
    resp = llm_chat(messages, max_tokens=max_tokens)

    # Save to working memory
    working_memory_add("user",      question_clean)
    working_memory_add("assistant", resp)

    # Save to episodic memory
    episodic_save(question_clean, resp)

    # Process KEEP tags
    keep_items = parse_keep_tags(resp)
    for tag, content in keep_items:
        notes_append(content, tag)
        print(f"[notes] saved {tag} block")

    # Auto-learn
    auto_learn_check(question_clean, resp, _profile)

    # Cache
    cache_store(cache_key, resp)

    return resp


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 17 — CLI Command Handlers
# ═══════════════════════════════════════════════════════════════════════════════

def print_help():
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  {ASSISTANT_NAME}  •  {AGENT_VERSION}
║  Commands
╠══════════════════════════════════════════════════════════╣
║  MEMORY
║    /ingest <path>         Ingest file or directory
║    /reingest [path]       Re-ingest files from BASE_DIR
║    /rebuild               Wipe + re-create semantic memory
║    /ingest_notes          Ingest notes.md into memory
║    /consolidate           Manually trigger episode consolidation
║    /memory_status         Show episodic memory stats
║    /memory_clear_episodic Wipe all episodic memories
║    /cache_clear           Clear query cache
║    /history_clear         Clear conversation history
║
║  AUTO-LEARN
║    /review_pending        Review & approve pending facts
║    /pending_count         Show pending fact count
║    /pending_purge         Discard all pending facts
║    /auto_learn_on         Enable auto-learn
║    /auto_learn_off        Disable auto-learn
║    /auto_learn_toggle     Toggle auto-learn
║    /auto_learn_status     Show auto-learn status
║
║  MODEL
║    /model                 Show active model
║    /model_use <name>      Switch to model by name
║    /model_toggle          Toggle between models
║
║  WEB
║    /web <query>           Open web search
║    /web_engine <name>     Switch search engine
║    /web_toggle            Toggle web search on/off
║    /crawl <url>           Crawl a website
║
║  RUN
║    /run <cmd>             Run a whitelisted command
║    /run_whitelist_show    Show command whitelist
║    /run_whitelist_add <c> Add command to whitelist
║
║  PROFILE
║    /profile_show          Show current profile
║    /profile_add <k> <v>   Set a profile value (dot-path)
║
║  DIAGNOSTICS
║    /doctor                System health check
║    /mega_torture          Memory stress test
║    /depth_test            Depth retrieval test
║    /help                  Show this help
║    /quit                  Exit
╚══════════════════════════════════════════════════════════╝
  Modes: prefix query with FAST: / DEEP: / QUOTEPLUS:
""")


def cmd_doctor():
    """Print system health status."""
    ep_status = episodic_status()
    model_cfg = get_active_model_cfg()

    print(f"""
[doctor] ── {ASSISTANT_NAME} {AGENT_VERSION} ──────────────────────────
  Base dir:          {BASE_DIR}
  Profile:           {Path(__file__).parent / 'profile.json'}
  Notes:             {NOTES_FILE}
  ChromaDB:          {CHROMA_PATH}

  LLM:
    Active model:    {pcfg('llm.active_model', 'n/a')}
    Model name:      {model_cfg.get('model', 'n/a')}
    Base URL:        {model_cfg.get('base_url', 'n/a')}

  Memory collections:
    L0 overview:     {_col_count(COL_L0)} entries
    L1 mid:          {_col_count(COL_L1)} entries
    L2 detail:       {_col_count(COL_L2)} entries
    L3 fine:         {_col_count(COL_L3)} entries
    Episodic:        {ep_status['count']} entries

  Episodic memory:
    Count:           {ep_status['count']}
    Avg importance:  {ep_status['avg_importance']:.3f}
    Oldest:          {ep_status['oldest']}

  Working memory:    {len(_working_memory)} messages
  Query cache:       {len(_query_cache)} entries
  Pending facts:     {len(_pending_facts)}
  Auto-learn:        {'on' if _auto_learn_enabled else 'off'}
  Web search:        {'on' if _web_enabled else 'off'}
  Active engine:     {pcfg('web.active_engine', 'n/a')}
  Blocklist terms:   {len(_blocklist)}

  Libraries:
    chromadb:        {'✓' if HAS_CHROMA else '✗'}
    openai:          {'✓' if HAS_OPENAI else '✗'}
    requests:        {'✓' if HAS_REQUESTS else '✗'}
    beautifulsoup4:  {'✓' if HAS_BS4 else '✗'}
    pdfplumber:      {'✓' if HAS_PDF is True else ('PyPDF2' if HAS_PDF else '✗')}
    python-docx:     {'✓' if HAS_DOCX else '✗'}
──────────────────────────────────────────────────────────
""")


def _col_count(name: str) -> int:
    c = col(name)
    if c is None:
        return 0
    try:
        return c.count()
    except Exception:
        return 0


def cmd_consolidate():
    """Manually trigger episodic consolidation."""
    model_cfg = get_active_model_cfg()
    n = run_consolidation(_llm_client, model_cfg, _profile)
    print(f"[consolidate] {n} episode(s) consolidated into semantic memory")


def cmd_memory_status():
    """Show episodic memory statistics."""
    st = episodic_status()
    print(f"""[memory_status]
  Episodic count:    {st['count']}
  Avg importance:    {st['avg_importance']:.3f}
  Oldest entry:      {st['oldest']}
  Working memory:    {len(_working_memory)} messages
  Cache entries:     {len(_query_cache)}
  Pending facts:     {len(_pending_facts)}
""")


def cmd_memory_clear_episodic():
    """Wipe all episodic memories after confirmation."""
    c = col(COL_EPISODIC)
    if c is None:
        print("[episodic] not available")
        return
    count = c.count()
    confirm = input(f"[episodic] This will delete {count} episodic memories. Type YES to confirm: ")
    if confirm.strip().upper() != "YES":
        print("[episodic] cancelled")
        return
    try:
        if not HAS_CHROMA or _chroma_client is None:
            print("[episodic] chroma not available")
            return
        _chroma_client.delete_collection(name=COL_EPISODIC)
        kwargs = {"embedding_function": _embed_fn} if _embed_fn else {}
        _collections[COL_EPISODIC] = _chroma_client.get_or_create_collection(
            name=COL_EPISODIC, **kwargs
        )
        print(f"[episodic] cleared {count} memories")
    except Exception as e:
        print(f"[episodic] clear error: {e}")


def cmd_model_show():
    active = pcfg("llm.active_model", "?")
    cfg    = get_active_model_cfg()
    print(f"[model] active={active}  model={cfg.get('model')}  url={cfg.get('base_url')}")


def cmd_model_use(name: str):
    models = pcfg("llm.models", {})
    if name not in models:
        print(f"[model] unknown model '{name}'. Available: {list(models.keys())}")
        return
    _profile["llm"]["active_model"] = name
    build_llm_client()
    print(f"[model] switched to {name}")


def cmd_model_toggle():
    models = list(pcfg("llm.models", {}).keys())
    if len(models) < 2:
        print("[model] only one model configured")
        return
    active = pcfg("llm.active_model", models[0])
    idx    = models.index(active) if active in models else 0
    next_  = models[(idx + 1) % len(models)]
    cmd_model_use(next_)


def cmd_web_engine(name: str):
    engines = pcfg("web.engines", {})
    if name not in engines:
        print(f"[web] unknown engine '{name}'. Available: {list(engines.keys())}")
        return
    _profile["web"]["active_engine"] = name
    print(f"[web] engine set to {name}")


def cmd_web_toggle():
    global _web_enabled
    _web_enabled = not _web_enabled
    print(f"[web] web search {'enabled' if _web_enabled else 'disabled'}")


def cmd_run_whitelist_show():
    wl = pcfg("run.whitelist", [])
    print(f"[run] whitelist: {wl}")


def cmd_run_whitelist_add(cmd_name: str):
    if "run" not in _profile:
        _profile["run"] = {"whitelist": []}
    if "whitelist" not in _profile["run"]:
        _profile["run"]["whitelist"] = []
    wl = _profile["run"]["whitelist"]
    if cmd_name not in wl:
        wl.append(cmd_name)
        print(f"[run] '{cmd_name}' added to whitelist")
    else:
        print(f"[run] '{cmd_name}' already in whitelist")


def cmd_profile_show():
    print(json.dumps(_profile, indent=2))


def cmd_profile_add(key_path: str, value: str):
    """Set a dot-path key in the profile to a value."""
    parts = key_path.split(".")
    node  = _profile
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    # Try to parse value as JSON (handles numbers, booleans, lists)
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        parsed = value
    node[parts[-1]] = parsed
    print(f"[profile] set {key_path} = {parsed}")
    save_profile()


def cmd_auto_learn_on():
    global _auto_learn_enabled
    _auto_learn_enabled = True
    print("[auto-learn] enabled")


def cmd_auto_learn_off():
    global _auto_learn_enabled
    _auto_learn_enabled = False
    print("[auto-learn] disabled")


def cmd_auto_learn_toggle():
    global _auto_learn_enabled
    _auto_learn_enabled = not _auto_learn_enabled
    print(f"[auto-learn] {'enabled' if _auto_learn_enabled else 'disabled'}")


def cmd_auto_learn_status():
    print(f"[auto-learn] {'enabled' if _auto_learn_enabled else 'disabled'}, pending={len(_pending_facts)}")


def cmd_crawl(url: str):
    """Crawl a URL and ingest results."""
    print(f"[crawl] starting crawl of {url}…")
    pages = crawl_site(url, _profile)
    if not pages:
        print("[crawl] no pages retrieved")
        return
    print(f"[crawl] retrieved {len(pages)} page(s)")
    auto_ingest = pcfg("web.auto_ingest_after_crawl", False)
    if auto_ingest:
        for page_url, text in pages.items():
            if text.strip():
                ingest_text(text, page_url, _profile)
        print(f"[crawl] ingested {len(pages)} page(s) into memory")
    else:
        print("[crawl] auto_ingest_after_crawl=false — use /ingest to ingest manually")


def cmd_mega_torture():
    """Memory stress test: ingest synthetic data and query it."""
    print("[torture] starting mega torture test…")
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Gilligan is a helpful AI assistant with bio-inspired memory.",
        "ChromaDB stores embeddings for semantic search and retrieval.",
        "Episodic memory stores conversation history with decay over time.",
        "The consolidation process promotes old episodes into semantic facts.",
    ]
    for i, text in enumerate(test_texts):
        ingest_text(text, f"torture_test_{i}", _profile)

    print("[torture] ingested test data — querying…")
    results = retrieve_semantic("Gilligan memory architecture", mode="deep")
    print(f"[torture] retrieved {len(results)} results")
    for i, r in enumerate(results[:3], 1):
        print(f"  [{i}] {r[:100]}…")
    print("[torture] done")


def cmd_depth_test():
    """Test depth retrieval across all layers."""
    print("[depth_test] querying all memory layers…")
    query = "test retrieval"
    for layer_name in [COL_L0, COL_L1, COL_L2, COL_L3, COL_EPISODIC]:
        results = _query_col(layer_name, query, 3)
        print(f"  {layer_name}: {len(results)} result(s)")
    print("[depth_test] done")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 18 — Startup Sequence
# ═══════════════════════════════════════════════════════════════════════════════

def startup():
    """Full startup sequence."""
    global _auto_learn_enabled

    print(f"\n╔══════════════════════════════════════════════╗")
    print(f"║  {ASSISTANT_NAME}  •  {AGENT_VERSION:<30}║")
    print(f"╚══════════════════════════════════════════════╝\n")

    # 1. Load profile
    load_profile()
    _auto_learn_enabled = mcfg("auto_learn_enabled", True)

    # 2. Load blocklist
    load_blocklist()

    # 3. Ensure BASE_DIR and notes file
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_notes_file()

    # 4. Init ChromaDB
    init_chroma()

    # 5. Build LLM client
    build_llm_client()

    # Warn if github provider is active but its env var is unset
    active_cfg = get_active_model_cfg()
    if active_cfg.get("provider") == "github":
        api_key_env = active_cfg.get("api_key_env", "GITHUB_TOKEN")
        if not os.environ.get(api_key_env, ""):
            print(f"[warn] active model uses provider=github but ${api_key_env} is not set — LLM calls will fail")

    # 6. Episodic decay
    pruned = 0
    if HAS_CHROMA:
        pruned = run_episodic_decay(_profile)

    # 7. Consolidation
    consolidated = 0
    if HAS_CHROMA and mcfg("consolidation_on_startup", True):
        model_cfg = get_active_model_cfg()
        consolidated = run_consolidation(_llm_client, model_cfg, _profile)

    # 8. Status
    ep_count = episodic_count()
    print(f"[memory] episodic={ep_count}  pruned={pruned}  consolidated={consolidated}")
    print(f"[ready] Type your question or /help for commands.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 19 — Main Loop
# ═══════════════════════════════════════════════════════════════════════════════

def dispatch_command(raw: str) -> bool:
    """
    Parse and dispatch a / command.
    Returns True to continue loop, False to quit.
    """
    parts = raw.strip().split(None, 2)
    cmd   = parts[0].lower()
    arg1  = parts[1] if len(parts) > 1 else ""
    arg2  = parts[2] if len(parts) > 2 else ""

    if cmd == "/help":
        print_help()
    elif cmd == "/quit":
        print(f"[{ASSISTANT_NAME}] Goodbye.")
        return False
    elif cmd == "/doctor":
        cmd_doctor()
    elif cmd == "/ingest":
        if arg1:
            cmd_ingest(arg1)
        else:
            print("[ingest] usage: /ingest <path>")
    elif cmd == "/reingest":
        cmd_reingest(arg1)
    elif cmd == "/rebuild":
        cmd_rebuild()
    elif cmd == "/ingest_notes":
        cmd_ingest_notes()
    elif cmd == "/consolidate":
        cmd_consolidate()
    elif cmd == "/memory_status":
        cmd_memory_status()
    elif cmd == "/memory_clear_episodic":
        cmd_memory_clear_episodic()
    elif cmd == "/cache_clear":
        cache_clear()
        print("[cache] cleared")
    elif cmd == "/history_clear":
        working_memory_clear()
        print("[history] cleared")
    elif cmd == "/review_pending":
        cmd_review_pending()
    elif cmd == "/pending_count":
        cmd_pending_count()
    elif cmd == "/pending_purge":
        cmd_pending_purge()
    elif cmd == "/auto_learn_on":
        cmd_auto_learn_on()
    elif cmd == "/auto_learn_off":
        cmd_auto_learn_off()
    elif cmd == "/auto_learn_toggle":
        cmd_auto_learn_toggle()
    elif cmd == "/auto_learn_status":
        cmd_auto_learn_status()
    elif cmd == "/model":
        cmd_model_show()
    elif cmd == "/model_use":
        if arg1:
            cmd_model_use(arg1)
        else:
            print("[model] usage: /model_use <name>")
    elif cmd == "/model_toggle":
        cmd_model_toggle()
    elif cmd == "/web":
        web_parts = raw.split(None, 1)
        if len(web_parts) > 1:
            query = web_parts[1]
            if _web_enabled:
                open_web_search(query)
            else:
                print("[web] web search is disabled. Use /web_toggle to enable.")
        else:
            print("[web] usage: /web <search query>")
    elif cmd == "/web_engine":
        if arg1:
            cmd_web_engine(arg1)
        else:
            engines = list(pcfg("web.engines", {}).keys())
            print(f"[web] usage: /web_engine <name>  available: {engines}")
    elif cmd == "/web_toggle":
        cmd_web_toggle()
    elif cmd == "/crawl":
        if arg1:
            cmd_crawl(arg1)
        else:
            print("[crawl] usage: /crawl <url>")
    elif cmd == "/run":
        run_parts = raw.split(None, 1)
        run_cmd = run_parts[1] if len(run_parts) > 1 else ""
        if run_cmd:
            out = run_sandbox(run_cmd)
            print(out)
        else:
            print("[run] usage: /run <command>")
    elif cmd == "/run_whitelist_show":
        cmd_run_whitelist_show()
    elif cmd == "/run_whitelist_add":
        if arg1:
            cmd_run_whitelist_add(arg1)
        else:
            print("[run] usage: /run_whitelist_add <command>")
    elif cmd == "/profile_show":
        cmd_profile_show()
    elif cmd == "/profile_add":
        if arg1 and arg2:
            cmd_profile_add(arg1, arg2)
        else:
            print("[profile] usage: /profile_add <dot.key.path> <value>")
    elif cmd == "/mega_torture":
        cmd_mega_torture()
    elif cmd == "/depth_test":
        cmd_depth_test()
    else:
        print(f"[?] Unknown command: {cmd}  (type /help for list)")

    return True


def main_loop():
    """Main interactive REPL."""
    while True:
        try:
            raw = input(f"{ASSISTANT_NAME}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n[{ASSISTANT_NAME}] Goodbye.")
            break

        if not raw:
            continue

        # Wake-word detection (optional — strip if present)
        lower = raw.lower()
        for wake in WAKE_WORDS:
            if lower.startswith(wake + " "):
                raw = raw[len(wake):].strip()
                break
            elif lower == wake:
                raw = ""
                break

        if not raw:
            continue

        # Command dispatch
        if raw.startswith("/"):
            if not dispatch_command(raw):
                break
            continue

        # Detect query mode prefix
        mode = "auto"
        if raw.upper().startswith("FAST:"):
            mode = "fast"
            raw  = raw[5:].strip()
        elif raw.upper().startswith("DEEP:"):
            mode = "deep"
            raw  = raw[5:].strip()
        elif raw.upper().startswith("QUOTEPLUS:"):
            mode = "deep"
            raw  = raw[len("QUOTEPLUS:"):].strip()

        if not raw:
            continue

        # Answer
        print()
        resp = answer(raw, mode=mode)
        print(f"{ASSISTANT_NAME}: {resp}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    startup()
    main_loop()
