"""
Juris AI — Evaluation Suite Shared Fixtures
Provides async-ready fixtures for backend components and HTTP client.
"""

import sys
import os
import json
import asyncio
from pathlib import Path

import pytest
import pytest_asyncio
import httpx

# ── Make sure the backend package is importable ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    SQLITE_DB_PATH,
    OLLAMA_BASE_URL,
    LLM_MODEL,
    EMBEDDING_MODEL,
    ensure_directories,
)

# ── Constants ────────────────────────────────────────────────────────────────
BASE_URL = os.getenv("JURIS_BASE_URL", "http://localhost:8000")
EVAL_RESULTS_DIR = PROJECT_ROOT / "evaluation_results"
EVAL_CHARTS_DIR = PROJECT_ROOT / "evaluation_charts"
TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"


def pytest_configure(config):
    """Create output directories before tests run."""
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_directories()


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_test_data(filename: str):
    """Load a JSON file from the test_data directory."""
    path = TEST_DATA_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_result(filename: str, data):
    """Save evaluation results to a JSON file."""
    path = EVAL_RESULTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for all async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def base_url():
    return BASE_URL


@pytest_asyncio.fixture(scope="session")
async def http_client():
    """Async HTTP client for calling the running backend."""
    async with httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=10.0),
    ) as client:
        yield client


@pytest_asyncio.fixture(scope="session")
async def retriever():
    """Initialize the LegalRetriever for direct RAG testing."""
    import chromadb
    from app.rag.retriever import LegalRetriever

    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    ret = LegalRetriever(collection=collection)
    return ret


@pytest_asyncio.fixture(scope="session")
async def crm_tool():
    """Initialize the CRM tool for direct testing."""
    from app.tools.crm_tool import CRMTool

    crm = CRMTool(db_path=SQLITE_DB_PATH)
    await crm.initialize_schema()
    return crm


@pytest_asyncio.fixture(scope="session")
async def statute_tool():
    """Initialize the statute lookup tool."""
    from app.tools.statute_lookup_tool import StatuteLookupTool

    tool = StatuteLookupTool(db_path=SQLITE_DB_PATH)
    await tool.initialize_schema()
    return tool


@pytest_asyncio.fixture(scope="session")
async def deadline_tool():
    """Initialize the deadline calculator tool."""
    from app.tools.deadline_calc_tool import DeadlineCalcTool
    return DeadlineCalcTool()


@pytest_asyncio.fixture(scope="session")
async def case_search_tool():
    """Initialize the case search tool."""
    import chromadb
    from app.tools.case_search_tool import CaseSearchTool

    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return CaseSearchTool(chroma_collection=collection)


@pytest_asyncio.fixture(scope="session")
async def tool_orchestrator():
    """Initialize the tool orchestrator with all tools registered."""
    from app.tools.tool_registry import register_all_tools
    from app.tools.orchestrator import ToolOrchestrator
    import chromadb

    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    registry = register_all_tools(
        db_path=SQLITE_DB_PATH,
        chroma_collection=collection,
    )
    return ToolOrchestrator(registry)


@pytest.fixture(scope="session")
def hw_info():
    """Collect hardware information for the report."""
    import platform
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "ollama_url": OLLAMA_BASE_URL,
    }
    try:
        import psutil
        info["cpu_count"] = psutil.cpu_count(logical=True)
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        import multiprocessing
        info["cpu_count"] = multiprocessing.cpu_count()
        info["ram_gb"] = "N/A (psutil not installed)"
    return info
