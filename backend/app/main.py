"""
Juris AI — FastAPI Application
Main entry point: routes, lifespan, static file serving.
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import chromadb
from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from app.config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    SQLITE_DB_PATH,
    HOST,
    PORT,
    ensure_directories,
)
from app.utils.logger import setup_logger
from app.llm.engine import LLMEngine
from app.conversation_manager import ConversationManager
from app.rag.retriever import LegalRetriever
from app.tools.tool_registry import register_all_tools
from app.tools.orchestrator import ToolOrchestrator
from app.tools.crm_tool import CRMTool
from app.tools.statute_lookup_tool import StatuteLookupTool
from app.ws_handler import handle_websocket, manager as ws_manager
from app.models.schemas import (
    HealthResponse,
    IndexStats,
    MetricsResponse,
    ClientCreate,
    ClientResponse,
)


# ─── Global State ─────────────────────────────────────────────────────────────
_start_time: float = 0.0
_llm_engine: Optional[LLMEngine] = None
_retriever: Optional[LegalRetriever] = None
_tool_orchestrator: Optional[ToolOrchestrator] = None
_crm: Optional[CRMTool] = None
_chroma_collection = None
_http_conversation_managers = {}
_http_session_locks = {}
_metrics = {
    "total_queries": 0,
    "rag_hits": 0,
    "rag_misses": 0,
    "tool_invocations": {
        "crm_tool": 0,
        "statute_lookup": 0,
        "deadline_calculator": 0,
        "case_search": 0,
    },
    "retrieval_latencies": [],
    "generation_latencies": [],
    "hallucination_guard_triggered": 0,
}


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan: startup and shutdown logic.
    
    Startup:
    1. Initialize logger
    2. Check Ollama connectivity
    3. Check ChromaDB collection
    4. Initialize SQLite schema
    5. Register tools
    6. Mount static files
    7. Log ready message
    
    Shutdown:
    1. Close WebSocket connections
    2. Log shutdown
    """
    global _start_time, _llm_engine, _retriever, _tool_orchestrator
    global _crm, _chroma_collection
    global _http_conversation_managers, _http_session_locks

    # ── Startup ───────────────────────────────────────────────────────────
    _start_time = time.time()

    # Mute Uvicorn's health check logs
    import logging
    class HealthCheckFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return record.getMessage().find("/api/health") == -1 and record.getMessage().find("/health") == -1
            
    logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

    # 1. Initialize logger
    setup_logger()

    # 1. Initialize logger
    setup_logger()
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║        Juris AI — Starting Up                ║")
    logger.info("╚══════════════════════════════════════════════╝")

    # Ensure data directories exist
    ensure_directories()

    # 2. Check Ollama connectivity and LLM model
    _llm_engine = LLMEngine()
    model_available = await _llm_engine.check_availability()
    if not model_available:
        logger.critical(
            "LLM model not available! Run: ollama pull phi4-mini && ollama pull bge-m3"
        )
    else:
        logger.info("✓ Ollama connected and LLM model available")

    # 3. Check ChromaDB collection
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _chroma_collection = chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        collection_count = _chroma_collection.count()

        if collection_count < 100:
            logger.warning(
                "ChromaDB collection '{}' has only {} entries. "
                "Run: python scripts/run_indexer.py",
                CHROMA_COLLECTION_NAME, collection_count
            )
        else:
            logger.info(
                "✓ ChromaDB collection '{}' loaded with {} entries",
                CHROMA_COLLECTION_NAME, collection_count
            )
    except Exception as e:
        logger.error("ChromaDB initialization failed: {}", e)
        _chroma_collection = None

    # 4. Initialize SQLite schema (CRM + Statutes)
    _crm = CRMTool(db_path=SQLITE_DB_PATH)
    await _crm.initialize_schema()

    statute_tool = StatuteLookupTool(db_path=SQLITE_DB_PATH)
    await statute_tool.initialize_schema()
    logger.info("✓ SQLite schema initialized")

    # 5. Register all tools
    registry = register_all_tools(
        db_path=SQLITE_DB_PATH,
        chroma_collection=_chroma_collection,
    )
    _tool_orchestrator = ToolOrchestrator(registry)
    logger.info("✓ All tools registered")

    # 6. Initialize retriever
    _retriever = LegalRetriever(collection=_chroma_collection)
    logger.info("✓ Legal retriever initialized")

    # 7. Log ready message
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║   Juris AI is ready on http://localhost:{}  ║", PORT)
    logger.info("╚══════════════════════════════════════════════╝")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────

    # 1. Close all active WebSocket connections
    await ws_manager.close_all()
    logger.info("✓ WebSocket connections closed")

    _http_conversation_managers.clear()
    _http_session_locks.clear()

    # 2. Log shutdown
    uptime = time.time() - _start_time
    logger.info(
        "Juris AI shut down cleanly | uptime={:.0f}s | queries={}",
        uptime, _metrics["total_queries"]
    )


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Juris AI",
    description="Pakistani Criminal Law Intelligence System",
    version="1.0.0",
    lifespan=lifespan,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"


# ─── WebSocket Endpoint ──────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat sessions."""
    await handle_websocket(
        websocket=websocket,
        session_id=session_id,
        retriever=_retriever,
        tool_orchestrator=_tool_orchestrator,
        llm_engine=_llm_engine,
        crm=_crm,
    )


class ChatStreamRequest(BaseModel):
    """Request body for HTTP chat streaming."""
    session_id: str
    message: str
    client_id: Optional[str] = None


async def _get_or_create_http_conversation_manager(
    session_id: str,
    client_id: Optional[str],
) -> tuple[ConversationManager, bool]:
    """Get or create per-session conversation manager for HTTP streaming."""
    cache_key = f"{session_id}_{client_id}" if client_id else session_id
    manager = _http_conversation_managers.get(cache_key)

    if manager is None:
        manager = ConversationManager(
            session_id=session_id,
            client_id=client_id,
            retriever=_retriever,
            tool_orchestrator=_tool_orchestrator,
            llm_engine=_llm_engine,
            crm=_crm,
        )
        _http_conversation_managers[cache_key] = manager

    client_loaded = False
    if client_id:
        if manager.client_id != client_id:
            manager.client_id = client_id
        client_loaded = await manager.initialize()

    return manager, client_loaded


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatStreamRequest, request: Request):
    """Stream chat events as NDJSON over HTTP for low TTFT and resilient UI updates."""
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if not _retriever or not _tool_orchestrator or not _llm_engine or not _crm:
        raise HTTPException(status_code=503, detail="Server not fully initialized")

    llm_ready = _llm_engine.is_available
    if not llm_ready:
        llm_ready = await _llm_engine.check_availability()

    if not llm_ready:
        raise HTTPException(
            status_code=503,
            detail="Server Offline: local LLM is unavailable at Ollama.",
        )

    conv_manager, client_loaded = await _get_or_create_http_conversation_manager(
        payload.session_id,
        payload.client_id,
    )

    cache_key = f"{payload.session_id}_{payload.client_id}" if payload.client_id else payload.session_id
    session_lock = _http_session_locks.setdefault(cache_key, asyncio.Lock())
    if session_lock.locked():
        return JSONResponse(
            status_code=409,
            content={"error": "A previous response is still streaming for this session."},
        )

    async def event_stream():
        async with session_lock:
            try:
                history_data = []
                for msg in conv_manager.history:
                    history_data.append({
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                        "metadata": msg.metadata
                    })

                session_ready = {
                    "type": "session_ready",
                    "session_id": payload.session_id,
                    "client_loaded": client_loaded,
                    "history": history_data,
                }
                yield json.dumps(session_ready, ensure_ascii=False) + "\n"

                async for event in conv_manager.process_message(message):
                    if await request.is_disconnected():
                        logger.info("HTTP stream disconnected | session={}", payload.session_id)
                        break
                    yield json.dumps(event, ensure_ascii=False) + "\n"
            except Exception as e:
                logger.error(
                    "HTTP stream failed | session={} | error={}",
                    payload.session_id,
                    str(e),
                )
                error_event = {
                    "type": "error",
                    "message": f"Failed to process message: {str(e)}",
                }
                yield json.dumps(error_event, ensure_ascii=False) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─── REST API Endpoints ──────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint — returns system status."""
    uptime = time.time() - _start_time
    index_size = _chroma_collection.count() if _chroma_collection else 0
    model_loaded = _llm_engine.is_available if _llm_engine else False

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        index_size=index_size,
        uptime_seconds=round(uptime, 1),
    )


@app.get("/api/health", response_model=HealthResponse)
async def api_health_check():
    """Health alias for frontend dev proxy and API namespacing."""
    return await health_check()


@app.get("/metrics")
async def get_metrics():
    """Metrics endpoint — returns operational statistics."""
    uptime = time.time() - _start_time

    retrieval_lats = _metrics["retrieval_latencies"]
    gen_lats = _metrics["generation_latencies"]

    return {
        "uptime_seconds": round(uptime, 1),
        "total_queries": _metrics["total_queries"],
        "rag_hits": _metrics["rag_hits"],
        "rag_misses": _metrics["rag_misses"],
        "tool_invocations": _metrics["tool_invocations"],
        "avg_retrieval_latency_ms": round(
            sum(retrieval_lats) / len(retrieval_lats), 1
        ) if retrieval_lats else 0.0,
        "avg_generation_latency_ms": round(
            sum(gen_lats) / len(gen_lats), 1
        ) if gen_lats else 0.0,
        "hallucination_guard_triggered": _metrics["hallucination_guard_triggered"],
    }


@app.get("/api/index/stats")
async def index_stats():
    """Return ChromaDB collection statistics."""
    if not _chroma_collection:
        return {"error": "ChromaDB not initialized"}

    total = _chroma_collection.count()

    # Count by doc_type
    try:
        statute_result = _chroma_collection.get(
            where={"doc_type": "statute"},
            limit=1,
            include=[],
        )
        statutes = len(statute_result["ids"]) if statute_result else 0
    except Exception:
        statutes = 0

    try:
        judgment_result = _chroma_collection.get(
            where={"doc_type": "judgment"},
            limit=1,
            include=[],
        )
        judgments = len(judgment_result["ids"]) if judgment_result else 0
    except Exception:
        judgments = 0

    return {
        "collection_name": CHROMA_COLLECTION_NAME,
        "total_documents": total,
        "statutes_count": statutes,
        "judgments_count": judgments,
    }


@app.post("/api/index/rebuild")
async def rebuild_index():
    """Trigger re-indexing (admin use)."""
    # This runs synchronously and will block — for admin use only
    try:
        from app.rag.indexer import LegalIndexer
        indexer = LegalIndexer()
        summary = indexer.run_full_pipeline()
        return {"status": "complete", "summary": summary}
    except Exception as e:
        logger.error("Re-indexing failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


# ─── CRM REST Endpoints ──────────────────────────────────────────────────────

@app.get("/api/clients")
async def list_clients():
    """List all CRM clients."""
    if not _crm:
        raise HTTPException(status_code=503, detail="CRM not initialized")
    result = await _crm.run(action="list")
    return {"clients": result.data or []}


@app.post("/api/clients")
async def create_client(client: ClientCreate):
    """Create a new CRM client."""
    if not _crm:
        raise HTTPException(status_code=503, detail="CRM not initialized")

    result = await _crm.run(
        action="create",
        client_id=client.client_id,
        name=client.name,
        cnic=client.cnic,
        contact=client.contact,
        case_type=client.case_type,
        charges=client.charges,
        bail_status=client.bail_status,
        court_name=client.court_name,
        next_hearing_date=client.next_hearing_date,
        notes=client.notes,
    )

    if result.success:
        return {"status": "created", "data": result.data}
    else:
        raise HTTPException(status_code=400, detail=result.error_message)


@app.get("/api/clients/{client_id}")
async def get_client(client_id: str):
    """Get a specific client profile."""
    if not _crm:
        raise HTTPException(status_code=503, detail="CRM not initialized")

    result = await _crm.run(action="get", client_id=client_id)

    if result.success:
        return {"client": result.data}
    else:
        raise HTTPException(status_code=404, detail=result.error_message)


@app.delete("/api/clients/{client_id}")
async def delete_client(client_id: str):
    """Delete a specific client profile."""
    if not _crm:
        raise HTTPException(status_code=503, detail="CRM not initialized")

    result = await _crm.run(action="delete", client_id=client_id)

    if result.success:
        return {"status": "deleted", "client_id": client_id}
    else:
        raise HTTPException(status_code=404, detail=result.error_message)


@app.get("/api/clients/{client_id}/history")
async def get_client_history(client_id: str):
    """
    Return the saved chat history for a client.

    This allows the frontend to immediately display prior conversation
    when a client profile is loaded, before any new message is sent.
    """
    from app.config import CHAT_HISTORY_DIR
    import json as _json
    from pathlib import Path as _Path

    history_path = _Path(CHAT_HISTORY_DIR) / f"{client_id}_history.json"
    if not history_path.exists():
        return {"history": [], "client_id": client_id}

    try:
        data = _json.loads(history_path.read_text(encoding="utf-8"))
        # Only expose role/content/timestamp/metadata — no internals
        history = [
            {
                "role": item.get("role", "user"),
                "content": item.get("content", ""),
                "timestamp": item.get("timestamp"),
                "metadata": item.get("metadata", {}),
            }
            for item in data
            if item.get("role") in ("user", "assistant")
        ]
        return {"history": history, "client_id": client_id}
    except Exception as e:
        logger.error("Failed to read history for client {}: {}", client_id, e)
        raise HTTPException(status_code=500, detail="Failed to read chat history")


app.mount("/", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="frontend")
