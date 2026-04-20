"""
Juris AI — Pydantic Models & Data Schemas
All request/response models, tool schemas, and internal data structures.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket Message Models
# ═══════════════════════════════════════════════════════════════════════════════

class WSSessionInit(BaseModel):
    """Client → Server: Initialize a session, optionally with a CRM client ID."""
    type: str = "session_init"
    client_id: Optional[str] = None


class WSUserMessage(BaseModel):
    """Client → Server: User sends a message."""
    type: str = "user_message"
    content: str


class WSPing(BaseModel):
    """Client → Server: Heartbeat ping."""
    type: str = "ping"


class WSSessionReady(BaseModel):
    """Server → Client: Session initialized and ready."""
    type: str = "session_ready"
    session_id: str
    client_loaded: bool = False


class WSToken(BaseModel):
    """Server → Client: A single streamed token from the LLM."""
    type: str = "token"
    content: str


class WSToolInvoked(BaseModel):
    """Server → Client: A tool has been invoked."""
    type: str = "tool_invoked"
    tool_name: str
    status: str = "running"


class WSToolResult(BaseModel):
    """Server → Client: Result from a tool invocation."""
    type: str = "tool_result"
    tool_name: str
    summary: str


class WSRagRetrieved(BaseModel):
    """Server → Client: RAG retrieval completed."""
    type: str = "rag_retrieved"
    count: int
    citations: List[str]


class WSDone(BaseModel):
    """Server → Client: Generation complete with metadata."""
    type: str = "done"
    citations: List[str] = Field(default_factory=list)
    rag_used: bool = False
    tools_used: List[str] = Field(default_factory=list)


class WSError(BaseModel):
    """Server → Client: Error occurred."""
    type: str = "error"
    message: str


class WSPong(BaseModel):
    """Server → Client: Heartbeat pong."""
    type: str = "pong"


# ═══════════════════════════════════════════════════════════════════════════════
# RAG Data Models
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievedChunk:
    """A single chunk retrieved from the vector store."""
    text: str
    source: str           # "ayesha_jadoon_dataset" | "ppc_markdown" | "sc_judgments"
    act: str              # e.g. "Pakistan Penal Code"
    section: str          # e.g. "302" or "" if judgment
    doc_type: str         # "statute" | "judgment"
    distance: float       # cosine distance (lower = more relevant)
    citation: str         # formatted citation string, e.g. "PPC §302"

    def to_context_string(self) -> str:
        """Format this chunk for injection into the LLM prompt."""
        header = f"[{self.citation}] ({self.doc_type})"
        return f"{header}\n{self.text}"


@dataclass
class DocumentChunk:
    """A document chunk ready for indexing into the vector store."""
    text: str
    metadata: Dict[str, str]
    chunk_hash: str       # SHA-256 hash of text for deduplication


# ═══════════════════════════════════════════════════════════════════════════════
# Conversation Models
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Message:
    """A single message in a conversation."""
    role: str             # "user" | "assistant" | "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Models
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ToolResult:
    """Result from a tool invocation."""
    success: bool
    data: Any
    formatted_text: str    # Human-readable result to inject into prompt
    error_message: str = ""


@dataclass
class ToolCall:
    """A parsed tool call from LLM output."""
    tool_name: str
    arguments: Dict[str, Any]


# ═══════════════════════════════════════════════════════════════════════════════
# CRM Models
# ═══════════════════════════════════════════════════════════════════════════════

class ClientCreate(BaseModel):
    """Request body for creating a new CRM client."""
    name: str
    cnic: Optional[str] = None
    contact: Optional[str] = None
    case_type: Optional[str] = None
    charges: Optional[str] = None
    bail_status: Optional[str] = "unknown"
    court_name: Optional[str] = None
    next_hearing_date: Optional[str] = None
    notes: Optional[str] = None


class ClientResponse(BaseModel):
    """Response body for a CRM client."""
    client_id: str
    name: str
    cnic: Optional[str] = None
    contact: Optional[str] = None
    case_type: Optional[str] = None
    charges: Optional[str] = None
    bail_status: Optional[str] = None
    court_name: Optional[str] = None
    next_hearing_date: Optional[str] = None
    notes: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# API Response Models
# ═══════════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    """Response for GET /health endpoint."""
    status: str
    model_loaded: bool
    index_size: int
    uptime_seconds: float


class IndexStats(BaseModel):
    """Response for GET /api/index/stats."""
    collection_name: str
    total_documents: int
    statutes_count: int = 0
    judgments_count: int = 0


class MetricsResponse(BaseModel):
    """Response for GET /metrics endpoint."""
    uptime_seconds: float
    total_queries: int
    rag_hits: int
    rag_misses: int
    tool_invocations: Dict[str, int]
    avg_retrieval_latency_ms: float
    avg_generation_latency_ms: float
    hallucination_guard_triggered: int


# ═══════════════════════════════════════════════════════════════════════════════
# Deadline Models
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DeadlineResult:
    """A single computed legal deadline."""
    description: str
    due_date: str         # YYYY-MM-DD
    days_from_trigger: int
    authority: str        # e.g. "CrPC Section 497"
    note: str = ""

    def to_formatted_string(self) -> str:
        """Format this deadline for display."""
        result = f"{self.description}: Due by {self.due_date} ({self.days_from_trigger} days from trigger) — Authority: {self.authority}"
        if self.note:
            result += f"\n  Note: {self.note}"
        return result
