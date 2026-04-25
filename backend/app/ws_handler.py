"""
Juris AI — WebSocket Connection Handler
Manages individual WebSocket connections and message routing.
"""

import json
from typing import Optional, Dict, Any

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from app.conversation_manager import ConversationManager
from app.rag.retriever import LegalRetriever
from app.tools.orchestrator import ToolOrchestrator
from app.tools.crm_tool import CRMTool
from app.llm.engine import LLMEngine


class ConnectionManager:
    """
    Manages all active WebSocket connections.
    
    Tracks active sessions and provides broadcast capabilities.
    """

    def __init__(self):
        """Initialize the connection manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.conversation_managers: Dict[str, ConversationManager] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """
        Accept a WebSocket connection and register it.
        
        Args:
            websocket: The WebSocket connection.
            session_id: The unique session ID.
        """
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info("WebSocket connected | session={}", session_id)

    def disconnect(self, session_id: str) -> None:
        """
        Remove a WebSocket connection.
        
        Args:
            session_id: The session ID to disconnect.
        """
        self.active_connections.pop(session_id, None)
        self.conversation_managers.pop(session_id, None)
        logger.info("WebSocket disconnected | session={}", session_id)

    async def send_message(self, session_id: str, message: dict) -> None:
        """
        Send a JSON message to a specific session.
        
        Args:
            session_id: Target session ID.
            message: Dict to serialize and send as JSON.
        """
        ws = self.active_connections.get(session_id)
        if ws:
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.error("Failed to send message to {}: {}", session_id, e)

    def get_active_count(self) -> int:
        """Return the number of active connections."""
        return len(self.active_connections)

    async def close_all(self) -> None:
        """Gracefully close all active WebSocket connections."""
        for session_id, ws in list(self.active_connections.items()):
            try:
                await ws.close()
            except Exception:
                pass
        self.active_connections.clear()
        self.conversation_managers.clear()
        logger.info("All WebSocket connections closed")


# Global connection manager instance
manager = ConnectionManager()


async def handle_websocket(
    websocket: WebSocket,
    session_id: str,
    retriever: LegalRetriever,
    tool_orchestrator: ToolOrchestrator,
    llm_engine: LLMEngine,
    crm: CRMTool,
) -> None:
    """
    Handle a single WebSocket connection lifecycle.
    
    Manages the full message loop: receive messages, route them,
    and send responses according to the WebSocket protocol spec.
    
    Args:
        websocket: The WebSocket connection.
        session_id: The unique session ID from the URL path.
        retriever: The RAG retriever instance.
        tool_orchestrator: The tool orchestrator instance.
        llm_engine: The LLM engine instance.
        crm: The CRM tool instance.
    """
    await manager.connect(websocket, session_id)

    # Create conversation manager for this session (initially without client)
    conv_manager = ConversationManager(
        session_id=session_id,
        retriever=retriever,
        tool_orchestrator=tool_orchestrator,
        llm_engine=llm_engine,
        crm=crm,
    )
    manager.conversation_managers[session_id] = conv_manager

    try:
        while True:
            # Receive message
            try:
                raw_data = await websocket.receive_text()
            except WebSocketDisconnect:
                break

            # Parse message
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                await manager.send_message(session_id, {
                    "type": "error",
                    "message": "Invalid JSON message format."
                })
                logger.warning("Invalid JSON from session {}", session_id)
                continue

            msg_type = data.get("type", "")
            logger.info("WS received | session={} | type={}", session_id, msg_type)

            # Route by message type
            if msg_type == "session_init":
                await _handle_session_init(session_id, data, conv_manager, crm)

            elif msg_type == "user_message":
                await _handle_user_message(session_id, data, conv_manager)

            elif msg_type == "ping":
                await manager.send_message(session_id, {"type": "pong"})

            else:
                await manager.send_message(session_id, {
                    "type": "error",
                    "message": f"Unknown message type: '{msg_type}'"
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket error | session={} | error={}", session_id, str(e))
        try:
            await manager.send_message(session_id, {
                "type": "error",
                "message": f"Server error: {str(e)}"
            })
        except Exception:
            pass
    finally:
        manager.disconnect(session_id)


async def _handle_session_init(
    session_id: str,
    data: dict,
    conv_manager: ConversationManager,
    crm: CRMTool,
) -> None:
    """Handle session initialization with optional client loading."""
    client_id = data.get("client_id")
    client_loaded = False
    history_data = []

    if client_id:
        conv_manager.client_id = client_id
        client_loaded = await conv_manager.initialize()
        
        # Format history to send to client
        for msg in conv_manager.history:
            history_data.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "metadata": msg.metadata
            })

    await manager.send_message(session_id, {
        "type": "session_ready",
        "session_id": session_id,
        "client_loaded": client_loaded,
        "history": history_data,
    })

    logger.info(
        "Session initialized | session={} | client_loaded={} | history_size={}",
        session_id, client_loaded, len(history_data)
    )


async def _handle_user_message(
    session_id: str,
    data: dict,
    conv_manager: ConversationManager,
) -> None:
    """Handle a user message by processing it through the conversation manager."""
    content = data.get("content", "").strip()

    if not content:
        await manager.send_message(session_id, {
            "type": "error",
            "message": "Empty message. Please type a question."
        })
        return

    # Process the message through the conversation pipeline
    try:
        async for event in conv_manager.process_message(content):
            await manager.send_message(session_id, event)
    except Exception as e:
        logger.error(
            "Message processing failed | session={} | error={}",
            session_id, str(e)
        )
        await manager.send_message(session_id, {
            "type": "error",
            "message": f"Failed to process message: {str(e)}"
        })
