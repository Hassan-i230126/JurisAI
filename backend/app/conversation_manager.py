"""
Juris AI — Conversation Manager
Central orchestration brain: routes queries through RAG, tools, and LLM.
"""

import json
import random
import time
from typing import AsyncGenerator, List, Optional, Dict, Any

from loguru import logger

from app.config import MAX_HISTORY_TURNS
from app.models.schemas import Message, RetrievedChunk
from app.rag.retriever import LegalRetriever
from app.tools.orchestrator import ToolOrchestrator
from app.tools.crm_tool import CRMTool
from app.llm.engine import LLMEngine
from app.llm.prompts import (
    SYSTEM_PROMPT,
    FEW_SHOT_TOOL_EXAMPLES,
    UNCERTAINTY_INSTRUCTION,
    GREETING_RESPONSES,
    is_legal_question,
    is_greeting,
)


class ConversationManager:
    """
    The central orchestration engine for each user session.
    
    Manages the per-turn flow: receive message → classify → retrieve →
    detect tools → construct prompt → stream response → save history.
    """

    def __init__(
        self,
        session_id: str,
        retriever: LegalRetriever,
        tool_orchestrator: ToolOrchestrator,
        llm_engine: LLMEngine,
        crm: CRMTool,
        client_id: Optional[str] = None,
    ):
        """
        Initialize a conversation manager for a session.
        
        Args:
            session_id: Unique session identifier.
            retriever: The RAG retriever instance.
            tool_orchestrator: The tool orchestrator instance.
            llm_engine: The LLM engine instance.
            crm: The CRM tool instance.
            client_id: Optional CRM client ID to bind to this session.
        """
        self.session_id = session_id
        self.client_id = client_id
        self.history: List[Message] = []
        self.retriever = retriever
        self.tool_orchestrator = tool_orchestrator
        self.llm_engine = llm_engine
        self.crm = crm
        self._client_context: Optional[str] = None
        self._tools_used_in_turn: List[str] = []
        self._rag_used_in_turn = False
        self._citations_in_turn: List[str] = []

    async def initialize(self) -> bool:
        """
        Initialize the session, loading client profile if specified.
        
        Returns:
            True if a client was successfully loaded.
        """
        if self.client_id:
            context = await self.crm.get_client_context(self.client_id)
            if context:
                self._client_context = context
                logger.info(
                    "Session {} initialized with client {}",
                    self.session_id, self.client_id
                )
                return True
            else:
                logger.warning(
                    "Session {} — client {} not found",
                    self.session_id, self.client_id
                )
        return False

    async def process_message(
        self,
        user_message: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a user message through the full per-turn pipeline.
        
        Yields WebSocket-ready message dicts as events occur:
        - rag_retrieved: RAG results found
        - tool_invoked/tool_result: Tool usage
        - token: Streamed LLM tokens
        - done: Generation complete with metadata
        
        Args:
            user_message: The raw user message text.
            
        Yields:
            Dicts ready to be serialized and sent via WebSocket.
        """
        start_time = time.time()
        self._tools_used_in_turn = []
        self._rag_used_in_turn = False
        self._citations_in_turn = []

        # Step 1: Save user message to history
        self.history.append(Message(role="user", content=user_message))

        # Step 2: Routing decision
        if is_greeting(user_message) and not is_legal_question(user_message):
            # Greeting/chitchat — respond directly without RAG or tools
            greeting = random.choice(GREETING_RESPONSES)
            self.history.append(Message(role="assistant", content=greeting))

            for token in greeting:
                yield {"type": "token", "content": token}

            yield {
                "type": "done",
                "citations": [],
                "rag_used": False,
                "tools_used": [],
            }
            return

        # Step 3: RAG Retrieval
        rag_found_nothing = False
        retrieved_chunks: List[RetrievedChunk] = []

        if is_legal_question(user_message):
            try:
                retrieved_chunks = await self.retriever.retrieve(
                    user_message, top_k=3
                )
                if retrieved_chunks:
                    self._rag_used_in_turn = True
                    self._citations_in_turn = [c.citation for c in retrieved_chunks]

                    yield {
                        "type": "rag_retrieved",
                        "count": len(retrieved_chunks),
                        "citations": self._citations_in_turn,
                    }
                else:
                    rag_found_nothing = True
            except Exception as e:
                logger.error("RAG retrieval failed: {}", str(e))
                rag_found_nothing = True

        # Step 4: Tool Detection
        tool_result_text: Optional[str] = None

        tool_call = self.tool_orchestrator.detect_tool_from_message(user_message)
        if tool_call:
            yield {
                "type": "tool_invoked",
                "tool_name": tool_call.tool_name,
                "status": "running",
            }

            tool_result = await self.tool_orchestrator.execute_tool(tool_call)
            self._tools_used_in_turn.append(tool_call.tool_name)

            yield {
                "type": "tool_result",
                "tool_name": tool_call.tool_name,
                "summary": tool_result.formatted_text[:200],
            }

            if tool_result.success:
                tool_result_text = tool_result.formatted_text

        # Step 5: Prompt Construction
        messages = self._build_prompt(
            user_message=user_message,
            retrieved_chunks=retrieved_chunks,
            tool_result_text=tool_result_text,
            rag_found_nothing=rag_found_nothing,
        )

        # Step 6: Stream Response
        full_response = []

        async for token in self.llm_engine.generate_stream(messages):
            full_response.append(token)
            yield {"type": "token", "content": token}

        response_text = "".join(full_response)

        # Check for tool calls in LLM output
        llm_tool_call = self.tool_orchestrator.parse_tool_call_from_llm(response_text)
        if llm_tool_call and llm_tool_call.tool_name not in self._tools_used_in_turn:
            # Execute the LLM-requested tool
            yield {
                "type": "tool_invoked",
                "tool_name": llm_tool_call.tool_name,
                "status": "running",
            }

            tool_result = await self.tool_orchestrator.execute_tool(llm_tool_call)
            self._tools_used_in_turn.append(llm_tool_call.tool_name)

            yield {
                "type": "tool_result",
                "tool_name": llm_tool_call.tool_name,
                "summary": tool_result.formatted_text[:200],
            }

            if tool_result.success:
                # Re-generate with tool result
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"[TOOL RESULT]\n{tool_result.formatted_text}"
                })

                full_response = []
                async for token in self.llm_engine.generate_stream(messages):
                    full_response.append(token)
                    yield {"type": "token", "content": token}

                response_text = "".join(full_response)

        # Step 7: Apply hallucination guard
        response_text = self.llm_engine.apply_hallucination_guard(
            response_text, rag_found_nothing
        )

        # Step 8: Save assistant response to history
        self.history.append(Message(
            role="assistant",
            content=response_text,
            metadata={
                "rag_used": self._rag_used_in_turn,
                "tools_used": self._tools_used_in_turn,
                "citations": self._citations_in_turn,
            }
        ))

        # Step 9: Trim history to MAX_HISTORY_TURNS pairs
        self._trim_history()

        # Log interaction if client is loaded
        if self.client_id:
            try:
                summary = f"Query: {user_message[:100]}..."
                await self.crm._log_interaction(
                    self.client_id, self.session_id, summary
                )
            except Exception:
                pass  # Non-critical

        latency_ms = (time.time() - start_time) * 1000
        logger.info(
            "Turn complete | session={} | rag={} | tools={} | latency_ms={:.0f}",
            self.session_id, self._rag_used_in_turn,
            self._tools_used_in_turn, latency_ms
        )

        # Step 10: Send completion signal
        yield {
            "type": "done",
            "citations": self._citations_in_turn,
            "rag_used": self._rag_used_in_turn,
            "tools_used": self._tools_used_in_turn,
        }

    def _build_prompt(
        self,
        user_message: str,
        retrieved_chunks: List[RetrievedChunk],
        tool_result_text: Optional[str],
        rag_found_nothing: bool,
    ) -> List[Dict[str, str]]:
        """
        Construct the full prompt message list for the LLM.
        
        Order:
        1. System prompt (always)
        2. Client context (if loaded)
        3. Legal context (if RAG found chunks)
        4. Tool result (if tool was invoked)
        5. Conversation history (last MAX_HISTORY_TURNS turns)
        6. User message
        7. Uncertainty instruction (if RAG found nothing and no tool result)
        """
        messages = []

        # 1. System prompt
        system_content = SYSTEM_PROMPT

        # Add few-shot tool examples if tools have been used in this session
        if self._tools_used_in_turn:
            system_content += "\n\n" + FEW_SHOT_TOOL_EXAMPLES

        # 2. Client context
        if self._client_context:
            system_content += f"\n\n{self._client_context}"

        # 3. Legal context from RAG
        if retrieved_chunks:
            context_parts = []
            for chunk in retrieved_chunks:
                # Truncate each chunk to 500 characters
                truncated = chunk.text[:500]
                context_parts.append(chunk.to_context_string()[:600])

            legal_context = "\n\n".join(context_parts)
            system_content += f"\n\n[LEGAL CONTEXT]\n{legal_context}"

        # 4. Tool result
        if tool_result_text:
            system_content += f"\n\n[TOOL RESULT]\n{tool_result_text}"

        # 7. Uncertainty instruction
        if rag_found_nothing and not tool_result_text:
            system_content += f"\n\n{UNCERTAINTY_INSTRUCTION}"

        messages.append({"role": "system", "content": system_content})

        # 5. Conversation history (last MAX_HISTORY_TURNS * 2 messages)
        max_msgs = MAX_HISTORY_TURNS * 2
        history_msgs = self.history[:-1]  # Exclude the current user message
        if len(history_msgs) > max_msgs:
            history_msgs = history_msgs[-max_msgs:]

        for msg in history_msgs:
            messages.append({"role": msg.role, "content": msg.content})

        # 6. Current user message
        messages.append({"role": "user", "content": user_message})

        return messages

    def _trim_history(self) -> None:
        """
        Trim conversation history to MAX_HISTORY_TURNS pairs.
        
        On overflow, drops the oldest user+assistant pair.
        """
        max_messages = MAX_HISTORY_TURNS * 2  # Each turn = 1 user + 1 assistant
        if len(self.history) > max_messages:
            overflow = len(self.history) - max_messages
            self.history = self.history[overflow:]
