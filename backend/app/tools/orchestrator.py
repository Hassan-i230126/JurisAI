"""
Juris AI — Tool Orchestrator
Detects tool calls in LLM output, executes them, and returns results.
"""

import re
import json
import asyncio
from typing import Optional, Tuple

from loguru import logger

from app.models.schemas import ToolResult, ToolCall
from app.tools.tool_registry import ToolRegistry


class ToolOrchestrator:
    """
    Orchestrates tool detection and execution.
    
    Two modes of tool triggering:
    1. Keyword-based: detects tool-relevant patterns in user messages
    2. LLM-generated: parses <tool_call> blocks from LLM output
    """

    def __init__(self, registry: ToolRegistry):
        """
        Initialize the orchestrator with a tool registry.
        
        Args:
            registry: The populated ToolRegistry containing all tools.
        """
        self.registry = registry

    def detect_tool_from_message(self, message: str) -> Optional[ToolCall]:
        """
        Detect if a user message should trigger a tool based on keyword patterns.
        
        This is a lightweight, rule-based classifier that runs before
        the LLM generates a response. It enables proactive tool use.
        
        Args:
            message: The raw user message text.
            
        Returns:
            A ToolCall if a pattern matches, None otherwise.
        """
        msg_lower = message.lower().strip()

        # Statute lookup: "Section 302" or "section 302 PPC" patterns
        section_match = re.search(
            r'section\s+(\d+[\w-]*)\s*(?:of\s+(?:the\s+)?)?'
            r'(ppc|crpc|qso|ata|pakistan\s+penal\s+code|'
            r'code\s+of\s+criminal\s+procedure|'
            r'anti[- ]?terrorism\s+act)?',
            msg_lower
        )
        if section_match:
            section_num = section_match.group(1)
            act_raw = section_match.group(2) or ""
            act_map = {
                "ppc": "PPC",
                "pakistan penal code": "PPC",
                "crpc": "CrPC",
                "code of criminal procedure": "CrPC",
                "qso": "QSO",
                "ata": "ATA",
                "anti-terrorism act": "ATA",
                "anti terrorism act": "ATA",
                "antiterrorism act": "ATA",
            }
            act = act_map.get(act_raw.strip(), "PPC")
            return ToolCall(
                tool_name="statute_lookup",
                arguments={"act": act, "section_number": section_num}
            )

        # Deadline calculator: keywords related to deadlines and time limits
        deadline_keywords = [
            "deadline", "filing", "days", "appeal period", "time limit",
            "how many days", "how long", "limitation", "remand period",
            "challan", "bail application"
        ]
        if any(kw in msg_lower for kw in deadline_keywords):
            # Extract trigger event and date if present
            trigger = self._extract_trigger_event(msg_lower)
            date = self._extract_date(message)
            if trigger:
                return ToolCall(
                    tool_name="deadline_calculator",
                    arguments={"trigger_event": trigger, "event_date": date or ""}
                )

        # Case search: patterns for case references
        case_patterns = [
            r'case\s+(?:no\.?\s*)?(\d+)',
            r'citation',
            r'judgment\s+(?:in|about|regarding)',
            r'precedent',
            r'supreme\s+court\s+(?:ruling|decision|judgment)',
        ]
        for pattern in case_patterns:
            if re.search(pattern, msg_lower):
                return ToolCall(
                    tool_name="case_search",
                    arguments={"query": message}
                )

        return None

    def parse_tool_call_from_llm(self, text: str) -> Optional[ToolCall]:
        """
        Parse a <tool_call> block from LLM output.
        
        The LLM is instructed to emit tool calls in this format:
        <tool_call>
        {"tool": "tool_name", "arguments": {...}}
        </tool_call>
        
        Args:
            text: The full or partial LLM output text.
            
        Returns:
            A ToolCall if a valid block is found, None otherwise.
        """
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return None

        try:
            parsed = json.loads(match.group(1))
            tool_name = parsed.get("tool", "")
            arguments = parsed.get("arguments", {})
            if tool_name and self.registry.get(tool_name):
                return ToolCall(tool_name=tool_name, arguments=arguments)
            else:
                logger.warning("LLM requested unknown tool: {}", tool_name)
                return None
        except json.JSONDecodeError as e:
            logger.error("Failed to parse tool call JSON: {}", e)
            return None

    async def execute_tool(self, tool_call: ToolCall, timeout: float = 5.0) -> ToolResult:
        """
        Execute a tool call with a timeout.
        
        Args:
            tool_call: The parsed tool call to execute.
            timeout: Maximum execution time in seconds (default 5s per spec).
            
        Returns:
            ToolResult from the tool, or an error result on timeout/failure.
        """
        tool = self.registry.get(tool_call.tool_name)
        if not tool:
            logger.error("Tool not found: {}", tool_call.tool_name)
            return ToolResult(
                success=False,
                data=None,
                formatted_text=f"Tool '{tool_call.tool_name}' is not available.",
                error_message=f"Unknown tool: {tool_call.tool_name}"
            )

        import time
        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                tool.run(**tool_call.arguments),
                timeout=timeout
            )
            latency_ms = (time.time() - start_time) * 1000
            logger.info(
                "Tool executed | name={} | success={} | latency_ms={:.0f}",
                tool_call.tool_name, result.success, latency_ms
            )
            return result

        except asyncio.TimeoutError:
            latency_ms = (time.time() - start_time) * 1000
            logger.warning(
                "Tool timed out | name={} | timeout_s={} | latency_ms={:.0f}",
                tool_call.tool_name, timeout, latency_ms
            )
            return ToolResult(
                success=False,
                data=None,
                formatted_text="Tool timed out. Please try again.",
                error_message=f"Tool '{tool_call.tool_name}' exceeded {timeout}s timeout"
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(
                "Tool failed | name={} | error={} | latency_ms={:.0f}",
                tool_call.tool_name, str(e), latency_ms
            )
            return ToolResult(
                success=False,
                data=None,
                formatted_text=f"Tool error: {str(e)}",
                error_message=str(e)
            )

    def _extract_trigger_event(self, msg_lower: str) -> Optional[str]:
        """
        Extract a deadline trigger event from the user's message.
        
        Maps natural language phrases to DEADLINE_RULES keys.
        """
        trigger_map = {
            "arrest": "arrest",
            "bail": "arrest",
            "bail application": "arrest",
            "acquittal": "acquittal_order",
            "acquitted": "acquittal_order",
            "conviction": "conviction_order",
            "convicted": "conviction_order",
            "appeal": "conviction_order",
            "revision": "order_date",
            "challan": "arrest",
            "remand": "arrest",
            "sessions court": "sessions_court_bail_refusal",
            "high court": "sessions_court_bail_refusal",
        }
        for keyword, trigger in trigger_map.items():
            if keyword in msg_lower:
                return trigger
        return "arrest"  # Default trigger

    def _extract_date(self, message: str) -> Optional[str]:
        """
        Extract a date from the user's message.
        
        Supports YYYY-MM-DD and DD/MM/YYYY formats.
        """
        # Try YYYY-MM-DD
        match = re.search(r'(\d{4}-\d{2}-\d{2})', message)
        if match:
            return match.group(1)

        # Try DD/MM/YYYY
        match = re.search(r'(\d{2}/\d{2}/\d{4})', message)
        if match:
            return match.group(1)

        return None
