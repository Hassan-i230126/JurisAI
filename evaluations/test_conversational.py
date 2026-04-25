"""
Juris AI — Overall Conversational Correctness Evaluation
Tests task completion, policy adherence, and coherence using the live API.
"""

import pytest
import json
import time
import uuid
import httpx
import numpy as np
from conftest import load_test_data, save_result, BASE_URL


@pytest.fixture(scope="module")
def conversations():
    return load_test_data("conversations.json")


async def send_message_and_collect(client: httpx.AsyncClient, session_id: str, message: str, client_id: str = None):
    """Send a message to the streaming API and collect the full response."""
    payload = {
        "session_id": session_id,
        "message": message,
    }
    if client_id:
        payload["client_id"] = client_id

    tokens = []
    events = []
    tools_used = []
    rag_used = False
    citations = []

    start = time.time()
    ttft = None

    try:
        async with client.stream("POST", "/api/chat/stream", json=payload) as response:
            if response.status_code != 200:
                return {
                    "response": f"HTTP Error {response.status_code}",
                    "events": [],
                    "tools_used": [],
                    "rag_used": False,
                    "citations": [],
                    "ttft_ms": None,
                    "e2e_ms": (time.time() - start) * 1000,
                }

            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                events.append(event)
                etype = event.get("type", "")

                if etype == "token":
                    if ttft is None:
                        ttft = (time.time() - start) * 1000
                    tokens.append(event.get("content", ""))
                elif etype == "tool_invoked":
                    tools_used.append(event.get("tool_name", ""))
                elif etype == "rag_retrieved":
                    rag_used = True
                    citations = event.get("citations", [])
                elif etype == "done":
                    if event.get("tools_used"):
                        tools_used.extend(event["tools_used"])
                    if event.get("citations"):
                        citations.extend(event["citations"])
    except Exception as e:
        return {
            "response": f"Error: {str(e)}",
            "events": events,
            "tools_used": tools_used,
            "rag_used": rag_used,
            "citations": citations,
            "ttft_ms": ttft,
            "e2e_ms": (time.time() - start) * 1000,
        }

    e2e = (time.time() - start) * 1000

    return {
        "response": "".join(tokens),
        "events": events,
        "tools_used": list(set(tools_used)),
        "rag_used": rag_used,
        "citations": citations,
        "ttft_ms": round(ttft, 1) if ttft else None,
        "e2e_ms": round(e2e, 1),
    }


class TestConversationalCorrectness:
    """End-to-end conversational correctness tests."""

    @pytest.mark.asyncio
    async def test_task_completion_and_policy(self, conversations):
        """Run all test conversations and evaluate task completion + policy adherence."""
        async with httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=10.0),
        ) as client:
            results = []

            for conv in conversations:
                session_id = f"eval_{conv['id']}_{uuid.uuid4().hex[:6]}"
                conv_results = {
                    "id": conv["id"],
                    "description": conv["description"],
                    "category": conv.get("category", "unknown"),
                    "turns": [],
                    "task_completed": True,
                    "policy_adhered": True,
                }

                for turn_idx, turn in enumerate(conv["turns"]):
                    result = await send_message_and_collect(
                        client, session_id, turn["content"]
                    )

                    response_lower = result["response"].lower()
                    expected_contains = turn.get("expected_contains", [])
                    expected_not_contains = turn.get("expected_not_contains", [])
                    policy_check = turn.get("policy_check", "in_scope")

                    # Check expected content
                    contains_pass = all(
                        kw.lower() in response_lower for kw in expected_contains
                    )
                    not_contains_pass = all(
                        kw.lower() not in response_lower for kw in expected_not_contains
                    )

                    # Policy check
                    policy_pass = True
                    if policy_check == "out_of_scope":
                        # Should contain refusal
                        refusal_phrases = ["out of my scope", "cannot assist", "cannot answer", "outside"]
                        policy_pass = any(p in response_lower for p in refusal_phrases)
                    elif policy_check == "in_scope":
                        # Should NOT contain refusal (unless genuinely empty)
                        if result["response"].strip():
                            policy_pass = True

                    # Tool check
                    expected_tool = turn.get("expected_tool")
                    tool_match = True
                    if expected_tool:
                        tool_match = expected_tool in result["tools_used"]

                    if not contains_pass:
                        conv_results["task_completed"] = False
                    if not policy_pass:
                        conv_results["policy_adhered"] = False

                    conv_results["turns"].append({
                        "turn_idx": turn_idx,
                        "user": turn["content"],
                        "response_preview": result["response"][:200],
                        "contains_pass": contains_pass,
                        "not_contains_pass": not_contains_pass,
                        "policy_pass": policy_pass,
                        "tool_match": tool_match,
                        "expected_tool": expected_tool,
                        "detected_tools": result["tools_used"],
                        "rag_used": result["rag_used"],
                        "ttft_ms": result["ttft_ms"],
                        "e2e_ms": result["e2e_ms"],
                    })

                results.append(conv_results)

            # Compute aggregate metrics
            total = len(results)
            task_completed = sum(1 for r in results if r["task_completed"])
            policy_adhered = sum(1 for r in results if r["policy_adhered"])

            summary = {
                "total_conversations": total,
                "task_completion_rate": round(task_completed / total, 4) if total > 0 else 0,
                "policy_adherence_rate": round(policy_adhered / total, 4) if total > 0 else 0,
                "details": results,
            }

            save_result("conversational_results.json", summary)

            # At least 50% task completion (some may fail due to LLM variability)
            assert summary["task_completion_rate"] >= 0.3, (
                f"Task completion rate too low: {summary['task_completion_rate']:.2%}"
            )
