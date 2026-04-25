"""
Juris AI — Latency Performance Evaluation
Measures TTFT, inter-token latency, and end-to-end response time
across four scenarios with 10 trials each.
"""

import pytest
import json
import time
import uuid
import httpx
import numpy as np
from conftest import save_result, BASE_URL

TRIALS_PER_SCENARIO = 10


# ── Test prompts per scenario ────────────────────────────────────────────────

SCENARIOS = {
    "simple_dialogue": {
        "description": "Simple dialogue — no RAG, no tool",
        "prompts": [
            "Hello, I need legal help.",
            "Hi there, good morning.",
            "Thank you for your help.",
            "Hey, can you assist me?",
            "Good evening, I have a question.",
            "Greetings, I need your expertise.",
            "Hi, how are you?",
            "Hello Juris AI.",
            "Hey, good afternoon.",
            "Thanks for the information.",
        ],
    },
    "rag_only": {
        "description": "RAG-only — legal query requiring retrieval",
        "prompts": [
            "What is the punishment for murder under PPC?",
            "Explain bail provisions for non-bailable offenses.",
            "What does the Anti-Terrorism Act say about terrorism?",
            "How do I file an FIR for a robbery case?",
            "What is kidnapping under Pakistani criminal law?",
            "Explain culpable homicide under PPC.",
            "What is the law on digital evidence in Pakistan?",
            "What are the rights of an accused during investigation?",
            "What is criminal breach of trust under PPC?",
            "Explain defamation under Pakistan Penal Code.",
        ],
    },
    "tool_only": {
        "description": "Tool-only — deadline calculation or statute lookup",
        "prompts": [
            "What does Section 302 PPC say?",
            "Look up Section 379 of the Pakistan Penal Code.",
            "My client was arrested on 2024-03-15. What are the deadlines?",
            "Section 497 CrPC bail provisions.",
            "Calculate deadlines after conviction on 2024-06-01.",
            "Section 420 PPC cheating.",
            "What does Section 154 CrPC say about FIR?",
            "My client was arrested on 2024-01-10. Deadline?",
            "Section 302 Pakistan Penal Code punishment.",
            "Section 392 PPC robbery.",
        ],
    },
    "mixed": {
        "description": "Mixed — RAG + tool (legal query with section reference)",
        "prompts": [
            "Explain Section 302 PPC and find related Supreme Court judgments on murder.",
            "What is the bail procedure under Section 497 CrPC? Calculate deadline from arrest on 2024-02-01.",
            "Section 379 PPC theft — what are the punishments and related precedents?",
            "What are the deadlines after arrest and what does Section 167 CrPC say?",
            "Find Supreme Court judgments on bail in murder cases under Section 302 PPC.",
            "Section 420 PPC cheating — punishment and appeal deadline after conviction 2024-05-01.",
            "Explain Section 392 PPC robbery and find Supreme Court precedents.",
            "What does Section 154 CrPC say? How many days to file challan after arrest on 2024-04-01?",
            "Section 302 PPC murder — bail provisions and deadlines after arrest 2024-03-01.",
            "Anti-terrorism act provisions and Supreme Court judgment on terrorism conviction appeal.",
        ],
    },
}


async def measure_single_turn(client: httpx.AsyncClient, prompt: str, session_id: str):
    """Send a single prompt and measure latency metrics."""
    payload = {"session_id": session_id, "message": prompt}

    token_times = []
    ttft = None
    start = time.time()

    try:
        async with client.stream("POST", "/api/chat/stream", json=payload) as response:
            if response.status_code != 200:
                return None

            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if event.get("type") == "token":
                    now = time.time()
                    if ttft is None:
                        ttft = (now - start) * 1000
                    token_times.append(now)

                elif event.get("type") == "done":
                    break

    except Exception:
        return None

    e2e = (time.time() - start) * 1000

    # Compute inter-token latencies
    inter_token_latencies = []
    for i in range(1, len(token_times)):
        itl = (token_times[i] - token_times[i - 1]) * 1000
        inter_token_latencies.append(itl)

    avg_itl = float(np.mean(inter_token_latencies)) if inter_token_latencies else 0.0

    return {
        "ttft_ms": round(ttft, 2) if ttft else None,
        "e2e_ms": round(e2e, 2),
        "avg_inter_token_ms": round(avg_itl, 2),
        "token_count": len(token_times),
    }


class TestLatency:
    """Latency benchmarks across four scenarios."""

    @pytest.mark.asyncio
    async def test_latency_all_scenarios(self):
        """Run 10 trials per scenario and collect latency metrics."""
        all_results = {}

        async with httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=httpx.Timeout(connect=5.0, read=180.0, write=30.0, pool=10.0),
        ) as client:
            for scenario_name, scenario in SCENARIOS.items():
                trials = []
                prompts = scenario["prompts"]

                for i in range(min(TRIALS_PER_SCENARIO, len(prompts))):
                    session_id = f"latency_{scenario_name}_{uuid.uuid4().hex[:6]}"
                    prompt = prompts[i % len(prompts)]

                    result = await measure_single_turn(client, prompt, session_id)
                    if result:
                        result["trial"] = i + 1
                        result["prompt"] = prompt
                        trials.append(result)

                # Compute statistics
                if trials:
                    ttfts = [t["ttft_ms"] for t in trials if t["ttft_ms"] is not None]
                    e2es = [t["e2e_ms"] for t in trials]
                    itls = [t["avg_inter_token_ms"] for t in trials if t["avg_inter_token_ms"] > 0]

                    stats = {
                        "scenario": scenario_name,
                        "description": scenario["description"],
                        "trials_completed": len(trials),
                        "ttft": {
                            "mean": round(float(np.mean(ttfts)), 2) if ttfts else None,
                            "median": round(float(np.median(ttfts)), 2) if ttfts else None,
                            "p90": round(float(np.percentile(ttfts, 90)), 2) if ttfts else None,
                            "p99": round(float(np.percentile(ttfts, 99)), 2) if ttfts else None,
                        },
                        "e2e": {
                            "mean": round(float(np.mean(e2es)), 2) if e2es else None,
                            "median": round(float(np.median(e2es)), 2) if e2es else None,
                            "p90": round(float(np.percentile(e2es, 90)), 2) if e2es else None,
                            "p99": round(float(np.percentile(e2es, 99)), 2) if e2es else None,
                        },
                        "inter_token": {
                            "mean": round(float(np.mean(itls)), 2) if itls else None,
                            "median": round(float(np.median(itls)), 2) if itls else None,
                        },
                        "trials": trials,
                    }
                else:
                    stats = {
                        "scenario": scenario_name,
                        "description": scenario["description"],
                        "trials_completed": 0,
                        "ttft": {},
                        "e2e": {},
                        "inter_token": {},
                        "trials": [],
                    }

                all_results[scenario_name] = stats

        save_result("latency_results.json", all_results)

        # Basic assertions
        for name, stats in all_results.items():
            assert stats["trials_completed"] > 0, f"No trials completed for {name}"
