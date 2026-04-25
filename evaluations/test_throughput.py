"""
Juris AI — Throughput / Concurrency Evaluation
Simulates concurrent users and measures latency degradation.
"""

import pytest
import json
import time
import uuid
import asyncio
import httpx
import numpy as np
from conftest import save_result, BASE_URL

CONCURRENCY_LEVELS = [1, 2, 3, 5]
TURNS_PER_USER = 3

# Fixed conversation prompts for each simulated user
USER_PROMPTS = [
    "What is the punishment for murder under Section 302 PPC?",
    "Explain bail provisions under CrPC.",
    "What are the rights of an accused?",
]


async def simulate_user(client: httpx.AsyncClient, user_id: int, prompts: list):
    """Simulate a single user sending multiple turns."""
    session_id = f"throughput_u{user_id}_{uuid.uuid4().hex[:6]}"
    turn_results = []

    for turn_idx, prompt in enumerate(prompts[:TURNS_PER_USER]):
        payload = {"session_id": session_id, "message": prompt}
        ttft = None
        start = time.time()
        token_count = 0

        try:
            async with client.stream("POST", "/api/chat/stream", json=payload) as response:
                if response.status_code != 200:
                    turn_results.append({
                        "turn": turn_idx,
                        "error": f"HTTP {response.status_code}",
                        "e2e_ms": (time.time() - start) * 1000,
                    })
                    continue

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if event.get("type") == "token":
                        if ttft is None:
                            ttft = (time.time() - start) * 1000
                        token_count += 1
                    elif event.get("type") == "done":
                        break

        except Exception as e:
            turn_results.append({
                "turn": turn_idx,
                "error": str(e),
                "e2e_ms": (time.time() - start) * 1000,
            })
            continue

        e2e = (time.time() - start) * 1000
        turn_results.append({
            "turn": turn_idx,
            "ttft_ms": round(ttft, 2) if ttft else None,
            "e2e_ms": round(e2e, 2),
            "token_count": token_count,
        })

    return {"user_id": user_id, "turns": turn_results}


class TestThroughput:
    """Concurrency and throughput evaluation."""

    @pytest.mark.asyncio
    async def test_concurrency_levels(self):
        """Test increasing concurrency levels and measure latency degradation."""
        all_results = {}

        for n_users in CONCURRENCY_LEVELS:
            async with httpx.AsyncClient(
                base_url=BASE_URL,
                timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=30.0),
                limits=httpx.Limits(max_connections=n_users + 5, max_keepalive_connections=n_users + 5),
            ) as client:
                start = time.time()

                # Launch n_users concurrently
                tasks = [
                    simulate_user(client, uid, USER_PROMPTS)
                    for uid in range(n_users)
                ]
                user_results = await asyncio.gather(*tasks, return_exceptions=True)
                wall_time = (time.time() - start) * 1000

                # Collect metrics
                ttfts = []
                e2es = []
                errors = 0
                total_turns = 0

                for ur in user_results:
                    if isinstance(ur, Exception):
                        errors += 1
                        continue
                    for t in ur["turns"]:
                        total_turns += 1
                        if "error" in t:
                            errors += 1
                        else:
                            if t.get("ttft_ms"):
                                ttfts.append(t["ttft_ms"])
                            e2es.append(t["e2e_ms"])

                turns_per_second = total_turns / (wall_time / 1000) if wall_time > 0 else 0

                level_stats = {
                    "concurrent_users": n_users,
                    "total_turns": total_turns,
                    "errors": errors,
                    "wall_time_ms": round(wall_time, 2),
                    "turns_per_second": round(turns_per_second, 4),
                    "ttft": {
                        "mean": round(float(np.mean(ttfts)), 2) if ttfts else None,
                        "median": round(float(np.median(ttfts)), 2) if ttfts else None,
                        "p90": round(float(np.percentile(ttfts, 90)), 2) if len(ttfts) >= 2 else None,
                    },
                    "e2e": {
                        "mean": round(float(np.mean(e2es)), 2) if e2es else None,
                        "median": round(float(np.median(e2es)), 2) if e2es else None,
                        "p90": round(float(np.percentile(e2es, 90)), 2) if len(e2es) >= 2 else None,
                    },
                    "user_details": [
                        ur if not isinstance(ur, Exception) else {"error": str(ur)}
                        for ur in user_results
                    ],
                }

                all_results[f"{n_users}_users"] = level_stats

        save_result("throughput_results.json", all_results)

        # Basic assertion: at least 1-user level should complete
        assert all_results.get("1_users", {}).get("total_turns", 0) > 0
