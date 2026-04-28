"""
Juris AI — DeepEval Conversational Correctness Evaluation
Adds semantic LLM-as-judge metrics on top of the unchanged string-matching
test_conversational.py:

  - Answer Relevancy  : Did the AI actually answer the question asked?
  - Faithfulness      : Is the legal answer grounded in retrieved context?
  - Hallucination     : Did the AI fabricate legal facts not in context?
  - GEval (coherence) : Does the AI maintain coherent context across turns?

Uses Google Gemini Flash as the judge (free-tier, rate-limited via gemini_judge.py).
Results saved to evaluation_results/conversational_deepeval_*.json.
"""

import json
import time
import uuid
import pytest
import httpx
import numpy as np
from conftest import load_test_data, save_result, BASE_URL
from gemini_judge import GeminiJudge

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def conversations():
    return load_test_data("conversations.json")


@pytest.fixture(scope="module")
def judge():
    return GeminiJudge()


# ── Backend interaction helper ────────────────────────────────────────────────

async def stream_turn(client: httpx.AsyncClient, session_id: str, message: str) -> dict:
    """
    Send one turn to the backend streaming endpoint.
    Returns actual_output, retrieval_context, and tools_used.
    """
    payload = {"session_id": session_id, "message": message}
    tokens, context_chunks, tools_used = [], [], []
    ttft = None
    start = time.time()

    try:
        async with client.stream("POST", "/api/chat/stream", json=payload) as resp:
            if resp.status_code != 200:
                return {
                    "actual_output": f"HTTP Error {resp.status_code}",
                    "retrieval_context": [], "tools_used": [],
                    "ttft_ms": None, "e2e_ms": (time.time() - start) * 1000,
                }
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type", "")
                if etype == "token":
                    if ttft is None:
                        ttft = (time.time() - start) * 1000
                    tokens.append(event.get("content", ""))
                elif etype == "rag_retrieved":
                    for cit in event.get("citations", []):
                        text = cit.get("text") or cit.get("chunk") or str(cit)
                        if text:
                            context_chunks.append(text)
                elif etype == "tool_invoked":
                    tools_used.append(event.get("tool_name", ""))
                elif etype == "done":
                    if event.get("tools_used"):
                        tools_used.extend(event["tools_used"])
    except Exception as e:
        return {
            "actual_output": f"Error: {str(e)}",
            "retrieval_context": [], "tools_used": [],
            "ttft_ms": ttft, "e2e_ms": (time.time() - start) * 1000,
        }

    return {
        "actual_output": "".join(tokens),
        "retrieval_context": context_chunks,
        "tools_used": list(set(tools_used)),
        "ttft_ms": round(ttft, 1) if ttft else None,
        "e2e_ms": round((time.time() - start) * 1000, 1),
    }


# ── Test class ────────────────────────────────────────────────────────────────

class TestConversationalDeepEval:
    """
    Semantic evaluation of Juris AI conversational quality.
    Only runs on RAG-heavy categories (legal_rag, mixed, coherence) where
    string-matching is weakest.  Greeting / out-of-domain categories are
    already well-covered by test_conversational.py.
    JUDGE_SAMPLE_CATS limits which conversations are included so we stay
    under the free-tier 15-RPM rate limit.
    """

    JUDGE_CATEGORIES = {"legal_rag", "mixed", "coherence"}

    @pytest.mark.asyncio
    async def test_answer_relevancy(self, conversations, judge):
        """
        Does the AI answer the legal question directly?
        Threshold: 0.70
        """
        threshold = 0.70
        scores, details = [], []

        async with httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=httpx.Timeout(connect=5.0, read=180.0, write=30.0, pool=10.0),
        ) as client:
            for conv in conversations:
                if conv.get("category") not in self.JUDGE_CATEGORIES:
                    continue

                session_id = f"deepeval_rel_{conv['id']}_{uuid.uuid4().hex[:6]}"

                # Run preceding turns to build context
                for turn in conv["turns"][:-1]:
                    await stream_turn(client, session_id, turn["content"])

                last_turn = conv["turns"][-1]
                result = await stream_turn(client, session_id, last_turn["content"])

                if not result["actual_output"] or result["actual_output"].startswith("Error"):
                    continue

                test_case = LLMTestCase(
                    input=last_turn["content"],
                    actual_output=result["actual_output"],
                    retrieval_context=result["retrieval_context"] or [""],
                )
                try:
                    metric = AnswerRelevancyMetric(threshold=threshold, model=judge, verbose_mode=False)
                    metric.measure(test_case)
                    score = metric.score if metric.score is not None else 0.0
                    passed = metric.is_successful()
                    reason = metric.reason
                except Exception as exc:
                    print(f"\n[WARN] AnswerRelevancy failed for conv '{conv['id']}': {exc}")
                    score, passed, reason = 0.0, False, str(exc)

                scores.append(score)
                details.append({
                    "conv_id": conv["id"], "description": conv["description"],
                    "category": conv["category"], "input": last_turn["content"],
                    "output_preview": result["actual_output"][:300],
                    "score": round(score, 4), "passed": passed, "reason": reason,
                })

        avg = float(np.mean(scores)) if scores else 0.0
        save_result("conversational_deepeval_relevancy.json", {
            "metric": "answer_relevancy", "judge_model": judge.get_model_name(),
            "threshold": threshold, "samples_evaluated": len(scores),
            "avg_score": round(avg, 4),
            "pass_rate": round(sum(1 for d in details if d["passed"]) / len(details), 4) if details else 0,
            "details": details,
        })
        assert avg >= threshold, (
            f"Average Answer Relevancy {avg:.2%} < threshold {threshold:.0%}. "
            f"The LLM may be giving vague or evasive legal answers."
        )

    @pytest.mark.asyncio
    async def test_faithfulness_rag_turns(self, conversations, judge):
        """
        Every claim in RAG-grounded answers must be traceable to retrieved chunks.
        Catches hallucinated statutes, wrong punishments, invented deadlines.
        Threshold: 0.70
        """
        threshold = 0.70
        scores, details = [], []

        async with httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=httpx.Timeout(connect=5.0, read=180.0, write=30.0, pool=10.0),
        ) as client:
            for conv in conversations:
                if conv.get("category") not in {"legal_rag", "mixed"}:
                    continue

                session_id = f"deepeval_faith_{conv['id']}_{uuid.uuid4().hex[:6]}"

                for turn in conv["turns"][:-1]:
                    await stream_turn(client, session_id, turn["content"])

                last_turn = conv["turns"][-1]
                result = await stream_turn(client, session_id, last_turn["content"])

                # Skip if no retrieval context (not a RAG turn)
                if not result["retrieval_context"] or not result["actual_output"]:
                    continue
                if result["actual_output"].startswith("Error"):
                    continue

                test_case = LLMTestCase(
                    input=last_turn["content"],
                    actual_output=result["actual_output"],
                    retrieval_context=result["retrieval_context"],
                )
                try:
                    metric = FaithfulnessMetric(threshold=threshold, model=judge, verbose_mode=False)
                    metric.measure(test_case)
                    score = metric.score if metric.score is not None else 0.0
                    passed = metric.is_successful()
                    reason = metric.reason
                except Exception as exc:
                    print(f"\n[WARN] Faithfulness failed for conv '{conv['id']}': {exc}")
                    score, passed, reason = 0.0, False, str(exc)

                scores.append(score)
                details.append({
                    "conv_id": conv["id"], "description": conv["description"],
                    "category": conv["category"], "input": last_turn["content"],
                    "output_preview": result["actual_output"][:300],
                    "score": round(score, 4), "passed": passed, "reason": reason,
                })

        avg = float(np.mean(scores)) if scores else 0.0
        save_result("conversational_deepeval_faithfulness.json", {
            "metric": "faithfulness", "judge_model": judge.get_model_name(),
            "threshold": threshold, "samples_evaluated": len(scores),
            "avg_score": round(avg, 4),
            "pass_rate": round(sum(1 for d in details if d["passed"]) / len(details), 4) if details else 0,
            "details": details,
        })
        assert avg >= threshold, (
            f"Average Faithfulness {avg:.2%} < threshold {threshold:.0%}. "
            f"The LLM may be hallucinating legal content not present in the retrieved statutes."
        )

    @pytest.mark.asyncio
    async def test_hallucination(self, conversations, judge):
        """
        HallucinationMetric: score of 0 = no hallucination, 1 = full hallucination.
        Assert average hallucination score BELOW 0.25 (mostly grounded).
        """
        max_hallucination = 0.25
        scores, details = [], []

        async with httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=httpx.Timeout(connect=5.0, read=180.0, write=30.0, pool=10.0),
        ) as client:
            for conv in conversations:
                if conv.get("category") not in {"legal_rag", "mixed"}:
                    continue

                session_id = f"deepeval_halluc_{conv['id']}_{uuid.uuid4().hex[:6]}"

                for turn in conv["turns"][:-1]:
                    await stream_turn(client, session_id, turn["content"])

                last_turn = conv["turns"][-1]
                result = await stream_turn(client, session_id, last_turn["content"])

                if not result["retrieval_context"] or not result["actual_output"]:
                    continue
                if result["actual_output"].startswith("Error"):
                    continue

                # HallucinationMetric uses `context`, not `retrieval_context`
                test_case = LLMTestCase(
                    input=last_turn["content"],
                    actual_output=result["actual_output"],
                    context=result["retrieval_context"],
                )
                try:
                    metric = HallucinationMetric(
                        threshold=max_hallucination,
                        model=judge,
                        verbose_mode=False,
                    )
                    metric.measure(test_case)
                    score = metric.score if metric.score is not None else 1.0
                    passed = metric.is_successful()
                    reason = metric.reason
                except Exception as exc:
                    print(f"\n[WARN] HallucinationMetric failed for conv '{conv['id']}': {exc}")
                    score, passed, reason = 1.0, False, str(exc)

                scores.append(score)
                details.append({
                    "conv_id": conv["id"], "description": conv["description"],
                    "category": conv["category"], "input": last_turn["content"],
                    "output_preview": result["actual_output"][:300],
                    "hallucination_score": round(score, 4), "passed": passed, "reason": reason,
                })

        avg = float(np.mean(scores)) if scores else 1.0
        save_result("conversational_deepeval_hallucination.json", {
            "metric": "hallucination", "judge_model": judge.get_model_name(),
            "max_threshold": max_hallucination, "samples_evaluated": len(scores),
            "avg_hallucination_score": round(avg, 4),
            "pass_rate": round(sum(1 for d in details if d["passed"]) / len(details), 4) if details else 0,
            "details": details,
        })
        assert avg <= max_hallucination, (
            f"Average Hallucination score {avg:.2%} exceeds allowed maximum {max_hallucination:.0%}. "
            f"The LLM is fabricating legal information not present in the retrieved context."
        )

    @pytest.mark.asyncio
    async def test_coherence_multiturn(self, conversations, judge):
        """
        Multi-turn coherence: does the AI correctly reference earlier turns?
        Uses a custom GEval rubric.
        Threshold: 0.65
        """
        threshold = 0.65
        scores, details = [], []

        coherence_metric = GEval(
            name="Legal Conversational Coherence",
            criteria=(
                "Evaluate whether the AI assistant's final response demonstrates "
                "awareness of the full prior conversation. Specifically: "
                "(1) Does it correctly reference the legal topic discussed earlier? "
                "(2) Does it avoid contradicting its own prior statements? "
                "(3) Does it build logically on the prior exchange rather than "
                "treating each turn as independent? "
                "Score 0-1 where 1 = perfect coherence."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.CONTEXT,
            ],
            model=judge,
            threshold=threshold,
            verbose_mode=False,
        )

        async with httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=httpx.Timeout(connect=5.0, read=180.0, write=30.0, pool=10.0),
        ) as client:
            for conv in conversations:
                if conv.get("category") != "coherence" or len(conv["turns"]) < 2:
                    continue

                session_id = f"deepeval_coh_{conv['id']}_{uuid.uuid4().hex[:6]}"
                all_responses = []

                for turn in conv["turns"]:
                    result = await stream_turn(client, session_id, turn["content"])
                    all_responses.append({
                        "user": turn["content"],
                        "assistant": result["actual_output"],
                    })

                if len(all_responses) < 2:
                    continue

                last_input = all_responses[-1]["user"]
                last_output = all_responses[-1]["assistant"]
                if not last_output or last_output.startswith("Error"):
                    continue

                context_lines = []
                for i, pair in enumerate(all_responses[:-1]):
                    context_lines.append(f"Turn {i+1} User: {pair['user']}")
                    context_lines.append(f"Turn {i+1} Assistant: {pair['assistant'][:300]}")
                context = ["\n".join(context_lines)]

                test_case = LLMTestCase(
                    input=last_input,
                    actual_output=last_output,
                    context=context,
                )
                try:
                    coherence_metric.measure(test_case)
                    score = coherence_metric.score if coherence_metric.score is not None else 0.0
                    passed = coherence_metric.is_successful()
                    reason = coherence_metric.reason
                except Exception as exc:
                    print(f"\n[WARN] GEval coherence failed for conv '{conv['id']}': {exc}")
                    score, passed, reason = 0.0, False, str(exc)

                scores.append(score)
                details.append({
                    "conv_id": conv["id"], "description": conv["description"],
                    "num_turns": len(conv["turns"]),
                    "final_input": last_input,
                    "final_output_preview": last_output[:300],
                    "score": round(score, 4), "passed": passed, "reason": reason,
                })

        avg = float(np.mean(scores)) if scores else 0.0
        save_result("conversational_deepeval_coherence.json", {
            "metric": "geval_coherence", "judge_model": judge.get_model_name(),
            "threshold": threshold, "samples_evaluated": len(scores),
            "avg_score": round(avg, 4),
            "pass_rate": round(sum(1 for d in details if d["passed"]) / len(details), 4) if details else 0,
            "details": details,
        })
        assert avg >= threshold, (
            f"Average Multi-turn Coherence {avg:.2%} < threshold {threshold:.0%}. "
            f"The LLM may not be maintaining proper context across conversation turns."
        )
