"""
Juris AI — DeepEval RAG Evaluation
Replaces fragile string-matching with semantic LLM-as-judge metrics:
  - Faithfulness         : Does the answer stay grounded in the retrieved context?
  - Answer Relevancy     : Does the answer actually address the question?
  - Contextual Precision : Are the retrieved chunks ranked correctly?
  - Contextual Recall    : Do the retrieved chunks cover the expected content?

Uses Google Gemini Flash as the judge (free-tier API key via gemini_judge.py).
Results saved to evaluation_results/rag_deepeval_*.json.
"""

import json
import uuid
import pytest
import httpx
import numpy as np
from conftest import load_test_data, save_result, BASE_URL
from gemini_judge import GeminiJudge

from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rag_ground_truth():
    return load_test_data("rag_ground_truth.json")


@pytest.fixture(scope="module")
def judge():
    """Shared Gemini judge — one client per test module."""
    return GeminiJudge()


# ── Backend helper ────────────────────────────────────────────────────────────

async def get_rag_response(query: str) -> dict:
    """
    Call the live Juris AI backend and collect:
      - actual_output     : full LLM answer text
      - retrieval_context : list of legal chunk texts from RAG
    """
    session_id = f"deepeval_rag_{uuid.uuid4().hex[:8]}"
    payload = {"session_id": session_id, "message": query}
    tokens = []
    context_chunks = []

    async with httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=httpx.Timeout(connect=5.0, read=180.0, write=30.0, pool=10.0),
    ) as client:
        try:
            async with client.stream("POST", "/api/chat/stream", json=payload) as resp:
                if resp.status_code != 200:
                    return {"actual_output": f"HTTP {resp.status_code}", "retrieval_context": []}
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    etype = event.get("type", "")
                    if etype == "token":
                        tokens.append(event.get("content", ""))
                    elif etype == "rag_retrieved":
                        for cit in event.get("citations", []):
                            text = cit.get("text") or cit.get("chunk") or str(cit)
                            if text:
                                context_chunks.append(text)
        except Exception as e:
            return {"actual_output": f"Error: {e}", "retrieval_context": []}

    return {
        "actual_output": "".join(tokens),
        "retrieval_context": context_chunks,
    }


# ── Test class ────────────────────────────────────────────────────────────────

class TestRAGDeepEval:
    """
    DeepEval semantic evaluation of the RAG pipeline.
    Each query hits the live backend; the answer + retrieved context is
    judged by Gemini Flash.  JUDGE_SAMPLE is kept small (5) to stay within
    the free-tier rate limit of 15 RPM.
    """

    JUDGE_SAMPLE = 3   # keep low for free-tier (30 RPM); each metric makes 2-4 API calls

    @pytest.mark.asyncio
    async def test_faithfulness(self, rag_ground_truth, judge):
        """
        Faithfulness: every claim in the answer must be traceable to the
        retrieved legal chunks.  Hallucinated statutes/punishments score 0.
        Threshold: 0.70
        """
        threshold = 0.70
        scores, details = [], []

        for gt in rag_ground_truth[: self.JUDGE_SAMPLE]:
            query = gt["query"]
            result = await get_rag_response(query)
            if not result["retrieval_context"] or not result["actual_output"]:
                continue

            test_case = LLMTestCase(
                input=query,
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
                print(f"\n[WARN] FaithfulnessMetric failed for '{query}': {exc}")
                score, passed, reason = 0.0, False, str(exc)

            scores.append(score)
            details.append({"query": query, "score": round(score, 4), "passed": passed, "reason": reason})

        avg = float(np.mean(scores)) if scores else 0.0
        save_result("rag_deepeval_faithfulness.json", {
            "metric": "faithfulness",
            "judge_model": judge.get_model_name(),
            "threshold": threshold,
            "samples_evaluated": len(scores),
            "avg_score": round(avg, 4),
            "pass_rate": round(sum(1 for d in details if d["passed"]) / len(details), 4) if details else 0,
            "details": details,
        })

        assert avg >= threshold, (
            f"Average Faithfulness {avg:.2%} < threshold {threshold:.0%}. "
            f"The LLM may be hallucinating legal facts not present in retrieved context."
        )

    @pytest.mark.asyncio
    async def test_answer_relevancy(self, rag_ground_truth, judge):
        """
        Answer Relevancy: the answer must directly address the user's legal question.
        Penalises vague, evasive, or off-topic responses.
        Threshold: 0.70
        """
        threshold = 0.70
        scores, details = [], []

        for gt in rag_ground_truth[: self.JUDGE_SAMPLE]:
            query = gt["query"]
            result = await get_rag_response(query)
            if not result["actual_output"]:
                continue

            test_case = LLMTestCase(
                input=query,
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
                print(f"\n[WARN] AnswerRelevancyMetric failed for '{query}': {exc}")
                score, passed, reason = 0.0, False, str(exc)

            scores.append(score)
            details.append({"query": query, "score": round(score, 4), "passed": passed, "reason": reason})

        avg = float(np.mean(scores)) if scores else 0.0
        save_result("rag_deepeval_relevancy.json", {
            "metric": "answer_relevancy",
            "judge_model": judge.get_model_name(),
            "threshold": threshold,
            "samples_evaluated": len(scores),
            "avg_score": round(avg, 4),
            "pass_rate": round(sum(1 for d in details if d["passed"]) / len(details), 4) if details else 0,
            "details": details,
        })

        assert avg >= threshold, (
            f"Average Answer Relevancy {avg:.2%} < threshold {threshold:.0%}. "
            f"Responses may be too generic or not directly answering the legal question."
        )

    @pytest.mark.asyncio
    async def test_contextual_precision(self, rag_ground_truth, judge):
        """
        Contextual Precision: are the top-ranked chunks actually most useful?
        Penalises retrieval that surfaces irrelevant chunks first.
        Threshold: 0.60 (lower because we have no reranker).
        """
        threshold = 0.60
        scores, details = [], []

        for gt in rag_ground_truth[: self.JUDGE_SAMPLE]:
            query = gt["query"]
            expected_keywords = gt.get("expected_keywords", [])
            if not expected_keywords:
                continue
            expected_output = f"The response should discuss: {', '.join(expected_keywords)}."
            result = await get_rag_response(query)
            if not result["retrieval_context"] or not result["actual_output"]:
                continue

            test_case = LLMTestCase(
                input=query,
                actual_output=result["actual_output"],
                expected_output=expected_output,
                retrieval_context=result["retrieval_context"],
            )
            try:
                metric = ContextualPrecisionMetric(threshold=threshold, model=judge, verbose_mode=False)
                metric.measure(test_case)
                score = metric.score if metric.score is not None else 0.0
                passed = metric.is_successful()
                reason = metric.reason
            except Exception as exc:
                print(f"\n[WARN] ContextualPrecisionMetric failed for '{query}': {exc}")
                score, passed, reason = 0.0, False, str(exc)

            scores.append(score)
            details.append({"query": query, "score": round(score, 4), "passed": passed, "reason": reason})

        avg = float(np.mean(scores)) if scores else 0.0
        save_result("rag_deepeval_precision.json", {
            "metric": "contextual_precision",
            "judge_model": judge.get_model_name(),
            "threshold": threshold,
            "samples_evaluated": len(scores),
            "avg_score": round(avg, 4),
            "pass_rate": round(sum(1 for d in details if d["passed"]) / len(details), 4) if details else 0,
            "details": details,
        })

        assert avg >= threshold, (
            f"Average Contextual Precision {avg:.2%} < threshold {threshold:.0%}. "
            f"The retriever may be returning loosely related chunks ranked above specific ones."
        )

    @pytest.mark.asyncio
    async def test_contextual_recall(self, rag_ground_truth, judge):
        """
        Contextual Recall: do the chunks collectively cover all content
        needed to answer the question?
        Threshold: 0.60
        """
        threshold = 0.60
        scores, details = [], []

        for gt in rag_ground_truth[: self.JUDGE_SAMPLE]:
            query = gt["query"]
            expected_keywords = gt.get("expected_keywords", [])
            if not expected_keywords:
                continue
            expected_output = f"The response should discuss: {', '.join(expected_keywords)}."
            result = await get_rag_response(query)
            if not result["retrieval_context"] or not result["actual_output"]:
                continue

            test_case = LLMTestCase(
                input=query,
                actual_output=result["actual_output"],
                expected_output=expected_output,
                retrieval_context=result["retrieval_context"],
            )
            try:
                metric = ContextualRecallMetric(threshold=threshold, model=judge, verbose_mode=False)
                metric.measure(test_case)
                score = metric.score if metric.score is not None else 0.0
                passed = metric.is_successful()
                reason = metric.reason
            except Exception as exc:
                print(f"\n[WARN] ContextualRecallMetric failed for '{query}': {exc}")
                score, passed, reason = 0.0, False, str(exc)

            scores.append(score)
            details.append({"query": query, "score": round(score, 4), "passed": passed, "reason": reason})

        avg = float(np.mean(scores)) if scores else 0.0
        save_result("rag_deepeval_recall.json", {
            "metric": "contextual_recall",
            "judge_model": judge.get_model_name(),
            "threshold": threshold,
            "samples_evaluated": len(scores),
            "avg_score": round(avg, 4),
            "pass_rate": round(sum(1 for d in details if d["passed"]) / len(details), 4) if details else 0,
            "details": details,
        })

        assert avg >= threshold, (
            f"Average Contextual Recall {avg:.2%} < threshold {threshold:.0%}. "
            f"Retrieved context may be missing key legal information needed to answer the query."
        )
