"""
Juris AI — RAG Component Evaluation
Measures retrieval precision@k, recall@k, faithfulness, and context relevance.
"""

import pytest
import time
import json
import numpy as np
from conftest import load_test_data, save_result


@pytest.fixture(scope="module")
def rag_ground_truth():
    return load_test_data("rag_ground_truth.json")


class TestRAGRetrieval:
    """Evaluate retrieval quality: precision, recall, context relevance."""

    @pytest.mark.asyncio
    async def test_precision_recall(self, retriever, rag_ground_truth):
        """Compute precision@k and recall@k across all ground-truth queries."""
        results = []
        precision_scores = []
        recall_scores = []
        context_relevance_scores = []
        retrieval_latencies = []

        for gt in rag_ground_truth:
            query = gt["query"]
            expected_acts = set(gt.get("relevant_acts", []))
            expected_sections = set(gt.get("relevant_sections", []))
            expected_doc_types = set(gt.get("relevant_doc_types", []))
            expected_keywords = [kw.lower() for kw in gt.get("expected_keywords", [])]

            # Time the retrieval
            start = time.time()
            chunks = await retriever.retrieve(query, top_k=5)
            latency_ms = (time.time() - start) * 1000
            retrieval_latencies.append(latency_ms)

            k = len(chunks)
            if k == 0:
                results.append({
                    "query": query,
                    "chunks_retrieved": 0,
                    "precision_at_k": 0.0,
                    "recall_at_k": 0.0,
                    "context_relevance": 0.0,
                    "latency_ms": round(latency_ms, 1),
                })
                precision_scores.append(0.0)
                recall_scores.append(0.0)
                context_relevance_scores.append(0.0)
                continue

            # Evaluate each retrieved chunk for relevance
            relevant_count = 0
            context_useful_count = 0

            for chunk in chunks:
                is_relevant = False

                # Check act match
                if expected_acts and chunk.act:
                    act_short_map = {
                        "Pakistan Penal Code": "PPC",
                        "Code of Criminal Procedure": "CrPC",
                        "Qanun-e-Shahadat Order": "QSO",
                        "Anti-Terrorism Act": "ATA",
                    }
                    chunk_act = act_short_map.get(chunk.act, chunk.act)
                    if chunk_act in expected_acts or chunk.act in expected_acts:
                        is_relevant = True

                # Check section match
                if expected_sections and chunk.section:
                    if chunk.section in expected_sections:
                        is_relevant = True

                # Check doc_type match
                if expected_doc_types and chunk.doc_type:
                    if chunk.doc_type in expected_doc_types:
                        is_relevant = True

                if is_relevant:
                    relevant_count += 1

                # Context relevance: does the chunk text contain expected keywords?
                chunk_text_lower = chunk.text.lower()
                keyword_hits = sum(1 for kw in expected_keywords if kw in chunk_text_lower)
                if keyword_hits > 0:
                    context_useful_count += 1

            precision = relevant_count / k if k > 0 else 0.0
            # Recall: how many expected items were found
            expected_total = max(len(expected_acts) + len(expected_sections), 1)
            found_items = 0
            retrieved_acts = set()
            retrieved_sections = set()
            for chunk in chunks:
                act_short_map = {
                    "Pakistan Penal Code": "PPC",
                    "Code of Criminal Procedure": "CrPC",
                    "Qanun-e-Shahadat Order": "QSO",
                    "Anti-Terrorism Act": "ATA",
                }
                chunk_act = act_short_map.get(chunk.act, chunk.act)
                if chunk_act in expected_acts:
                    retrieved_acts.add(chunk_act)
                if chunk.section in expected_sections:
                    retrieved_sections.add(chunk.section)

            found_items = len(retrieved_acts) + len(retrieved_sections)
            recall = found_items / expected_total if expected_total > 0 else 0.0
            recall = min(recall, 1.0)

            ctx_relevance = context_useful_count / k if k > 0 else 0.0

            precision_scores.append(precision)
            recall_scores.append(recall)
            context_relevance_scores.append(ctx_relevance)

            results.append({
                "query": query,
                "chunks_retrieved": k,
                "precision_at_k": round(precision, 4),
                "recall_at_k": round(recall, 4),
                "context_relevance": round(ctx_relevance, 4),
                "latency_ms": round(latency_ms, 1),
            })

        # Aggregate metrics
        summary = {
            "total_queries": len(rag_ground_truth),
            "avg_precision_at_k": round(float(np.mean(precision_scores)), 4),
            "avg_recall_at_k": round(float(np.mean(recall_scores)), 4),
            "avg_context_relevance": round(float(np.mean(context_relevance_scores)), 4),
            "avg_retrieval_latency_ms": round(float(np.mean(retrieval_latencies)), 1),
            "median_retrieval_latency_ms": round(float(np.median(retrieval_latencies)), 1),
            "details": results,
        }

        save_result("rag_results.json", summary)

        # Basic assertion: precision should be reasonable
        assert summary["avg_precision_at_k"] >= 0.1, (
            f"Average precision@k too low: {summary['avg_precision_at_k']}"
        )

    @pytest.mark.asyncio
    async def test_faithfulness(self, retriever):
        """
        Faithfulness: check that answers grounded in retrieved context
        contain relevant information from those chunks.
        
        We test this by verifying that for known queries, the retrieved
        chunks actually contain the expected legal content.
        """
        faithfulness_cases = [
            {
                "query": "What is the punishment for murder under Section 302 PPC?",
                "expected_in_context": ["302", "murder"],
            },
            {
                "query": "Bail provisions for non-bailable offenses",
                "expected_in_context": ["bail"],
            },
            {
                "query": "Procedure for filing an FIR",
                "expected_in_context": ["FIR", "information"],
            },
        ]

        faithfulness_scores = []

        for case in faithfulness_cases:
            chunks = await retriever.retrieve(case["query"], top_k=3)
            if not chunks:
                faithfulness_scores.append(0.0)
                continue

            # Combine all chunk texts
            combined = " ".join(c.text.lower() for c in chunks)
            expected = case["expected_in_context"]
            found = sum(1 for kw in expected if kw.lower() in combined)
            score = found / len(expected) if expected else 0.0
            faithfulness_scores.append(score)

        avg_faithfulness = float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0

        save_result("rag_faithfulness.json", {
            "avg_faithfulness": round(avg_faithfulness, 4),
            "scores": [round(s, 4) for s in faithfulness_scores],
        })

        assert avg_faithfulness >= 0.3, f"Faithfulness too low: {avg_faithfulness:.2%}"
