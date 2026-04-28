"""
Juris AI — Evaluation Runner
Single entry point: runs all pytest tests, then generates the report.

Usage:
    python evaluations/run_evals.py

Requires GEMINI_API_KEY in your .env file for the DeepEval (LLM-as-judge) phases.
Get a free key at: https://aistudio.google.com/app/apikey
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "evaluation_results"
PYTHON = sys.executable


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Juris AI — Evaluation Suite")
    print("=" * 60)

    # All pytest calls use cwd=EVAL_DIR so conftest.py is found
    cwd = str(EVAL_DIR)

    # ── Phase 1: Component tests ─────────────────────────────────────────
    print("\n[1/8] Running CRM tests...")
    subprocess.run([PYTHON, "-m", "pytest", "test_crm.py", "-v", "--tb=short"], cwd=cwd)

    print("\n[2/8] Running tool functional tests...")
    subprocess.run([PYTHON, "-m", "pytest", "test_tools.py", "-v", "--tb=short"], cwd=cwd)

    print("\n[3/8] Running tool invocation accuracy tests...")
    subprocess.run([PYTHON, "-m", "pytest", "test_tool_invocation.py", "-v", "--tb=short"], cwd=cwd)

    # ── Phase 2: RAG evaluation (custom string-matching — unchanged) ──────
    print("\n[4/8] Running RAG evaluation (string-matching)...")
    subprocess.run([PYTHON, "-m", "pytest", "test_rag.py", "-v", "--tb=short"], cwd=cwd)

    # ── Phase 3: DeepEval RAG — LLM-as-judge semantic evaluation ─────────
    print("\n[5/8] Running DeepEval RAG evaluation (LLM-as-judge via Gemini)...")
    print("      Requires GEMINI_API_KEY in environment. Skipped if key is missing.")
    subprocess.run(
        [PYTHON, "-m", "pytest", "test_rag_deepeval.py", "-v", "--tb=short", "-s"],
        cwd=cwd,
    )

    # ── Phase 4: Conversational correctness (custom string-matching — unchanged)
    print("\n[6/8] Running conversational correctness tests (string-matching)...")
    subprocess.run([PYTHON, "-m", "pytest", "test_conversational.py", "-v", "--tb=short"], cwd=cwd)

    # ── Phase 5: DeepEval Conversational — LLM-as-judge semantic evaluation
    print("\n[7/8] Running DeepEval conversational evaluation (LLM-as-judge via Gemini)...")
    print("      Requires GEMINI_API_KEY in environment. Skipped if key is missing.")
    subprocess.run(
        [PYTHON, "-m", "pytest", "test_conversational_deepeval.py", "-v", "--tb=short", "-s"],
        cwd=cwd,
    )

    # ── Phase 6: Performance benchmarks (unchanged) ───────────────────────
    print("\n[8/8] Running latency & throughput benchmarks...")
    subprocess.run([PYTHON, "-m", "pytest", "test_latency.py", "test_throughput.py", "-v", "--tb=short"], cwd=cwd)

    # ── Phase 4: Generate report ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Generating evaluation report...")
    print("=" * 60)
    subprocess.run([PYTHON, str(EVAL_DIR / "generate_report.py")], cwd=str(PROJECT_ROOT))

    print("\n✅ Evaluation complete!")
    print(f"   Report: {PROJECT_ROOT / 'evaluation_result.md'}")
    print(f"   Charts: {PROJECT_ROOT / 'evaluation_charts' / ''}")


if __name__ == "__main__":
    main()
