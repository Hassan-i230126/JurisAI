"""
Juris AI — Evaluation Runner
Single entry point: runs all pytest tests, then generates the report.

Usage:
    python evaluations/run_evals.py
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
    print("\n[1/6] Running CRM tests...")
    subprocess.run([PYTHON, "-m", "pytest", "test_crm.py", "-v", "--tb=short"], cwd=cwd)

    print("\n[2/6] Running tool functional tests...")
    subprocess.run([PYTHON, "-m", "pytest", "test_tools.py", "-v", "--tb=short"], cwd=cwd)

    print("\n[3/6] Running tool invocation accuracy tests...")
    subprocess.run([PYTHON, "-m", "pytest", "test_tool_invocation.py", "-v", "--tb=short"], cwd=cwd)

    # ── Phase 2: RAG evaluation ──────────────────────────────────────────
    print("\n[4/6] Running RAG evaluation...")
    subprocess.run([PYTHON, "-m", "pytest", "test_rag.py", "-v", "--tb=short"], cwd=cwd)

    # ── Phase 3: Conversational + Performance ────────────────────────────
    print("\n[5/6] Running conversational correctness tests...")
    subprocess.run([PYTHON, "-m", "pytest", "test_conversational.py", "-v", "--tb=short"], cwd=cwd)

    print("\n[6/6] Running latency & throughput benchmarks...")
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
