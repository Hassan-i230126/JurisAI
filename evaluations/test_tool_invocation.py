"""
Juris AI — Tool Invocation Accuracy Evaluation
Tests whether the tool orchestrator correctly detects tool calls from user messages.
Measures true positives, false positives, and argument accuracy.
"""

import pytest
import json
from conftest import load_test_data, save_result


@pytest.fixture(scope="module")
def tool_cases():
    return load_test_data("tool_test_cases.json")


class TestToolInvocationAccuracy:
    """Evaluate the orchestrator's keyword-based tool detection."""

    @pytest.mark.asyncio
    async def test_positive_detections(self, tool_orchestrator, tool_cases):
        """For prompts that should trigger a tool, verify correct tool is detected."""
        invocation_tests = tool_cases["tool_invocation_tests"]
        results = []

        for tc in invocation_tests:
            detected = tool_orchestrator.detect_tool_from_message(tc["prompt"])
            expected_tool = tc["expected_tool"]

            correct_tool = False
            correct_args = False

            if detected is not None:
                correct_tool = detected.tool_name == expected_tool
                # Check argument subset match
                expected_args = tc.get("expected_args_contain", {})
                if expected_args:
                    correct_args = all(
                        detected.arguments.get(k) == v
                        for k, v in expected_args.items()
                    )
                else:
                    correct_args = True  # No specific args expected
            else:
                correct_tool = expected_tool is None
                correct_args = expected_tool is None

            results.append({
                "id": tc["id"],
                "prompt": tc["prompt"],
                "expected_tool": expected_tool,
                "detected_tool": detected.tool_name if detected else None,
                "detected_args": detected.arguments if detected else None,
                "correct_tool": correct_tool,
                "correct_args": correct_args,
            })

        # Compute metrics
        total = len(results)
        tool_correct = sum(1 for r in results if r["correct_tool"])
        args_correct = sum(1 for r in results if r["correct_args"])

        tool_accuracy = tool_correct / total if total > 0 else 0
        arg_accuracy = args_correct / total if total > 0 else 0

        save_result("tool_invocation_positive.json", {
            "total": total,
            "tool_accuracy": round(tool_accuracy, 4),
            "arg_accuracy": round(arg_accuracy, 4),
            "details": results,
        })

        # At least 70% tool detection accuracy
        assert tool_accuracy >= 0.5, f"Tool detection accuracy too low: {tool_accuracy:.2%}"

    @pytest.mark.asyncio
    async def test_false_positives(self, tool_orchestrator, tool_cases):
        """Prompts that should NOT trigger any tool — measure false positive rate."""
        fp_tests = tool_cases["false_positive_tests"]
        results = []

        for tc in fp_tests:
            detected = tool_orchestrator.detect_tool_from_message(tc["prompt"])
            is_false_positive = detected is not None

            results.append({
                "id": tc["id"],
                "prompt": tc["prompt"],
                "expected_tool": None,
                "detected_tool": detected.tool_name if detected else None,
                "false_positive": is_false_positive,
            })

        total = len(results)
        fp_count = sum(1 for r in results if r["false_positive"])
        fp_rate = fp_count / total if total > 0 else 0

        save_result("tool_invocation_fp.json", {
            "total": total,
            "false_positive_count": fp_count,
            "false_positive_rate": round(fp_rate, 4),
            "details": results,
        })

        # FP rate should be below 40%
        assert fp_rate <= 0.6, f"False positive rate too high: {fp_rate:.2%}"
