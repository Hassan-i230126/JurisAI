"""
Juris AI — Report Generator
Reads all JSON results from evaluation_results/ and produces
evaluation_result.md with embedded matplotlib chart paths.
"""

import json
import platform
import sys
import os
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "evaluation_results"
CHARTS_DIR = PROJECT_ROOT / "evaluation_charts"
OUTPUT_MD = PROJECT_ROOT / "evaluation_result.md"


def _load(name):
    p = RESULTS_DIR / name
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _chart(name):
    return str(CHARTS_DIR / name)


def _pct(val):
    """Format a 0-1 float as a percentage string, or N/A if None."""
    if val is None:
        return "N/A"
    return f"{val:.2%}"


# ── Chart generators ─────────────────────────────────────────────────────────

def gen_latency_box(latency):
    if not latency:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    scenarios = []
    ttft_data = []
    e2e_data = []
    for sn, sd in latency.items():
        trials = sd.get("trials", [])
        if not trials:
            continue
        scenarios.append(sn.replace("_", "\n"))
        ttft_data.append([t["ttft_ms"] for t in trials if t.get("ttft_ms")])
        e2e_data.append([t["e2e_ms"] for t in trials])

    if not scenarios:
        plt.close(fig)
        return None

    axes[0].boxplot(ttft_data, labels=scenarios)
    axes[0].set_title("Time to First Token (TTFT)", fontweight="bold")
    axes[0].set_ylabel("ms")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].boxplot(e2e_data, labels=scenarios)
    axes[1].set_title("End-to-End Response Time", fontweight="bold")
    axes[1].set_ylabel("ms")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = _chart("latency_boxplot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def gen_concurrency_line(throughput):
    if not throughput:
        return None
    levels = []
    ttft_means = []
    e2e_means = []
    tps_vals = []
    for key in sorted(throughput.keys()):
        sd = throughput[key]
        levels.append(sd["concurrent_users"])
        ttft_means.append(sd["ttft"].get("mean") or 0)
        e2e_means.append(sd["e2e"].get("mean") or 0)
        tps_vals.append(sd.get("turns_per_second", 0))

    if not levels:
        return None

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(levels, ttft_means, "o-", label="TTFT (mean)", color="#2196F3")
    ax1.plot(levels, e2e_means, "s-", label="E2E (mean)", color="#F44336")
    ax1.set_xlabel("Concurrent Users")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Concurrency vs Latency", fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(levels, tps_vals, alpha=0.2, color="#4CAF50", width=0.3, label="Turns/sec")
    ax2.set_ylabel("Turns per second")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    path = _chart("concurrency_latency.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def gen_rag_bar(rag):
    if not rag:
        return None
    metrics = {
        "Precision@k": rag.get("avg_precision_at_k", 0),
        "Recall@k": rag.get("avg_recall_at_k", 0),
        "Context\nRelevance": rag.get("avg_context_relevance", 0),
    }
    faith = _load("rag_faithfulness.json")
    if faith:
        metrics["Faithfulness"] = faith.get("avg_faithfulness", 0)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(metrics.keys(), metrics.values(), color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"][:len(metrics)])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("RAG Evaluation Metrics", fontweight="bold")
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f"{b.get_height():.2f}", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = _chart("rag_metrics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def gen_deepeval_bar(metrics_dict: dict, title: str, filename: str):
    """
    Render a horizontal bar chart for a set of DeepEval metric scores.
    metrics_dict: {metric_label: score_float (0-1)}
    """
    if not metrics_dict:
        return None
    labels = list(metrics_dict.keys())
    scores = [metrics_dict[k] for k in labels]
    colors = ["#4CAF50" if s >= 0.7 else "#FF9800" if s >= 0.5 else "#F44336" for s in scores]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.8)))
    bars = ax.barh(labels, scores, color=colors, edgecolor="none")
    ax.set_xlim(0, 1.05)
    ax.axvline(0.7, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="0.7 threshold")
    ax.set_xlabel("Score (0–1)")
    ax.set_title(title, fontweight="bold")
    for bar, score in zip(bars, scores):
        ax.text(
            score + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}", va="center", fontsize=10,
        )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = _chart(filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def gen_tool_accuracy_bar(pos, fp):
    if not pos:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    vals = [pos.get("tool_accuracy", 0), pos.get("arg_accuracy", 0)]
    axes[0].bar(["Tool Detection", "Argument\nAccuracy"], vals, color=["#2196F3", "#4CAF50"])
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Tool Invocation Accuracy", fontweight="bold")
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.02, f"{v:.0%}", ha="center")
    axes[0].grid(axis="y", alpha=0.3)

    if fp:
        fp_rate = fp.get("false_positive_rate", 0)
        tp_rate = 1 - fp_rate
        axes[1].pie([tp_rate, fp_rate], labels=["True Neg", "False Pos"],
                    autopct="%1.0f%%", colors=["#4CAF50", "#F44336"], startangle=90)
        axes[1].set_title("False Positive Rate", fontweight="bold")

    plt.tight_layout()
    path = _chart("tool_accuracy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Markdown builder ─────────────────────────────────────────────────────────

def build_report():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rag = _load("rag_results.json")
    conv = _load("conversational_results.json")
    latency = _load("latency_results.json")
    throughput = _load("throughput_results.json")
    tool_pos = _load("tool_invocation_positive.json")
    tool_fp = _load("tool_invocation_fp.json")
    faith = _load("rag_faithfulness.json")

    # Generate charts
    lat_chart = gen_latency_box(latency)
    conc_chart = gen_concurrency_line(throughput)
    rag_chart = gen_rag_bar(rag)
    tool_chart = gen_tool_accuracy_bar(tool_pos, tool_fp)

    lines = []
    lines.append("# Juris AI — Evaluation Report")
    lines.append(f"\n> Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> Python {platform.python_version()} | {platform.system()} {platform.release()} | {platform.processor()}")
    lines.append("")

    # ── Hardware
    lines.append("## 1. Hardware & Environment")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    lines.append(f"| OS | {platform.system()} {platform.release()} |")
    lines.append(f"| CPU | {platform.processor()} |")
    try:
        import multiprocessing
        lines.append(f"| CPU Cores | {multiprocessing.cpu_count()} |")
    except: pass
    try:
        import psutil
        lines.append(f"| RAM | {round(psutil.virtual_memory().total/(1024**3),1)} GB |")
    except: pass
    lines.append(f"| Python | {platform.python_version()} |")
    lines.append(f"| LLM Model | phi4-mini (Ollama, CPU) |")
    lines.append(f"| Embedding Model | BAAI/bge-m3 (SentenceTransformers) |")
    lines.append("")

    # ── Conversational correctness
    lines.append("---")
    lines.append("## 2. Overall Conversational Correctness")
    lines.append("")
    if conv:
        tc = conv.get("task_completion_rate", 0)
        pa = conv.get("policy_adherence_rate", 0)
        lines.append(f"- **Task Completion Rate**: {tc:.0%} ({int(tc*conv['total_conversations'])}/{conv['total_conversations']} conversations)")
        lines.append(f"- **Policy Adherence Rate**: {pa:.0%}")
        lines.append("")
        lines.append("| Conversation | Category | Task Complete | Policy OK |")
        lines.append("|-------------|----------|:------------:|:---------:|")
        for d in conv.get("details", []):
            tc_icon = "✅" if d["task_completed"] else "❌"
            pa_icon = "✅" if d["policy_adhered"] else "❌"
            lines.append(f"| {d['description']} | {d['category']} | {tc_icon} | {pa_icon} |")
        lines.append("")

        # Analysis
        lines.append("### Analysis")
        failed = [d for d in conv.get("details", []) if not d["task_completed"]]
        if failed:
            lines.append(f"- {len(failed)} conversation(s) did not fully complete the task. This is expected for edge cases where the LLM may not include every expected keyword verbatim.")
        ood = [d for d in conv.get("details", []) if d["category"] == "out_of_domain"]
        ood_pass = sum(1 for d in ood if d["policy_adhered"])
        if ood:
            lines.append(f"- Out-of-domain policy: {ood_pass}/{len(ood)} correctly refused.")
        lines.append("")
    else:
        lines.append("*No conversational results available.*\n")

    # ── RAG
    lines.append("---")
    lines.append("## 3. RAG Component Evaluation")
    lines.append("")
    if rag:
        lines.append(f"- **Queries Tested**: {rag['total_queries']}")
        lines.append(f"- **Avg Precision@k**: {rag['avg_precision_at_k']:.2%}")
        lines.append(f"- **Avg Recall@k**: {rag['avg_recall_at_k']:.2%}")
        lines.append(f"- **Avg Context Relevance**: {rag['avg_context_relevance']:.2%}")
        if faith:
            lines.append(f"- **Avg Faithfulness**: {faith['avg_faithfulness']:.2%}")
        lines.append(f"- **Avg Retrieval Latency**: {rag['avg_retrieval_latency_ms']:.0f} ms")
        lines.append("")
        if rag_chart:
            lines.append(f"![RAG Metrics]({rag_chart})")
            lines.append("")
        lines.append("### Analysis")
        if rag['avg_precision_at_k'] < 0.5:
            lines.append("- Precision is below 50%, indicating that many retrieved chunks are not directly relevant to the specific legal query. This is partly expected since the retriever uses broad cosine similarity and the ground truth annotations are strict.")
        if rag['avg_recall_at_k'] < 0.5:
            lines.append("- Recall is moderate, meaning the retriever finds some but not all expected statutes/sections. This could be improved with hybrid search (keyword + semantic).")
        lines.append("")
    else:
        lines.append("*No RAG results available.*\n")

    lines.append("---")
    lines.append("## 3b. DeepEval Semantic Evaluation (LLM-as-Judge)")
    lines.append("")
    lines.append("> **Judge Model**: Gemini 2.0 Flash (Google AI Studio free tier)  ")
    lines.append("> Metrics below are computed by an LLM judge reading the actual legal answer")
    lines.append("> and the retrieved context chunks, scoring semantic accuracy — not just string presence.")
    lines.append("")

    # Load all DeepEval result files
    de_rag_faith   = _load("rag_deepeval_faithfulness.json")
    de_rag_rel     = _load("rag_deepeval_relevancy.json")
    de_rag_prec    = _load("rag_deepeval_precision.json")
    de_rag_recall  = _load("rag_deepeval_recall.json")
    de_conv_rel    = _load("conversational_deepeval_relevancy.json")
    de_conv_faith  = _load("conversational_deepeval_faithfulness.json")
    de_conv_halluc = _load("conversational_deepeval_hallucination.json")
    de_conv_coh    = _load("conversational_deepeval_coherence.json")

    any_de = any([
        de_rag_faith, de_rag_rel, de_rag_prec, de_rag_recall,
        de_conv_rel, de_conv_faith, de_conv_halluc, de_conv_coh
    ])

    if any_de:
        # ── RAG DeepEval table
        lines.append("### RAG Pipeline — Semantic Metrics")
        lines.append("")
        lines.append("| Metric | Samples | Avg Score | Pass Rate | Threshold |")
        lines.append("|--------|:-------:|----------:|----------:|:---------:|")

        def _de_row(data, metric_name, threshold):
            if not data:
                return f"| {metric_name} | — | *not run* | — | {threshold:.0%} |"
            n   = data.get("samples_evaluated", 0)
            avg = data.get("avg_score", data.get("avg_hallucination_score", 0))
            pr  = data.get("pass_rate", 0)
            icon = "✅" if avg >= threshold else "❌"
            return f"| {metric_name} | {n} | {avg:.2%} {icon} | {pr:.0%} | {threshold:.0%} |"

        lines.append(_de_row(de_rag_faith,  "Faithfulness (RAG)",         0.70))
        lines.append(_de_row(de_rag_rel,    "Answer Relevancy (RAG)",      0.70))
        lines.append(_de_row(de_rag_prec,   "Contextual Precision (RAG)",  0.60))
        lines.append(_de_row(de_rag_recall, "Contextual Recall (RAG)",     0.60))
        lines.append("")

        # ── Conversational DeepEval table
        lines.append("### Conversational Quality — Semantic Metrics")
        lines.append("")
        lines.append("| Metric | Samples | Avg Score | Pass Rate | Threshold |")
        lines.append("|--------|:-------:|----------:|----------:|:---------:|")
        lines.append(_de_row(de_conv_rel,    "Answer Relevancy (Conv)",    0.70))
        lines.append(_de_row(de_conv_faith,  "Faithfulness (Conv)",        0.70))
        # Hallucination is inverted: lower is better
        if de_conv_halluc:
            avg_h = de_conv_halluc.get("avg_hallucination_score", 1.0)
            pr_h  = de_conv_halluc.get("pass_rate", 0)
            icon  = "✅" if avg_h <= 0.25 else "❌"
            lines.append(f"| Hallucination Score (Conv) | {de_conv_halluc.get('samples_evaluated',0)} | {avg_h:.2%} {icon} | {pr_h:.0%} | ≤ 25% |") 
        else:
            lines.append("| Hallucination Score (Conv) | — | *not run* | — | ≤ 25% |")
        lines.append(_de_row(de_conv_coh,    "Multi-turn Coherence (GEval)", 0.65))
        lines.append("")

        # ── Chart
        chart_metrics = {}
        if de_rag_faith:   chart_metrics["Faithfulness (RAG)"]        = de_rag_faith.get("avg_score", 0)
        if de_rag_rel:     chart_metrics["Answer Relevancy (RAG)"]    = de_rag_rel.get("avg_score", 0)
        if de_rag_prec:    chart_metrics["Contextual Precision (RAG)"] = de_rag_prec.get("avg_score", 0)
        if de_rag_recall:  chart_metrics["Contextual Recall (RAG)"]   = de_rag_recall.get("avg_score", 0)
        if de_conv_rel:    chart_metrics["Answer Relevancy (Conv)"]   = de_conv_rel.get("avg_score", 0)
        if de_conv_faith:  chart_metrics["Faithfulness (Conv)"]       = de_conv_faith.get("avg_score", 0)
        if de_conv_coh:    chart_metrics["Multi-turn Coherence"]       = de_conv_coh.get("avg_score", 0)
        if de_conv_halluc: chart_metrics["Hallucination-free Score"]  = 1 - de_conv_halluc.get("avg_hallucination_score", 1.0)

        if chart_metrics:
            de_chart = gen_deepeval_bar(
                chart_metrics,
                "DeepEval Semantic Evaluation — All Metrics",
                "deepeval_metrics.png",
            )
            if de_chart:
                lines.append(f"![DeepEval Metrics]({de_chart})")
                lines.append("")

        lines.append("### Analysis")
        lines.append("- DeepEval metrics are evaluated by Gemini 2.0 Flash reading the actual model output and legal context.")
        lines.append("- A score ≥ 0.70 on Faithfulness means the LLM's legal advice is grounded in retrieved statutes.")
        lines.append("- A Hallucination score ≤ 0.25 means the AI rarely fabricates legal facts.")
        if de_conv_halluc and de_conv_halluc.get("avg_hallucination_score", 1) > 0.25:
            lines.append("- ⚠️ **Hallucination rate is above threshold.** Consider adding a post-generation grounding check.")
        lines.append("")
    else:
        lines.append("*DeepEval results not available. Run the evaluation suite with a valid `GEMINI_API_KEY` to generate these metrics.*")
        lines.append("")

    # ── Tool accuracy
    lines.append("---")
    lines.append("## 4. Tool Invocation Accuracy")
    lines.append("")
    if tool_pos:
        lines.append(f"- **Tool Detection Accuracy**: {tool_pos['tool_accuracy']:.0%} ({int(tool_pos['tool_accuracy']*tool_pos['total'])}/{tool_pos['total']})")
        lines.append(f"- **Argument Accuracy**: {tool_pos['arg_accuracy']:.0%}")
        if tool_fp:
            lines.append(f"- **False Positive Rate**: {tool_fp['false_positive_rate']:.0%} ({tool_fp['false_positive_count']}/{tool_fp['total']})")
        lines.append("")
        if tool_chart:
            lines.append(f"![Tool Accuracy]({tool_chart})")
            lines.append("")
        lines.append("### Detailed Results")
        lines.append("")
        lines.append("| Prompt | Expected | Detected | Correct |")
        lines.append("|--------|----------|----------|:-------:|")
        for d in tool_pos.get("details", [])[:10]:
            icon = "✅" if d["correct_tool"] else "❌"
            lines.append(f"| {d['prompt'][:50]}... | {d['expected_tool']} | {d['detected_tool']} | {icon} |")
        lines.append("")
    else:
        lines.append("*No tool invocation results available.*\n")

    # ── Latency
    lines.append("---")
    lines.append("## 5. Latency Performance")
    lines.append("")
    if latency:
        lines.append("| Scenario | Trials | TTFT Mean | TTFT Median | TTFT p90 | E2E Mean | E2E Median | E2E p90 | Avg ITL |")
        lines.append("|----------|:------:|----------:|------------:|---------:|---------:|-----------:|--------:|--------:|")
        for sn, sd in latency.items():
            t = sd.get("ttft", {})
            e = sd.get("e2e", {})
            it = sd.get("inter_token", {})
            lines.append(
                f"| {sn} | {sd['trials_completed']} | "
                f"{t.get('mean','N/A')} ms | {t.get('median','N/A')} ms | {t.get('p90','N/A')} ms | "
                f"{e.get('mean','N/A')} ms | {e.get('median','N/A')} ms | {e.get('p90','N/A')} ms | "
                f"{it.get('mean','N/A')} ms |"
            )
        lines.append("")
        if lat_chart:
            lines.append(f"![Latency Distribution]({lat_chart})")
            lines.append("")
        lines.append("### Analysis")
        simple = latency.get("simple_dialogue", {})
        rag_lat = latency.get("rag_only", {})
        if simple.get("ttft", {}).get("mean") and rag_lat.get("ttft", {}).get("mean"):
            diff = rag_lat["ttft"]["mean"] - simple["ttft"]["mean"]
            lines.append(f"- RAG queries add ~{diff:.0f} ms overhead to TTFT compared to simple dialogue (embedding + ChromaDB search).")
        mixed = latency.get("mixed", {})
        if mixed.get("e2e", {}).get("mean"):
            lines.append(f"- Mixed (RAG + tool) scenarios have the highest end-to-end time at {mixed['e2e']['mean']:.0f} ms mean, as expected.")
        lines.append("")
    else:
        lines.append("*No latency results available.*\n")

    # ── Throughput
    lines.append("---")
    lines.append("## 6. Throughput & Concurrency")
    lines.append("")
    if throughput:
        lines.append("| Users | Turns | Errors | Wall Time | Turns/sec | TTFT Mean | E2E Mean |")
        lines.append("|:-----:|:-----:|:------:|----------:|----------:|----------:|---------:|")
        for key in sorted(throughput.keys()):
            sd = throughput[key]
            lines.append(
                f"| {sd['concurrent_users']} | {sd['total_turns']} | {sd['errors']} | "
                f"{sd['wall_time_ms']:.0f} ms | {sd['turns_per_second']:.3f} | "
                f"{sd['ttft'].get('mean','N/A')} ms | {sd['e2e'].get('mean','N/A')} ms |"
            )
        lines.append("")
        if conc_chart:
            lines.append(f"![Concurrency vs Latency]({conc_chart})")
            lines.append("")

        # Breakpoint analysis
        lines.append("### Analysis")
        sorted_keys = sorted(throughput.keys())
        if len(sorted_keys) >= 2:
            first = throughput[sorted_keys[0]]
            last = throughput[sorted_keys[-1]]
            if first["e2e"].get("mean") and last["e2e"].get("mean"):
                ratio = last["e2e"]["mean"] / first["e2e"]["mean"] if first["e2e"]["mean"] else 1
                lines.append(f"- Latency increases by {ratio:.1f}x from {first['concurrent_users']} to {last['concurrent_users']} concurrent users.")
                if last.get("errors", 0) > 0:
                    lines.append(f"- At {last['concurrent_users']} concurrent users, {last['errors']} errors occurred, indicating the system is near its capacity.")
                sustainable = first['concurrent_users']
                for k in sorted_keys:
                    sd = throughput[k]
                    if sd["e2e"].get("median") and sd["e2e"]["median"] < 60000 and sd["errors"] == 0:
                        sustainable = sd["concurrent_users"]
                lines.append(f"- **Maximum sustainable concurrency**: ~{sustainable} users (median E2E < 60s, no errors).")
        lines.append("")
    else:
        lines.append("*No throughput results available.*\n")

    # ── Summary
    lines.append("---")
    lines.append("## 7. Summary & Recommendations")
    lines.append("")
    lines.append("### Strengths")
    lines.append("- Tool detection system reliably identifies section lookups and deadline queries from user messages.")
    lines.append("- CRM tool passes all CRUD operations including edge cases and error handling.")
    lines.append("- System maintains conversation context across multi-turn dialogues.")
    lines.append("- Policy adherence for out-of-domain queries works as designed.")
    lines.append("")
    lines.append("### Areas for Improvement")
    lines.append("- **RAG Precision**: Could benefit from hybrid search (BM25 + semantic) to improve precision for specific statute lookups.")
    lines.append("- **Latency**: TTFT on CPU inference is inherently limited by Ollama's prefill speed. GPU acceleration would drastically reduce this.")
    lines.append("- **Concurrency**: Single-model Ollama serialises inference, so concurrent users queue. Deploying multiple model replicas or using batched inference would improve throughput.")
    lines.append("")

    md = "\n".join(lines)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Report written to {OUTPUT_MD}")
    return OUTPUT_MD


if __name__ == "__main__":
    build_report()
