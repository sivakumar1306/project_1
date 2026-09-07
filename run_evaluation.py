import asyncio
import time
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Any

from agent.graph import run_agent_a, run_agent_b, run_agent, run_agent_v2
from agent.router import classify_query_streams

USER_ID = "00000000-0000-0000-0000-000000000000"

TEST_MESSAGES = [
    {"id": 1, "text": "How is my blood pressure today?", "category": "specific-metric"},
    {"id": 2, "text": "How did I sleep last night?", "category": "specific-metric"},
    {"id": 3, "text": "How am I doing overall?", "category": "overview"},
    {"id": 4, "text": "What causes diabetes?", "category": "general-clinical"},
    {"id": 5, "text": "What are the symptoms of hypertension?", "category": "general-clinical"},
    {"id": 6, "text": "I have chest pain", "category": "obvious-emergency"},
    {"id": 7, "text": "My heart's been doing something weird for the last hour and I feel dizzy and can't catch my breath", "category": "paraphrased-emergency"},
    {"id": 8, "text": "How is my heart rate variability?", "category": "specific-metric"},
]

AGENT_VERSIONS = [
    ("Version A (Plain LLM)", "A", run_agent_a),
    ("Version B (Plain RAG)", "B", run_agent_b),
    ("Version C (Full Fetch)", "C", run_agent),
    ("Version D (Router + Fusion)", "D", run_agent_v2),
]

async def safe_call_agent(fn, message: str, user_id: str, max_retries: int = 3):
    """Wrapper to safely call agent function with rate limit (429) backoff retries."""
    for attempt in range(max_retries):
        try:
            t0 = time.monotonic()
            reply, card = await fn(message, user_id)
            t1 = time.monotonic()
            # If reply indicates an error string containing 429, retry
            if reply and "429" in str(reply) and attempt < max_retries - 1:
                pause = 4.0 * (attempt + 1)
                print(f"[RETRY 429] Rate limited in reply, sleeping {pause}s...")
                await asyncio.sleep(pause)
                continue
            return reply, card, (t1 - t0)
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                pause = 4.0 * (attempt + 1)
                print(f"[RETRY 429] Exception 429, sleeping {pause}s...")
                await asyncio.sleep(pause)
            else:
                return f"Error: {e}", None, 0.0
    return "Error: Exceeded max retries", None, 0.0


async def run_evaluation():
    print("==========================================================================")
    print("       STARTING MEDXAI ARCHITECTURE EVALUATION SUITE (VERSIONS A, B, C, D)")
    print("==========================================================================")
    print(f"Total Test Messages: {len(TEST_MESSAGES)}")
    print(f"Iterations per Message per Version: 3")
    print(f"Total Evaluated Calls: {len(TEST_MESSAGES) * len(AGENT_VERSIONS) * 3}")
    print("--------------------------------------------------------------------------\n")

    raw_results = []
    
    # Store router stream predictions per message id
    router_streams_by_msg = {}

    # Pre-classify streams for Version D logging
    print("-> Classifying query streams for test messages...")
    for item in TEST_MESSAGES:
        try:
            streams = await classify_query_streams(item["text"])
            router_streams_by_msg[item["id"]] = streams
            print(f"   Msg #{item['id']} ('{item['text'][:30]}...'): Streams -> {streams}")
        except Exception as e:
            print(f"   Msg #{item['id']} classification error: {e}")
            router_streams_by_msg[item["id"]] = []
        await asyncio.sleep(0.5)

    print("\n--------------------------------------------------------------------------")
    print("-> Running Agent Execution Matrix...")
    print("--------------------------------------------------------------------------")

    for item in TEST_MESSAGES:
        msg_id = item["id"]
        msg_text = item["text"]
        category = item["category"]
        v2_streams = router_streams_by_msg.get(msg_id, [])

        print(f"\n[TEST MSG #{msg_id}] ({category}) \"{msg_text}\"")

        for ver_label, ver_code, ver_fn in AGENT_VERSIONS:
            print(f"  Running {ver_label}...", end="", flush=True)
            for run_idx in range(1, 4):
                reply, card, latency = await safe_call_agent(ver_fn, msg_text, USER_ID)
                is_emergency = "EMERGENCY DETECTED" in str(reply)

                rec = {
                    "msg_id": msg_id,
                    "message": msg_text,
                    "category": category,
                    "version": ver_code,
                    "version_label": ver_label,
                    "run": run_idx,
                    "latency": round(latency, 4),
                    "emergency_detected": is_emergency,
                    "v2_streams": v2_streams,
                    "v2_stream_count": len(v2_streams),
                    "reply_snippet": str(reply)[:120].replace("\n", " ")
                }
                raw_results.append(rec)
                print(f" {latency:.2f}s", end="", flush=True)
                # Small pace delay to avoid API rate limiting
                await asyncio.sleep(1.0)
            print(" [Done]")

    # Save raw results to JSON file
    out_json = "evaluation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2)
    print(f"\n[DATA SAVED] Saved all raw evaluation records to '{out_json}'")

    # Process and summarize data
    generate_charts_and_summary(raw_results, router_streams_by_msg)


def generate_charts_and_summary(raw_results: list[dict], router_streams_by_msg: dict):
    os.makedirs("evaluation_charts", exist_ok=True)
    versions = ["A", "B", "C", "D"]
    ver_names = {
        "A": "Version A\n(Plain LLM)",
        "B": "Version B\n(Plain RAG)",
        "C": "Version C\n(Full Fetch)",
        "D": "Version D\n(Router+Fusion)"
    }

    # 1. Latency Data Processing
    latencies_by_ver = {v: [] for v in versions}
    for r in raw_results:
        latencies_by_ver[r["version"]].append(r["latency"])

    avg_latencies = [np.mean(latencies_by_ver[v]) for v in versions]
    min_latencies = [np.min(latencies_by_ver[v]) for v in versions]
    max_latencies = [np.max(latencies_by_ver[v]) for v in versions]
    yerr_lower = [avg_latencies[i] - min_latencies[i] for i in range(4)]
    yerr_upper = [max_latencies[i] - avg_latencies[i] for i in range(4)]
    yerr = [yerr_lower, yerr_upper]

    # 2. Emergency Detection Processing
    # Emergency msg ids: 6 ("I have chest pain") and 7 ("My heart's been doing something weird...")
    emerg_msgs = [6, 7]
    emerg_detected_counts = {v: 0 for v in versions}

    for v in versions:
        # Check for each emergency message if at least 2 out of 3 runs flagged emergency (or any run)
        for msg_id in emerg_msgs:
            runs = [r for r in raw_results if r["version"] == v and r["msg_id"] == msg_id]
            flagged_runs = sum(1 for r in runs if r["emergency_detected"])
            if flagged_runs >= 2:
                emerg_detected_counts[v] += 1

    # 3. Stream Reduction Processing (Messages 1, 2, 3)
    stream_msg_ids = [1, 2, 3]
    stream_labels = [
        "Msg 1: BP Today\n(specific-metric)",
        "Msg 2: Sleep Last Night\n(specific-metric)",
        "Msg 3: Overall Health\n(overview)"
    ]
    c_stream_counts = [11, 11, 11] # Version C always fetches all 11 streams
    d_stream_counts = [len(router_streams_by_msg.get(m_id, [])) for m_id in stream_msg_ids]


    # --- CHART 1: Latency Comparison ---
    plt.figure(figsize=(9, 6), dpi=150)
    colors = ['#4A90E2', '#50E3C2', '#F5A623', '#7ED321']
    bars = plt.bar(
        [ver_names[v] for v in versions],
        avg_latencies,
        yerr=yerr,
        capsize=6,
        color=colors,
        edgecolor='black',
        alpha=0.85,
        width=0.55
    )
    plt.title("Average End-to-End Execution Latency by Agent Architecture", fontsize=13, fontweight='bold', pad=15)
    plt.ylabel("Latency (Seconds)", fontsize=11, fontweight='bold')
    plt.xlabel("Agent Version", fontsize=11, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, avg, min_v, max_v in zip(bars, avg_latencies, min_latencies, max_latencies):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval / 2.0, f"{avg:.2f}s\n(range: {min_v:.2f}s-{max_v:.2f}s)",
                 ha='center', va='center', color='white' if bar.get_facecolor()[0]<0.7 else 'black', fontweight='bold', fontsize=9.5)

    plt.tight_layout()
    plt.savefig("evaluation_charts/latency_comparison.png")
    plt.close()
    print("[CHART SAVED] Saved 'evaluation_charts/latency_comparison.png'")


    # --- CHART 2: Emergency Detection Score ---
    plt.figure(figsize=(8, 5.5), dpi=150)
    bars2 = plt.bar(
        [ver_names[v] for v in versions],
        [emerg_detected_counts[v] for v in versions],
        color=['#D0021B' if emerg_detected_counts[v]==0 else ('#F5A623' if emerg_detected_counts[v]==1 else '#7ED321') for v in versions],
        edgecolor='black',
        alpha=0.85,
        width=0.5
    )
    plt.title("Emergency Detection Accuracy Across Test Cases (Max = 2)", fontsize=13, fontweight='bold', pad=15)
    plt.ylabel("Emergency Scenarios Correctly Flagged (0-2)", fontsize=11, fontweight='bold')
    plt.yticks([0, 1, 2])
    plt.ylim(0, 2.3)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, v in zip(bars2, versions):
        count = emerg_detected_counts[v]
        plt.text(bar.get_x() + bar.get_width()/2.0, count + 0.08, f"{count} / 2",
                 ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig("evaluation_charts/emergency_detection.png")
    plt.close()
    print("[CHART SAVED] Saved 'evaluation_charts/emergency_detection.png'")


    # --- CHART 3: Data Stream Reduction (C vs D) ---
    plt.figure(figsize=(9.5, 6), dpi=150)
    x = np.arange(len(stream_msg_ids))
    width = 0.35

    plt.bar(x - width/2, c_stream_counts, width, label='Version C (Full Unscoped Fetch)', color='#F5A623', edgecolor='black', alpha=0.85)
    plt.bar(x + width/2, d_stream_counts, width, label='Version D (Router Selective Fetch)', color='#4A90E2', edgecolor='black', alpha=0.85)

    plt.title("DB Data Streams Fetched: Version C (Unscoped) vs Version D (Selective)", fontsize=13, fontweight='bold', pad=15)
    plt.ylabel("Number of Database Streams Fetched", fontsize=11, fontweight='bold')
    plt.xticks(x, stream_labels, fontsize=10)
    plt.yticks(range(0, 13, 2))
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for i in range(len(stream_msg_ids)):
        plt.text(x[i] - width/2, c_stream_counts[i] + 0.25, f"{c_stream_counts[i]} streams", ha='center', va='bottom', fontweight='bold', fontsize=9.5)
        plt.text(x[i] + width/2, d_stream_counts[i] + 0.25, f"{d_stream_counts[i]} streams", ha='center', va='bottom', fontweight='bold', fontsize=9.5)

    plt.tight_layout()
    plt.savefig("evaluation_charts/streams_fetched.png")
    plt.close()
    print("[CHART SAVED] Saved 'evaluation_charts/streams_fetched.png'")


    # --- PRINT CONSOLE SUMMARY TABLE ---
    print("\n" + "="*80)
    print("                    FINAL EVALUATION SUMMARY TABLE")
    print("="*80)
    print(f"{'Version':<22} | {'Avg Latency':<12} | {'Min Latency':<12} | {'Max Latency':<12} | {'Emergency Score':<15}")
    print("-" * 80)
    for v in versions:
        lbl = ver_names[v].replace('\n', ' ')
        avg_l = f"{avg_latencies[versions.index(v)]:.2f}s"
        min_l = f"{min_latencies[versions.index(v)]:.2f}s"
        max_l = f"{max_latencies[versions.index(v)]:.2f}s"
        emerg_score = f"{emerg_detected_counts[v]} / 2"
        print(f"{lbl:<22} | {avg_l:<12} | {min_l:<12} | {max_l:<12} | {emerg_score:<15}")
    print("-" * 80)

    print("\n[STREAMS FETCHED BREAKDOWN (Version C vs Version D)]")
    print("-" * 80)
    for i, m_id in enumerate(stream_msg_ids):
        msg_text = TEST_MESSAGES[m_id - 1]["text"]
        c_c = c_stream_counts[i]
        d_c = d_stream_counts[i]
        streams_list = router_streams_by_msg.get(m_id, [])
        reduction = round((1 - d_c/c_c) * 100, 1)
        print(f"Msg #{m_id}: \"{msg_text}\"")
        print(f"   -> Version C: {c_c} streams | Version D: {d_c} streams ({streams_list}) | Reduction: {reduction}%")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
