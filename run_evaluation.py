import asyncio
import time
import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from typing import Any

from agent.graph import run_agent, run_agent_v2
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
    ("Baseline Agent (Fetch-All)", "C", run_agent),
    ("Coupled Routing-Safety Agent", "D", run_agent_v2),
]

consecutive_429_count = 0

def extract_429_details(err_msg: str) -> str:
    """Extract model and limit details from 429 rate limit error string."""
    m_str = re.search(r"model [`']?([^`'\s]+)[`']?", err_msg)
    l_str = re.search(r"Limit \d+[^,\.]*", err_msg)
    model = m_str.group(1) if m_str else "Unknown Model"
    limit = l_str.group(0) if l_str else "Rate limit hit"
    return f"Model: {model} | Detail: {limit}"

async def safe_call_agent(fn, message: str, user_id: str, max_retries: int = 3):
    """Wrapper to safely call agent function with rate limit (429) backoff retries and tracking."""
    global consecutive_429_count
    # Enforce minimum 2.5s delay before every LLM execution call to respect QPM limits
    await asyncio.sleep(2.5)
    
    for attempt in range(max_retries):
        try:
            t0 = time.monotonic()
            reply, card = await fn(message, user_id)
            t1 = time.monotonic()
            
            # Check if reply string indicates 429
            if reply and "429" in str(reply):
                details = extract_429_details(str(reply))
                print(f"\n[429 RATE LIMIT DETECTED] {details}")
                if attempt < max_retries - 1:
                    pause = 5.0 * (attempt + 1)
                    print(f"[RETRY 429] Sleeping {pause}s before retry...")
                    await asyncio.sleep(pause)
                    continue
                else:
                    consecutive_429_count += 1
                    return str(reply), None, 0.0

            # Success response
            consecutive_429_count = 0
            return reply, card, (t1 - t0)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                details = extract_429_details(err_str)
                print(f"\n[429 EXCEPTION DETECTED] {details}")
                if attempt < max_retries - 1:
                    pause = 5.0 * (attempt + 1)
                    print(f"[RETRY 429] Sleeping {pause}s before retry...")
                    await asyncio.sleep(pause)
                else:
                    consecutive_429_count += 1
                    return f"Error 429: {err_str}", None, 0.0
            else:
                return f"Error: {e}", None, 0.0
                
    consecutive_429_count += 1
    return "Error: Exceeded max retries (429 rate limit)", None, 0.0


async def run_evaluation():
    global consecutive_429_count
    print("==========================================================================")
    print("       STARTING MEDXAI ARCHITECTURE EVALUATION SUITE")
    print("       Baseline Agent (Fetch-All) vs Coupled Routing-Safety Agent")
    print("==========================================================================")
    print(f"Total Test Messages: {len(TEST_MESSAGES)}")
    print(f"Iterations per Message per Version: 3")
    print(f"Total Evaluated Calls: {len(TEST_MESSAGES) * len(AGENT_VERSIONS) * 3}")
    print("--------------------------------------------------------------------------\n")

    raw_results = []
    router_streams_by_msg = {}

    # Pre-classify streams for Version D logging
    print("-> Classifying query streams for test messages...")
    for item in TEST_MESSAGES:
        if consecutive_429_count >= 3:
            print("\n==========================================================================")
            print("ABORTING: Rate limit exhausted — do not generate charts from this data")
            print("==========================================================================")
            return

        try:
            await asyncio.sleep(2.5)
            streams = await classify_query_streams(item["text"])
            router_streams_by_msg[item["id"]] = streams
            print(f"   Msg #{item['id']} ('{item['text'][:30]}...'): Streams -> {streams}")
            consecutive_429_count = 0
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                details = extract_429_details(err_str)
                print(f"   Msg #{item['id']} classification 429 ERROR: {details}")
                consecutive_429_count += 1
            else:
                print(f"   Msg #{item['id']} classification error: {e}")
            router_streams_by_msg[item["id"]] = []

    print("\n--------------------------------------------------------------------------")
    print("-> Running Agent Execution Matrix...")
    print("--------------------------------------------------------------------------")

    aborted = False
    for item in TEST_MESSAGES:
        if aborted or consecutive_429_count >= 3:
            aborted = True
            break

        msg_id = item["id"]
        msg_text = item["text"]
        category = item["category"]
        v2_streams = router_streams_by_msg.get(msg_id, [])

        print(f"\n[TEST MSG #{msg_id}] ({category}) \"{msg_text}\"")

        for ver_label, ver_code, ver_fn in AGENT_VERSIONS:
            if consecutive_429_count >= 3:
                aborted = True
                break

            print(f"  Running {ver_label}...", end="", flush=True)
            for run_idx in range(1, 4):
                if consecutive_429_count >= 3:
                    aborted = True
                    break

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

            print(" [Done]")

    if aborted or consecutive_429_count >= 3:
        print("\n==========================================================================")
        print("ABORTING: Rate limit exhausted — do not generate charts from this data")
        print("==========================================================================")
        return

    # Save raw results to JSON file
    out_json = "evaluation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2)
    print(f"\n[DATA SAVED] Saved all raw evaluation records to '{out_json}'")

    # Process and summarize data
    generate_charts_and_summary(raw_results, router_streams_by_msg)


def generate_charts_and_summary(raw_results: list[dict], router_streams_by_msg: dict):
    os.makedirs("evaluation_charts", exist_ok=True)
    versions = ["C", "D"]
    ver_names = {
        "C": "Baseline Agent\n(Fetch-All)",
        "D": "Coupled Routing-Safety\nAgent"
    }

    # 1. Latency Data Processing
    latencies_by_ver = {v: [] for v in versions}
    for r in raw_results:
        if r["version"] in latencies_by_ver:
            latencies_by_ver[r["version"]].append(r["latency"])

    avg_latencies = [np.mean(latencies_by_ver[v]) if latencies_by_ver[v] else 0.0 for v in versions]
    min_latencies = [np.min(latencies_by_ver[v]) if latencies_by_ver[v] else 0.0 for v in versions]
    max_latencies = [np.max(latencies_by_ver[v]) if latencies_by_ver[v] else 0.0 for v in versions]
    yerr_lower = [avg_latencies[i] - min_latencies[i] for i in range(2)]
    yerr_upper = [max_latencies[i] - avg_latencies[i] for i in range(2)]
    yerr = [yerr_lower, yerr_upper]

    # 2. Emergency Detection Processing
    emerg_msgs = [6, 7]
    emerg_detected_counts = {v: 0 for v in versions}

    for v in versions:
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
    c_stream_counts = [11, 11, 11] # Baseline Agent (Fetch-All) always fetches all 11 streams
    d_stream_counts = [len(router_streams_by_msg.get(m_id, [])) for m_id in stream_msg_ids]


    # --- CHART 1: Latency Comparison ---
    plt.figure(figsize=(8, 5.5), dpi=150)
    colors = ['#F5A623', '#4A90E2']
    bars = plt.bar(
        [ver_names[v] for v in versions],
        avg_latencies,
        yerr=yerr,
        capsize=6,
        color=colors,
        edgecolor='black',
        alpha=0.85,
        width=0.45
    )
    plt.title("Average End-to-End Execution Latency", fontsize=13, fontweight='bold', pad=15)
    plt.ylabel("Latency (Seconds)", fontsize=11, fontweight='bold')
    plt.xlabel("Agent Architecture", fontsize=11, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, avg, min_v, max_v in zip(bars, avg_latencies, min_latencies, max_latencies):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval / 2.0, f"{avg:.2f}s\n(range: {min_v:.2f}s-{max_v:.2f}s)",
                 ha='center', va='center', color='white' if bar.get_facecolor()[0]<0.7 else 'black', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig("evaluation_charts/latency_comparison.png")
    plt.close()
    print("[CHART SAVED] Saved 'evaluation_charts/latency_comparison.png'")


    # --- CHART 2: Emergency Detection Score ---
    plt.figure(figsize=(7.5, 5), dpi=150)
    bars2 = plt.bar(
        [ver_names[v] for v in versions],
        [emerg_detected_counts[v] for v in versions],
        color=['#F5A623' if emerg_detected_counts[v]==1 else '#7ED321' for v in versions],
        edgecolor='black',
        alpha=0.85,
        width=0.4
    )
    plt.title("Emergency Detection Accuracy Across Test Scenarios (Max = 2)", fontsize=13, fontweight='bold', pad=15)
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


    # --- CHART 3: Data Stream Reduction ---
    plt.figure(figsize=(9, 5.5), dpi=150)
    x = np.arange(len(stream_msg_ids))
    width = 0.35

    plt.bar(x - width/2, c_stream_counts, width, label='Baseline Agent (Fetch-All)', color='#F5A623', edgecolor='black', alpha=0.85)
    plt.bar(x + width/2, d_stream_counts, width, label='Coupled Routing-Safety Agent', color='#4A90E2', edgecolor='black', alpha=0.85)

    plt.title("DB Data Streams Fetched: Baseline Agent (Fetch-All) vs Coupled Routing-Safety Agent", fontsize=12, fontweight='bold', pad=15)
    plt.ylabel("Number of Database Streams Fetched", fontsize=11, fontweight='bold')
    plt.xticks(x, stream_labels, fontsize=9.5)
    plt.yticks(range(0, 13, 2))
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for i in range(len(stream_msg_ids)):
        plt.text(x[i] - width/2, c_stream_counts[i] + 0.25, f"{c_stream_counts[i]} streams", ha='center', va='bottom', fontweight='bold', fontsize=9)
        plt.text(x[i] + width/2, d_stream_counts[i] + 0.25, f"{d_stream_counts[i]} streams", ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig("evaluation_charts/streams_fetched.png")
    plt.close()
    print("[CHART SAVED] Saved 'evaluation_charts/streams_fetched.png'")


    # --- PRINT CONSOLE SUMMARY TABLE ---
    print("\n" + "="*85)
    print("                         FINAL EVALUATION SUMMARY TABLE")
    print("="*85)
    print(f"{'Agent Architecture':<32} | {'Avg Latency':<11} | {'Min Latency':<11} | {'Max Latency':<11} | {'Emergency Score':<15}")
    print("-" * 85)
    for v in versions:
        lbl = "Baseline Agent (Fetch-All)" if v == "C" else "Coupled Routing-Safety Agent"
        avg_l = f"{avg_latencies[versions.index(v)]:.2f}s"
        min_l = f"{min_latencies[versions.index(v)]:.2f}s"
        max_l = f"{max_latencies[versions.index(v)]:.2f}s"
        emerg_score = f"{emerg_detected_counts[v]} / 2"
        print(f"{lbl:<32} | {avg_l:<11} | {min_l:<11} | {max_l:<11} | {emerg_score:<15}")
    print("-" * 85)

    print("\n[STREAMS FETCHED BREAKDOWN]")
    print("-" * 85)
    for i, m_id in enumerate(stream_msg_ids):
        msg_text = TEST_MESSAGES[m_id - 1]["text"]
        c_c = c_stream_counts[i]
        d_c = d_stream_counts[i]
        streams_list = router_streams_by_msg.get(m_id, [])
        reduction = round((1 - d_c/c_c) * 100, 1) if c_c > 0 else 0
        print(f"Msg #{m_id}: \"{msg_text}\"")
        print(f"   -> Baseline Agent (Fetch-All): {c_c} streams | Coupled Routing-Safety Agent: {d_c} streams ({streams_list}) | Reduction: {reduction}%")
    print("="*85 + "\n")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
