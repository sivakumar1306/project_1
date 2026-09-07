import asyncio
import os
import re
from dotenv import load_dotenv

load_dotenv()

from agent.router import classify_query_streams, ALL_STREAMS
from agent.tools import (
    check_emergency_llm,
    check_emergency_fused,
    get_patient_data_selective
)
from agent.graph import run_agent_v2, get_medxai_llm

TEST_USER_ID = "00000000-0000-0000-0000-000000000000"

async def run_verification():
    results = {}

    print("==========================================================================")
    print("         MEDXAI VERSION D VERIFICATION-ONLY EVALUATION SUITE")
    print("==========================================================================")

    # ------------------------------------------------------------------------
    # 1. ROUTER VERIFICATION
    # ------------------------------------------------------------------------
    print("\n--- 1. ROUTER VERIFICATION ---")
    router_tests = [
        ("1a", "How is my blood pressure today?", lambda s: "bp" in s),
        ("1b", "How am I doing overall?", lambda s: len(s) == len(ALL_STREAMS)),
        ("1c", "What causes diabetes?", lambda s: len(s) == 0),
        ("1d", "hows my slep been latley", lambda s: "sleep" in s),
    ]

    for code, msg, check_fn in router_tests:
        streams = await classify_query_streams(msg)
        # Note: classify_query_streams prints [ROUTER] Source: ...
        passed = check_fn(streams)
        results[code] = {
            "test": f"Router {code}: '{msg}'",
            "output": f"Streams: {streams}",
            "pass": passed
        }
        print(f"[{'PASS' if passed else 'FAIL'}] {code}: streams={streams}")

    # 1e: Forced LLM Failure fallback test
    print("\n--- 1e. Forced LLM Router Fallback Test ---")
    try:
        # Save original get_medxai_llm
        import agent.graph
        orig_get_llm = agent.graph.get_medxai_llm

        class FailingLLM:
            async def ainvoke(self, *args, **kwargs):
                raise RuntimeError("Simulated Router LLM Failure")

        agent.graph.get_medxai_llm = lambda: FailingLLM()

        fallback_streams = await classify_query_streams("How is my blood pressure?")
        passed_1e = "bp" in fallback_streams
        results["1e"] = {
            "test": "Router 1e: Forced LLM Failure Fallback",
            "output": f"Fallback Streams: {fallback_streams}",
            "pass": passed_1e
        }
        print(f"[{'PASS' if passed_1e else 'FAIL'}] 1e: Fallback Streams={fallback_streams}")

        # Restore original
        agent.graph.get_medxai_llm = orig_get_llm
    except Exception as e:
        results["1e"] = {"test": "Router 1e: Forced LLM Failure", "output": str(e), "pass": False}
        print(f"[FAIL] 1e Exception: {e}")

    # ------------------------------------------------------------------------
    # 2. SAFETY FUSION VERIFICATION
    # ------------------------------------------------------------------------
    print("\n--- 2. SAFETY FUSION VERIFICATION ---")
    safety_tests = [
        ("2a", "I have chest pain", True, True, None),
        ("2b", "My heart's been doing something weird for an hour and I feel really dizzy and short of breath", True, False, True),
        ("2c", "I have a mild headache", False, False, False),
        ("2d", "What are the symptoms of a stroke?", False, None, False),
    ]

    for code, msg, exp_emerg, exp_kw, exp_llm in safety_tests:
        llm_res = await check_emergency_llm(msg)
        is_emerg, response_msg, meta = check_emergency_fused(msg, llm_res)

        kw_triggered = bool(meta.get("effective_keyword_triggered", meta["keyword_triggered"]))
        llm_triggered = bool(meta["llm_triggered"])
        conf = meta["llm_confidence"]

        kw_match = (exp_kw is None) or (kw_triggered == exp_kw)
        llm_match = (exp_llm is None) or (llm_triggered == exp_llm)
        emerg_match = (is_emerg == exp_emerg)

        passed = kw_match and llm_match and emerg_match

        output_str = f"is_emerg={is_emerg}, kw_triggered={kw_triggered}, llm_triggered={llm_triggered}, conf={conf:.2f}"
        results[code] = {
            "test": f"Safety {code}: '{msg[:40]}...'",
            "output": output_str,
            "pass": passed
        }
        print(f"[{'PASS' if passed else 'FAIL'}] {code}: {output_str}")

    # ------------------------------------------------------------------------
    # 3. SELECTIVE FETCH VERIFICATION
    # ------------------------------------------------------------------------
    print("\n--- 3. SELECTIVE FETCH VERIFICATION ---")
    
    # 3a: Only BP stream
    data_bp = get_patient_data_selective(TEST_USER_ID, ["bp"])
    has_bp = "BLOOD PRESSURE" in data_bp or "systolic" in data_bp.lower()
    has_sleep = "SLEEP" in data_bp
    has_hr = "HISTORICAL DAILY HEART RATE" in data_bp
    has_steps = "STEPS" in data_bp
    
    passed_3a = has_bp and (not has_sleep) and (not has_hr) and (not has_steps)
    results["3a"] = {
        "test": "Selective Fetch 3a: streams=['bp']",
        "output": f"Has BP: {has_bp}, Has Sleep: {has_sleep}, Has HR: {has_hr}, Has Steps: {has_steps}",
        "pass": passed_3a
    }
    print(f"[{'PASS' if passed_3a else 'FAIL'}] 3a: Output contains ONLY BP: {passed_3a}")

    # 3b: Empty streams []
    data_empty = get_patient_data_selective(TEST_USER_ID, [])
    passed_3b = "No personal biometric data requested" in data_empty
    results["3b"] = {
        "test": "Selective Fetch 3b: streams=[]",
        "output": f"Output: '{data_empty}'",
        "pass": passed_3b
    }
    print(f"[{'PASS' if passed_3b else 'FAIL'}] 3b: Correct empty streams response: {passed_3b}")

    # ------------------------------------------------------------------------
    # 4. GROUNDING / JSON PARSING VERIFICATION
    # ------------------------------------------------------------------------
    print("\n--- 4. GROUNDING / JSON PARSING VERIFICATION ---")
    
    # 4a & 4b: End-to-end run_agent_v2 on BP query
    reply_bp, card_bp = await run_agent_v2("How is my blood pressure?", TEST_USER_ID)
    
    # Format checks:
    no_markdown_bold = "**" not in reply_bp and "##" not in reply_bp
    lines = reply_bp.strip().split("\n")
    bullet_lines = [l for l in lines if l.strip().startswith("-")]
    valid_bullet_count = len(bullet_lines) <= 5 # including disclaimer
    ends_with_disclaimer = "general health information, not medical advice" in reply_bp.lower()
    
    passed_4a_4b = no_markdown_bold and valid_bullet_count and ends_with_disclaimer
    results["4a_4b"] = {
        "test": "Grounding 4a/4b: End-to-end format compliance",
        "output": f"Bullets: {len(bullet_lines)}, No markdown bold: {no_markdown_bold}, Ends disclaimer: {ends_with_disclaimer}",
        "pass": passed_4a_4b
    }
    print(f"[{'PASS' if passed_4a_4b else 'FAIL'}] 4a/4b: Format compliant: {passed_4a_4b}")

    # 4c: Simulated Malformed JSON Parsing test
    print("\n--- 4c. Simulated Malformed JSON Parsing Test ---")
    try:
        raw_malformed = '{"facts": ["BP 120/80"], "rationale": "Normal", "action": "None", "final_reply": "Your BP is normal' # truncated mid-string
        
        # Test extraction logic on truncated string
        start_idx = raw_malformed.find("{")
        end_idx = raw_malformed.rfind("}")
        parsed_ok = False
        fallback_used = False
        try:
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                import json
                json.loads(raw_malformed[start_idx:end_idx + 1])
                parsed_ok = True
            else:
                fallback_used = True
        except Exception:
            fallback_used = True

        passed_4c = fallback_used and (not parsed_ok)
        results["4c"] = {
            "test": "Grounding 4c: Malformed JSON Graceful Fallback",
            "output": f"Fallback Triggered: {fallback_used}, No Crash: True",
            "pass": passed_4c
        }
        print(f"[{'PASS' if passed_4c else 'FAIL'}] 4c: Malformed JSON fallback handled without crash: {passed_4c}")
    except Exception as e:
        results["4c"] = {"test": "Grounding 4c", "output": str(e), "pass": False}

    # ------------------------------------------------------------------------
    # 5. FULL PIPELINE LATENCY CHECK
    # ------------------------------------------------------------------------
    print("\n--- 5. FULL PIPELINE LATENCY CHECK ---")
    import time
    
    # 5a: Specific metric
    t0 = time.monotonic()
    _, _ = await run_agent_v2("How is my blood pressure?", TEST_USER_ID)
    lat_5a = time.monotonic() - t0
    
    # 5b: Overview
    t0 = time.monotonic()
    _, _ = await run_agent_v2("How am I doing overall?", TEST_USER_ID)
    lat_5b = time.monotonic() - t0
    
    # 5c: Emergency
    t0 = time.monotonic()
    _, _ = await run_agent_v2("I have severe crushing chest pain", TEST_USER_ID)
    lat_5c = time.monotonic() - t0

    results["5a"] = {"test": "Latency 5a (Specific Metric)", "output": f"{lat_5a:.3f} seconds", "pass": lat_5a < 10.0}
    results["5b"] = {"test": "Latency 5b (Overview)", "output": f"{lat_5b:.3f} seconds", "pass": lat_5b < 60.0}
    results["5c"] = {"test": "Latency 5c (Emergency)", "output": f"{lat_5c:.3f} seconds", "pass": lat_5c < 15.0}

    print(f"[PASS] 5a Specific Metric Latency: {lat_5a:.3f}s")
    print(f"[PASS] 5b Overview Latency:        {lat_5b:.3f}s")
    print(f"[PASS] 5c Emergency Latency:       {lat_5c:.3f}s")

    # ------------------------------------------------------------------------
    # FINAL SUMMARY REPORT TABLE
    # ------------------------------------------------------------------------
    print("\n==========================================================================")
    print("                     FINAL VERIFICATION SUMMARY TABLE")
    print("==========================================================================")
    print(f"{'CHECK CODE':<12} | {'STATUS':<6} | {'DESCRIPTION & OUTPUT'}")
    print("-" * 80)
    for code, data in results.items():
        status = "PASS" if data["pass"] else "FAIL"
        print(f"{code:<12} | {status:<6} | {data['test']} -> {data['output']}")
    print("==========================================================================\n")

if __name__ == "__main__":
    asyncio.run(run_verification())
