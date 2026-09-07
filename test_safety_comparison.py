import asyncio
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from agent.tools import check_emergency, check_emergency_llm, check_emergency_fused

TEST_MESSAGES = [
    "My heart's been doing something weird for the last hour and I feel dizzy and can't catch my breath",
    "I have chest pain",
    "What are the symptoms of a stroke?",
    "I have a mild headache",
    "I feel like something is really wrong, my chest feels tight and heavy and I'm sweating a lot"
]

async def main():
    old_fn = getattr(check_emergency, "func", check_emergency)
    
    diff_count = 0
    new_caught_more = 0
    new_avoided_false_alarms = 0
    
    print("=" * 80)
    print("SAFETY SYSTEM COMPARISON: OLD (Keyword Only) vs NEW (Keyword + LLM Fused)")
    print("=" * 80)
    
    for idx, msg in enumerate(TEST_MESSAGES, 1):
        # 1. Run OLD check
        old_output = old_fn(msg)
        old_is_emerg = "EMERGENCY DETECTED" in old_output
        old_status = "EMERGENCY" if old_is_emerg else "NOT EMERGENCY"
        
        # 2. Run NEW check
        llm_res = await check_emergency_llm(msg)
        is_emerg_new, resp_msg_new, meta = check_emergency_fused(msg, llm_res)
        new_status = "EMERGENCY" if is_emerg_new else "NOT EMERGENCY"
        
        llm_conf = meta.get("llm_confidence", 0.0)
        llm_reason = meta.get("llm_reason", "")
        
        # 3. Determine match / difference
        if old_is_emerg == is_emerg_new:
            match_diff = "Same result"
        elif not old_is_emerg and is_emerg_new:
            match_diff = "DIFFERENT - NEW caught what OLD missed"
            diff_count += 1
            new_caught_more += 1
        else:
            match_diff = "DIFFERENT - NEW avoided a false alarm that OLD raised"
            diff_count += 1
            new_avoided_false_alarms += 1
            
        print(f"\n{idx}. MESSAGE: \"{msg}\"")
        print(f"  OLD (keyword only):     {old_status}")
        print(f"  NEW (keyword + LLM):    {new_status}   (LLM confidence: {llm_conf:.2f}, reason: \"{llm_reason}\")")
        print(f"  MATCH OR DIFFERENCE:    {match_diff}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total test messages: {len(TEST_MESSAGES)}")
    print(f"Messages with different results: {diff_count}")
    print(f"  - Real emergencies caught by NEW (missed by OLD): {new_caught_more}")
    print(f"  - False alarms avoided by NEW (incorrectly flagged by OLD): {new_avoided_false_alarms}")
    print("=" * 80)

    # Interactive mode to test custom user inputs live
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE: Test your own custom messages live")
    print("=" * 80)
    print("Type any message to compare OLD vs NEW safety check (or type 'exit' or press Enter to quit).\n")

    while True:
        try:
            user_input = input("Enter test message > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
            
        if not user_input or user_input.lower() in ("exit", "quit"):
            print("Exiting interactive test mode.")
            break

        # Run OLD
        old_output = old_fn(user_input)
        old_is_emerg = "EMERGENCY DETECTED" in old_output
        old_status = "EMERGENCY" if old_is_emerg else "NOT EMERGENCY"

        # Run NEW
        llm_res = await check_emergency_llm(user_input)
        is_emerg_new, resp_msg_new, meta = check_emergency_fused(user_input, llm_res)
        new_status = "EMERGENCY" if is_emerg_new else "NOT EMERGENCY"

        llm_conf = meta.get("llm_confidence", 0.0)
        llm_reason = meta.get("llm_reason", "")
        kw_triggered = meta.get("keyword_triggered", [])
        kw_overridden = meta.get("keyword_overridden", False)

        if old_is_emerg == is_emerg_new:
            match_diff = "Same result"
        elif not old_is_emerg and is_emerg_new:
            match_diff = "DIFFERENT - NEW caught what OLD missed"
        else:
            match_diff = "DIFFERENT - NEW avoided a false alarm that OLD raised"

        print(f"\nMESSAGE: \"{user_input}\"")
        print(f"  OLD (keyword only):     {old_status}  (Keywords matched: {kw_triggered if kw_triggered else 'None'})")
        print(f"  NEW (keyword + LLM):    {new_status}   (LLM confidence: {llm_conf:.2f}, reason: \"{llm_reason}\")")
        if kw_overridden:
            print(f"  SAFETY OVERRIDE:        YES - Keyword '{kw_triggered}' matched, but LLM override prevented false alarm!")
        print(f"  MATCH OR DIFFERENCE:    {match_diff}\n")

if __name__ == "__main__":
    asyncio.run(main())
