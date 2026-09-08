import asyncio
import os
import sys
import textwrap
from dotenv import load_dotenv

load_dotenv()

# Force UTF-8 on Windows stdout if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from agent.graph import run_agent_v2

TEST_USER_ID = "00000000-0000-0000-0000-000000000000"
TOTAL_BASELINE_STREAMS = 11

# Terminal ANSI Color Helper (degrades gracefully if colorama or ANSI is unsupported)
USE_COLOR = True
try:
    import colorama
    colorama.init()
except Exception:
    if os.name == 'nt' and not os.getenv("TERM"):
        USE_COLOR = False

def clr(text: str, code: str) -> str:
    if not USE_COLOR:
        return text
    codes = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "cyan": "\033[96m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "dim": "\033[2m"
    }
    return f"{codes.get(code, '')}{text}{codes['reset']}"

def print_header(title: str, width: int = 74):
    top = "=" * width
    padded = title.center(width - 4)
    middle = f"| {clr(padded, 'bold')} |"
    bottom = "=" * width
    print("\n" + clr(top, "cyan"))
    print(clr(middle, "cyan"))
    print(clr(bottom, "cyan") + "\n")

def print_box(step_title: str, lines: list[str], width: int = 74, border_color: str = "cyan"):
    # Header line with title embedded
    title_text = f"- [ {step_title} ] "
    fill_len = max(0, width - len(title_text) - 1)
    top_line = "+" + title_text + "-" * fill_len + "+"
    bottom_line = "+" + "-" * (width - 2) + "+"

    print(clr(top_line, border_color))
    for line in lines:
        # Wrap content to fit inside border safely
        content_width = width - 4
        wrapped_sublines = textwrap.wrap(line, width=content_width) or [""]
        for subline in wrapped_sublines:
            space = " " * (content_width - len(subline))
            print(f"{clr('|', border_color)} {subline}{space} {clr('|', border_color)}")
    print(clr(bottom_line, border_color))

def format_demo_output(query: str, reply: str, card: dict, meta: dict, width: int = 74):
    if not meta:
        print_box("RESPONSE", [reply], width=width)
        return

    is_emerg = meta.get("is_emergency", False)
    safety_meta = meta.get("safety_meta", {})
    streams = meta.get("streams", [])
    timing = meta.get("timing", {})
    facts = meta.get("facts", [])
    grounding_res = meta.get("grounding_res", {})
    score_val = grounding_res.get("grounding_score", 1.0)
    total_nums = grounding_res.get("total_numbers_checked", 0)
    ungrounded = grounding_res.get("ungrounded_numbers", [])
    grounded_nums = total_nums - len(ungrounded)

    # 1. STEP 1: ROUTING & SAFETY
    t_stage1 = timing.get("stage1_safety_router", 0.0)
    t_total = timing.get("total", 0.0)
    
    stream_count = len(streams)
    data_reduction = round((1.0 - (stream_count / TOTAL_BASELINE_STREAMS)) * 100) if stream_count > 0 else 100

    step1_lines = []
    step1_lines.append(f"Execution Latency:     {t_stage1:.2f}s (Total pipeline: {t_total:.2f}s)")
    
    if is_emerg:
        step1_lines.append(f"Router decision:       SHORT-CIRCUITED (Emergency Detected)")
        step1_lines.append(f"Emergency Status:      {clr('[!] EMERGENCY TRIGGERED', 'red')}")
        kw_trig = safety_meta.get("keyword_triggered", [])
        llm_reason = safety_meta.get("llm_reason", "")
        step1_lines.append(f"Triggers:              Keywords={kw_trig} | Reason={llm_reason}")
        print_box(f"STEP 1: ROUTING & SAFETY (INSTANT EMERGENCY SHORT-CIRCUIT)", step1_lines, width=width, border_color="red")
    else:
        step1_lines.append(f"Router decision:       LLM Classification -> Stream Keys {streams}")
        step1_lines.append(f"Biometric Scope:       {stream_count} of {TOTAL_BASELINE_STREAMS} streams ({clr(f'{data_reduction}% data reduction', 'green')} vs baseline)")
        
        kw_trig = safety_meta.get("keyword_triggered", [])
        kw_str = f"Keyword={kw_trig if kw_trig else 'None'}"
        llm_trig = safety_meta.get("llm_triggered", False)
        conf = safety_meta.get("llm_confidence", 0.0)
        status_str = clr("Cleared", "green") if not is_emerg else clr("Triggered", "red")
        
        step1_lines.append(f"Emergency Check:       {kw_str} | LLM={llm_trig} (Conf: {conf:.2f}) -> {status_str}")

        kw_overridden = safety_meta.get("keyword_overridden", False)
        if kw_overridden:
            term_str = ", ".join(kw_trig) if kw_trig else "matched term"
            step1_lines.append(f"Override:              Keyword flagged '{term_str}' but LLM overruled (high-confidence non-emergency)")

        print_box(f"STEP 1: ROUTING & SAFETY GATE ({t_stage1:.2f}s)", step1_lines, width=width, border_color="cyan")

    print()

    # 2. STEP 2: GROUNDED REASONING
    if not is_emerg:
        t_fetch = timing.get("stage2_fetch", 0.0)
        t_llm = timing.get("stage3_llm", 0.0)
        step2_lines = []
        step2_lines.append(f"Selective DB Fetch:    {t_fetch:.2f}s | Grounded LLM Gen: {t_llm:.2f}s")

        if score_val >= 0.99:
            score_str = clr(f"{score_val:.2f} ({grounded_nums}/{total_nums} numbers verified verbatim)", "green")
        elif score_val >= 0.70:
            score_str = clr(f"{score_val:.2f} ({grounded_nums}/{total_nums} numbers verified)", "yellow")
        else:
            score_str = clr(f"{score_val:.2f} (WARNING: Ungrounded numbers: {ungrounded})", "red")
            
        step2_lines.append(f"Grounding Score:       {score_str}")

        if facts:
            step2_lines.append("Facts:")
            for f in facts:
                step2_lines.append(f" - {f}")
        else:
            step2_lines.append("Facts:                 None extracted")

        if meta.get("rationale"):
            step2_lines.append(f"Clinical Rationale:    {meta.get('rationale')}")

        if meta.get("action"):
            step2_lines.append(f"Recommended Action:    {meta.get('action')}")
            
        print_box(f"STEP 2: GROUNDED REASONING & VERIFICATION", step2_lines, width=width, border_color="blue")
        print()

    # 3. FINAL RESPONSE
    reply_lines = reply.splitlines()
    print_box("FINAL RESPONSE TO PATIENT", reply_lines, width=width, border_color="green")

    # 4. CARD ATTACHMENT PREVIEW
    if card:
        print()
        card_type = card.get("type", "unknown")
        card_data = card.get("data", {})
        card_lines = [
            f"Attached Widget:  {clr(card_type, 'bold')}",
            f"Data Summary:     {card_data}"
        ]
        print_box("FLUTTER UI CARD ATTACHMENT", card_lines, width=width, border_color="magenta")

async def main():
    print_header("MEDXAI - LIVE QUERY ANALYSIS DEMO")
    print(clr("  This interactive demo showcases the Version D Architecture:", "dim"))
    print(clr("  * Parallel Query Router & Safety Fusion Gate", "dim"))
    print(clr("  * Selective Biometric Stream Fetching (Data Reduction)", "dim"))
    print(clr("  * Fact -> Rationale -> Action Grounded Clinical Reasoning", "dim"))
    print(clr("  * Real-Time Numerical Grounding Verification Engine", "dim"))
    print(clr("  Type 'exit' or 'q' to quit.\n", "dim"))

    while True:
        try:
            user_input = input(clr("[Enter Health Question] > ", "bold")).strip()
            if not user_input or user_input.lower() in ["exit", "quit", "q"]:
                print(clr("\nExiting MedXAI Demo.", "dim"))
                break

            print(clr("\nProcessing query through Version D pipeline...", "dim"))
            reply, card, meta = await run_agent_v2(user_input, TEST_USER_ID, verbose=True, suppress_internal_log=True)
            
            q_str = f'"{user_input}"'
            print(f"\nQuery: {clr(q_str, 'bold')}\n")
            format_demo_output(user_input, reply, card, meta)
            print("\n" + "=" * 74 + "\n")

        except KeyboardInterrupt:
            print(clr("\nExiting.", "dim"))
            break
        except Exception as e:
            print(clr(f"Error executing demo query: {e}", "red"))

if __name__ == "__main__":
    asyncio.run(main())
