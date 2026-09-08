import asyncio
import os
import sys
import textwrap
import time
from dotenv import load_dotenv

load_dotenv()

# Force UTF-8 on Windows stdout if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from agent.graph import run_agent, run_agent_v2

TEST_USER_ID = "00000000-0000-0000-0000-000000000000"
TOTAL_BASELINE_STREAMS = 11

ALL_STREAMS_LIST = [
    "profile", "current_hr", "sleep", "hr_history",
    "hrv", "spo2", "steps", "bp", "temperature",
    "stress", "cycles"
]

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

def print_header(title: str, width: int = 76):
    top = "=" * width
    padded = title.center(width - 4)
    middle = f"| {clr(padded, 'bold')} |"
    bottom = "=" * width
    print("\n" + clr(top, "cyan"))
    print(clr(middle, "cyan"))
    print(clr(bottom, "cyan") + "\n")

def print_box(step_title: str, lines: list[str], width: int = 76, border_color: str = "cyan"):
    title_text = f"- [ {step_title} ] "
    fill_len = max(0, width - len(title_text) - 1)
    top_line = "+" + title_text + "-" * fill_len + "+"
    bottom_line = "+" + "-" * (width - 2) + "+"

    print(clr(top_line, border_color))
    for line in lines:
        content_width = width - 4
        wrapped_sublines = textwrap.wrap(line, width=content_width) or [""]
        for subline in wrapped_sublines:
            space = " " * (content_width - len(subline))
            print(f"{clr('|', border_color)} {subline}{space} {clr('|', border_color)}")
    print(clr(bottom_line, border_color))

async def main():
    print_header("MEDXAI ARCHITECTURE COMPARISON — VERSION C vs VERSION D", width=76)
    print(clr("  This interactive tool compares:", "dim"))
    print(clr("  • VERSION C (Old Baseline): Unselective full-database dump (11 streams fetched every time)", "dim"))
    print(clr("  • VERSION D (Latest Engine): Selective stream routing, safety fusion gate & grounding score", "dim"))
    print(clr("  Type any health question (or 'exit' or 'q' to quit).\n", "dim"))

    while True:
        try:
            user_input = input(clr("[Enter Health Question] > ", "bold")).strip()
            if not user_input or user_input.lower() in ["exit", "quit", "q"]:
                print(clr("\nExiting Architecture Comparison.", "dim"))
                break

            q_str = f'"{user_input}"'
            print(f"\nQuery: {clr(q_str, 'bold')}\n")

            # ── 1. RUN VERSION C (OLD BASELINE) ─────────────────────────────
            print(clr("Running VERSION C (Unselective Full-Database Fetch)...", "dim"))
            t_c_start = time.monotonic()
            reply_c, card_c = await run_agent(user_input, TEST_USER_ID)
            t_c_elapsed = time.monotonic() - t_c_start

            version_c_lines = [
                f"DB Fetch Strategy:     {clr('UNSELECTIVE FULL-DATABASE FETCH', 'yellow')}",
                f"Streams Fetched (11):  {ALL_STREAMS_LIST}",
                f"Payload Overhead:      {clr('100% full database payload sent to LLM prompt', 'yellow')}",
                f"Execution Latency:     {t_c_elapsed:.2f}s",
                f"Card Attached:         {card_c['type'] if card_c else 'None'}",
                "Response:",
            ]
            for ln in reply_c.splitlines():
                version_c_lines.append(f"  {ln}")

            print_box("VERSION C: OLD BASELINE (UNSELECTIVE FULL-DB DUMP)", version_c_lines, width=76, border_color="yellow")
            print()

            # ── 2. RUN VERSION D (LATEST ARCHITECTURE) ──────────────────────
            print(clr("Running VERSION D (Selective Stream Routing + Safety Fusion)...", "dim"))
            reply_d, card_d, meta_d = await run_agent_v2(user_input, TEST_USER_ID, verbose=True, suppress_internal_log=True)

            is_emerg = meta_d.get("is_emergency", False)
            safety_meta = meta_d.get("safety_meta", {})
            streams = meta_d.get("streams", [])
            timing = meta_d.get("timing", {})
            facts = meta_d.get("facts", [])
            grounding_res = meta_d.get("grounding_res", {})
            score_val = grounding_res.get("grounding_score", 1.0)
            total_nums = grounding_res.get("total_numbers_checked", 0)
            ungrounded = grounding_res.get("ungrounded_numbers", [])
            grounded_nums = total_nums - len(ungrounded)

            stream_count = len(streams)
            data_reduction = round((1.0 - (stream_count / TOTAL_BASELINE_STREAMS)) * 100) if stream_count > 0 else 100

            version_d_lines = []
            if is_emerg:
                version_d_lines.append(f"Router Decision:       SHORT-CIRCUITED (Emergency Detected)")
                version_d_lines.append(f"Emergency Status:      {clr('[!] EMERGENCY TRIGGERED', 'red')}")
                version_d_lines.append(f"Execution Latency:     {timing.get('total', 0.0):.2f}s (Sub-second short-circuit)")
            else:
                version_d_lines.append(f"Router Decision:       LLM Stream Classification -> Stream Keys {streams}")
                version_d_lines.append(f"DB Fetch Strategy:     {clr('SELECTIVE DYNAMIC STREAMING', 'green')}")
                version_d_lines.append(f"Payload Reduction:     {clr(f'{data_reduction}% LESS DATA FETCHED vs baseline', 'green')} ({stream_count} of 11 streams)")
                
                kw_trig = safety_meta.get("keyword_triggered", [])
                kw_str = f"Keyword={kw_trig if kw_trig else 'None'}"
                llm_trig = safety_meta.get("llm_triggered", False)
                conf = safety_meta.get("llm_confidence", 0.0)
                version_d_lines.append(f"Safety Fusion Gate:    {kw_str} | LLM={llm_trig} (Conf: {conf:.2f}) -> {clr('Cleared', 'green')}")
                
                if safety_meta.get("keyword_overridden"):
                    term_str = ", ".join(kw_trig) if kw_trig else "matched term"
                    version_d_lines.append(f"Safety Override:       Keyword flagged '{term_str}' but LLM overruled (high-confidence non-emergency)")

                if score_val >= 0.99:
                    score_str = clr(f"{score_val:.2f} ({grounded_nums}/{total_nums} numbers verified verbatim)", "green")
                elif score_val >= 0.70:
                    score_str = clr(f"{score_val:.2f} ({grounded_nums}/{total_nums} numbers verified)", "yellow")
                else:
                    score_str = clr(f"{score_val:.2f} (WARNING: Ungrounded numbers: {ungrounded})", "red")
                version_d_lines.append(f"Grounding Score:       {score_str}")

                if facts:
                    version_d_lines.append("Facts Extracted:")
                    for f in facts:
                        version_d_lines.append(f" - {f}")
                else:
                    version_d_lines.append("Facts Extracted:       None")

                if meta_d.get("rationale"):
                    version_d_lines.append(f"Clinical Rationale:    {meta_d.get('rationale')}")

                if meta_d.get("action"):
                    version_d_lines.append(f"Recommended Action:    {meta_d.get('action')}")

                version_d_lines.append(f"Execution Latency:     {timing.get('total', 0.0):.2f}s")
                version_d_lines.append(f"Card Attached:         {card_d['type'] if card_d else 'None'}")

            version_d_lines.append("Response:")
            for ln in reply_d.splitlines():
                version_d_lines.append(f"  {ln}")

            print_box("VERSION D: LATEST ARCHITECTURE (SELECTIVE STREAMING + GROUNDING)", version_d_lines, width=76, border_color="green")
            print("\n" + "=" * 76 + "\n")

        except KeyboardInterrupt:
            print(clr("\nExiting.", "dim"))
            break
        except Exception as e:
            print(clr(f"Error executing comparison query: {e}", "red"))

if __name__ == "__main__":
    asyncio.run(main())
