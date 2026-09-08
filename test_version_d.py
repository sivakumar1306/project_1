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

from agent.graph import run_agent

TEST_USER_ID = "00000000-0000-0000-0000-000000000000"

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
    print("\n" + clr(top, "yellow"))
    print(clr(middle, "yellow"))
    print(clr(bottom, "yellow") + "\n")

def print_box(step_title: str, lines: list[str], width: int = 76, border_color: str = "yellow"):
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
    print_header("MEDXAI BASELINE DEMO — VERSION C (FULL-DATABASE FETCH)", width=76)
    print(clr("  This terminal runs VERSION C (Old Baseline Architecture):", "dim"))
    print(clr("  • Unselectively fetches ALL 11 biometric streams from Supabase on every prompt", "dim"))
    print(clr("  • Dumps entire patient database history into the LLM prompt context window", "dim"))
    print(clr("  Type any health question (or 'exit' or 'q' to quit).\n", "dim"))

    while True:
        try:
            user_input = input(clr("[Enter Health Question] > ", "bold")).strip()
            if not user_input or user_input.lower() in ["exit", "quit", "q"]:
                print(clr("\nExiting Version C Baseline Demo.", "dim"))
                break

            q_str = f'"{user_input}"'
            print(f"\nQuery: {clr(q_str, 'bold')}\n")

            print(clr("Running VERSION C (Fetching all 11 DB streams into prompt)...", "yellow"))
            t_start = time.monotonic()
            reply_c, card_c = await run_agent(user_input, TEST_USER_ID)
            t_elapsed = time.monotonic() - t_start

            version_c_lines = [
                f"DB Fetch Strategy:     {clr('UNSELECTIVE FULL-DATABASE DUMP', 'yellow')}",
                f"Streams Fetched (11):  {ALL_STREAMS_LIST}",
                f"Payload Overhead:      {clr('100% full database payload sent to LLM prompt context', 'yellow')}",
                f"Execution Latency:     {t_elapsed:.2f}s",
                f"Card Attached:         {card_c['type'] if card_c else 'None'}",
                "Response:",
            ]
            for ln in reply_c.splitlines():
                version_c_lines.append(f"  {ln}")

            print_box("VERSION C: OLD BASELINE RESPONSE", version_c_lines, width=76, border_color="yellow")
            print("\n" + "=" * 76 + "\n")

        except KeyboardInterrupt:
            print(clr("\nExiting.", "dim"))
            break
        except Exception as e:
            print(clr(f"Error executing Version C query: {e}", "red"))

if __name__ == "__main__":
    asyncio.run(main())
