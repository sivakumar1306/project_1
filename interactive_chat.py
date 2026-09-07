import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from agent.graph import run_agent, run_agent_v2

TEST_USER_ID = "00000000-0000-0000-0000-000000000000"

async def main():
    print("==========================================================================")
    print("      MEDXAI TERMINAL TESTER (Version C vs Version D)")
    print(f"      Demo User ID: {TEST_USER_ID}")
    print("==========================================================================")
    print("Type your health question (or type 'exit' or 'q' to quit).\n")

    while True:
        try:
            user_input = input("\n[Enter Question] > ").strip()
            if not user_input or user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive terminal test.")
                break

            print("\n--------------------------------------------------------------------------")
            print("Running VERSION D (run_agent_v2) - Grounded Router & Safety Fusion...")
            print("--------------------------------------------------------------------------")
            reply_d, card_d = await run_agent_v2(user_input, TEST_USER_ID)
            print("\n[VERSION D RESPONSE]:")
            print(reply_d)
            if card_d:
                print(f"\n[CARD DATA ATTACHED]: {card_d['type']} -> {card_d['data']}")

        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
