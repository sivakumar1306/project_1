import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from agent.graph import run_agent, run_agent_v2

async def main():
    test_user_id = "00000000-0000-0000-0000-000000000000"
    
    test_queries = [
        ("Specific metric query", "How is my blood pressure today?"),
        ("General overview query", "How is my health doing overall?"),
        ("Clinical knowledge query", "What are the common symptoms of type 2 diabetes?"),
        ("Explicit Emergency query", "I have severe crushing chest pain and difficulty breathing"),
        ("Implicit/Paraphrased Emergency query", "My left arm feels numb and my jaw aches with strange tightness in my chest"),
    ]

    print("==========================================================================")
    print("      TESTING VERSION C (run_agent) VS VERSION D (run_agent_v2)")
    print("==========================================================================")

    for category, query in test_queries:
        print(f"\n\n--------------------------------------------------------------------------")
        print(f"CATEGORY: {category}")
        print(f"QUERY:    '{query}'")
        print(f"--------------------------------------------------------------------------")
        
        print("\n--- [VERSION C: Original run_agent] ---")
        reply_c, card_c = await run_agent(query, test_user_id)
        print(f"Reply C:\n{reply_c}")
        print(f"Card C: {card_c['type'] if card_c else None}")

        await asyncio.sleep(2.5)

        print("\n--- [VERSION D: Grounded run_agent_v2] ---")
        reply_d, card_d = await run_agent_v2(query, test_user_id)
        print(f"Reply D:\n{reply_d}")
        print(f"Card D: {card_d['type'] if card_d else None}")

        await asyncio.sleep(2.5)

if __name__ == "__main__":
    asyncio.run(main())
