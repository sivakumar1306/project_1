import asyncio
import json
import os
import re
from typing import List
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage

ALL_STREAMS = [
    "profile", "current_hr", "sleep", "hr_history",
    "hrv", "spo2", "steps", "bp", "temperature",
    "stress", "cycles"
]

ROUTER_SYSTEM_PROMPT = """You are a precise data-stream routing classifier for a smart health ring backend.
Given a user message, classify which of the following health biometric streams are needed to answer it:

Available stream keys:
- profile: User's profile (name, age, gender, height, weight)
- current_hr: Live/current heart rate reading
- sleep: Sleep duration, sleep score, sleep stages
- hr_history: Historical daily heart rate trends
- hrv: Heart rate variability
- spo2: Blood oxygen saturation
- steps: Step count, active calories
- bp: Blood pressure (systolic/diastolic)
- temperature: Body temperature readings
- stress: Stress levels
- cycles: Menstrual cycle logs, ovulation, period info

Rules:
1. If the query asks for a general overview, summary, or "how am I doing" / "how is my health" / "overall status", return ALL stream keys:
["profile", "current_hr", "sleep", "hr_history", "hrv", "spo2", "steps", "bp", "temperature", "stress", "cycles"]

2. If the query is purely a general medical or clinical knowledge question without any personal biometric request (e.g., "what causes diabetes", "how to treat a headache"), return an EMPTY list:
[]

3. If the query asks about specific biometric metric(s), return ONLY the matching stream key(s). For example:
- "How is my blood pressure?" -> ["bp"]
- "What is my heart rate right now?" -> ["current_hr"]
- "How did I sleep last night?" -> ["sleep"]

Output ONLY valid JSON containing a array of string stream keys. Do NOT include markdown code fences, commentary, or extra keys.
Example output:
["bp"]
"""

def fallback_keyword_router(message: str) -> List[str]:
    msg_lower = message.lower()
    
    # Check if general overview
    overview_keywords = ["how am i", "how's my health", "overall", "summary", "everything", "overview", "health status", "how am i doing"]
    if any(k in msg_lower for k in overview_keywords):
        return ALL_STREAMS[:]
    
    selected = []
    if "profile" in msg_lower or "my name" in msg_lower or "my age" in msg_lower:
        selected.append("profile")
    if any(k in msg_lower for k in ["sleep", "asleep", "wake"]):
        selected.append("sleep")
    if any(k in msg_lower for k in ["blood pressure", "systolic", "diastolic"]) or re.search(r'\bbp\b', msg_lower):
        selected.append("bp")
    if any(k in msg_lower for k in ["spo2", "sp02", "blood oxygen", "oxygen level"]):
        selected.append("spo2")
    if any(k in msg_lower for k in ["hrv", "variability"]):
        selected.append("hrv")
    if any(k in msg_lower for k in ["current heart rate", "live heart rate", "heart rate right now", "pulse right now"]):
        selected.append("current_hr")
    elif any(k in msg_lower for k in ["heart rate", "pulse", "bpm"]):
        selected.append("current_hr")
        selected.append("hr_history")
    if any(k in msg_lower for k in ["steps", "walked", "walking", "calories"]):
        selected.append("steps")
    if any(k in msg_lower for k in ["temperature", "temp", "fever"]):
        selected.append("temperature")
    if any(k in msg_lower for k in ["stress", "anxiety"]):
        selected.append("stress")
    if any(k in msg_lower for k in ["period", "cycle", "menstrual", "ovulation", "pms"]):
        selected.append("cycles")
        
    return selected

async def classify_query_streams(message: str) -> List[str]:
    """
    Query Router mapping a user query to relevant data stream names.
    Attempts LLM classification FIRST, with exponential backoff for 429 rate limits.
    Falls back to keyword matching ONLY if the LLM call fails or returns malformed output.
    """
    try:
        from agent.graph import get_medxai_llm
        llm = get_medxai_llm()

        for attempt in range(3):
            try:
                res = await llm.ainvoke([
                    SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                    HumanMessage(content=message)
                ])
                raw = res.content.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
                    raw = re.sub(r"```$", "", raw).strip()
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    valid_streams = [s for s in parsed if s in ALL_STREAMS]
                    print(f"[ROUTER] Source: llm | Streams: {valid_streams}")
                    return valid_streams
                break
            except Exception as err:
                if "429" in str(err) and attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                else:
                    raise err

    except Exception as e:
        print(f"[ROUTER] LLM router error ({e}), falling back to keyword router")

    fallback_streams = fallback_keyword_router(message)
    print(f"[ROUTER] Source: keyword_fallback | Streams: {fallback_streams}")
    return fallback_streams
