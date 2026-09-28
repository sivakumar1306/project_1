import asyncio
import json
import os
import re
from typing import List, Optional
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage

ALL_STREAMS = [
    "profile", "current_hr", "sleep", "hr_history",
    "hrv", "spo2", "steps", "bp", "temperature",
    "stress", "cycles", "fall_risk"
]

# Safety-asymmetric routing: when a query about the user's own state is
# safety-relevant but the router is unsure, widen the fetch toward these
# streams instead of risking a miss (trade a little efficiency for recall).
SAFETY_STREAMS = ["fall_risk", "current_hr", "spo2", "hrv"]
CONFIDENCE_THRESHOLD = 0.75

SAFETY_KEYWORDS = [
    "dizzy", "dizziness", "lightheaded", "light-headed", "faint", "unsteady", "balance",
    "fall", "falling", "fell", "wobbly", "vertigo", "black out", "blacked out", "weak legs",
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
- fall_risk: The user's own fall-risk score, balance/stability, risk of falling, dizziness or unsteadiness affecting them

Rules:
1. If the query asks for a general overview, summary, or "how am I doing" / "how is my health" / "overall status", return ALL stream keys:
["profile", "current_hr", "sleep", "hr_history", "hrv", "spo2", "steps", "bp", "temperature", "stress", "cycles"]

2. If the query is purely a general medical or clinical knowledge question without any personal biometric request (e.g., "what causes diabetes", "how to treat a headache"), return an EMPTY list:
[]

3. If the query asks about specific biometric metric(s), return ONLY the matching stream key(s). For example:
- "How is my blood pressure?" -> ["bp"]
- "What is my heart rate right now?" -> ["current_hr"]
- "How did I sleep last night?" -> ["sleep"]

4. Questions about the user's OWN fall risk, balance, unsteadiness or dizziness -> include "fall_risk".
   General questions about falls in other people (e.g. "why do older people fall?") -> [] (no personal data).

Also report:
- "confidence": your confidence (0.0-1.0) that the selected streams are sufficient to answer safely.
- "safety_relevant": true ONLY if the message is about the user's own current state AND could relate to a fall, injury, fainting, dizziness, breathing or cardiac risk. General knowledge questions are false.

Output ONLY a valid JSON object. Do NOT include markdown code fences or commentary.
Example output:
{"streams": ["bp"], "confidence": 0.95, "safety_relevant": false}
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
    if any(k in msg_lower for k in ["fall risk", "risk of falling", "falling", "balance", "unsteady", "dizzy", "lightheaded"]):
        selected.append("fall_risk")

    return selected


def _is_personal(msg_lower: str) -> bool:
    return bool(re.search(r"\b(i|i'm|im|my|me|am i)\b", msg_lower))


def keyword_safety_relevant(message: str) -> bool:
    m = message.lower()
    return _is_personal(m) and any(k in m for k in SAFETY_KEYWORDS)


def apply_safety_widening(streams: List[str], confidence: float, safety_relevant: bool) -> tuple[List[str], bool]:
    """
    Safety-asymmetric routing rule.
    - safety-relevant + low confidence  -> union with SAFETY_STREAMS
    - safety-relevant + high confidence -> make sure fall_risk is included
    Returns (final_streams, widened_flag).
    """
    if not safety_relevant:
        return streams, False
    final = list(streams)
    if confidence < CONFIDENCE_THRESHOLD:
        for s in SAFETY_STREAMS:
            if s not in final:
                final.append(s)
    elif "fall_risk" not in final:
        final.append("fall_risk")
    return final, final != list(streams)


def _parse_router_output(raw: str) -> tuple[List[str], float, Optional[bool]]:
    parsed = json.loads(raw)
    if isinstance(parsed, list):                       # legacy format
        return [x for x in parsed if x in ALL_STREAMS], 0.9, None
    if isinstance(parsed, dict):
        streams = [x for x in (parsed.get("streams") or []) if x in ALL_STREAMS]
        try:
            conf = float(parsed.get("confidence", 0.9))
        except Exception:
            conf = 0.9
        sr = parsed.get("safety_relevant")
        return streams, max(0.0, min(1.0, conf)), (bool(sr) if sr is not None else None)
    raise ValueError("router output is neither a list nor an object")

async def classify_query_streams_v2(message: str) -> tuple[List[str], dict]:
    """
    Query router with confidence scoring and safety-asymmetric widening.
    Returns (streams, meta) where meta has source, confidence, safety_relevant,
    raw_streams (before widening) and widened.
    """
    raw_streams: List[str] = []
    confidence = 0.5
    safety_relevant: Optional[bool] = None
    source = "keyword_fallback"
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
                raw_streams, confidence, safety_relevant = _parse_router_output(raw)
                source = "llm"
                break
            except Exception as err:
                if "429" in str(err) and attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                else:
                    raise err
    except Exception as e:
        print(f"[ROUTER] LLM router error ({e}), falling back to keyword router")

    if source != "llm":
        raw_streams = fallback_keyword_router(message)
        confidence = 0.5

    if safety_relevant is None:
        safety_relevant = keyword_safety_relevant(message)

    streams, widened = apply_safety_widening(raw_streams, confidence, safety_relevant)
    meta = {
        "source": source,
        "confidence": round(confidence, 2),
        "safety_relevant": safety_relevant,
        "raw_streams": raw_streams,
        "widened": widened,
    }
    print(f"[ROUTER] Source: {source} | Streams: {streams} | conf={confidence:.2f} | safety_relevant={safety_relevant} | widened={widened}")
    return streams, meta


async def classify_query_streams(message: str) -> List[str]:
    """Backward-compatible wrapper (returns streams only)."""
    streams, _ = await classify_query_streams_v2(message)
    return streams
