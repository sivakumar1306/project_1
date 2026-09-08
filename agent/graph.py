import asyncio
import time
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from agent.tools import (
    search_medical_knowledge,
    get_patient_data,
    check_emergency,
    analyze_symptoms
)
import os
import re
from datetime import datetime as dt, timedelta
from dotenv import load_dotenv
from typing import Any, Optional
from db.supabase import supabase

load_dotenv()

SYSTEM_PROMPT = """You are MedXAI, an intelligent AI health assistant connected to a patient's smart ring data. Emergency screening and patient biometrics have already been pre-processed and provided in the prompt context below. Do not attempt to invoke external tools or functions.

Important rules:
- Users may make spelling mistakes or typos — always interpret their intent charitably and respond helpfully. For example "dibeties" means "diabetes", "symtoms" means "symptoms", "herat" means "heart". Never reject a message due to spelling.
- If no ring biometric data is found for the user, respond clearly: "Please connect your ring to view analysis." Never substitute a plausible-sounding number.
- Never diagnose — only provide health insights and guidance
- Always recommend seeing a doctor for serious concerns
- CRITICAL — DATA ACCURACY: Refer to the provided PATIENT DATA before answering ANY question about the user's own biometrics. Only state numeric values that appear VERBATIM in that provided data. Never estimate, round, infer, average, or invent a number that isn't explicitly present in the data. If you cannot find a requested value anywhere in the provided data, say so explicitly instead of producing a number.
- When asked for the "current" or "live" heart rate specifically, use ONLY the value labeled "CURRENT HEART RATE" in the provided data. Do NOT substitute a value from "HISTORICAL DAILY HEART RATE" (those are daily avg/min/max, not current). If that reading is marked [STALE], say clearly that it's not real-time and state its actual age/date — do not present it as "current" without that caveat. If the data says no reading was found, say so plainly instead of guessing.
- Do not fabricate field labels or stats (e.g. "resting average", "recent max") that are not literally present in the provided data.
- Before sending your final reply, silently check every number you are about to state against the provided data. If a number cannot be found verbatim in the data, delete it and say the data is unavailable instead.
- NEVER pair a denial (any phrasing like "I cannot find", "I don't have", "no reading is available", "I cannot retrieve") with a specific real value in the same reply. If you have a real value to report, report it — do not deny having it. If you truly have no value, do not state a number at all. Check this before every reply: if your reply contains both a specific number and a denial phrase about that same metric, delete the denial and keep only the value with its staleness caveat.
- STRICT SCOPING: when the user asks about ONE specific metric by name (e.g. "how is my blood pressure"), your entire reply must be about that metric ONLY. Do not mention any other metric's data (heart rate, temperature, HRV, SpO2, sleep, steps) even if it's present in what was fetched — ignore that other data entirely for this reply. The only exception is a genuine emergency flagged by check_emergency.
- If the user's question does not name a specific metric (e.g. "how am I doing", "how's my health"), you may give a brief multi-metric overview — but if they name one metric, stay scoped to that one.

RESPONSE FORMAT — STRICTLY FOLLOW THIS:
- Use precise clinical/medical terminology (e.g. "tachycardia" instead of "fast heart rate", "hyperglycemia" instead of "high blood sugar"). Add a brief plain-language clarification in parentheses the first time you use an uncommon term.
- Do NOT use any markdown formatting — no asterisks, no bold, no headers, no numbering. Plain text only.
- Start with one short summary line (no label, no prefix — just the sentence).
- Give AT MOST 4 bullet points total (not counting the mandatory disclaimer bullet). If more metrics are relevant than that, group/merge related ones into a single bullet (e.g. combine HR+HRV+SpO2 into one "vitals are in normal range" bullet) rather than listing each one separately.
- Do not list every historical day's data — summarize the trend across the days (e.g. "sleep score improved from 63 to 89 over the week") in one bullet instead of one bullet per day.
- Every number stated must still come verbatim from tool output — summarizing must never introduce averages or values not present in the tool output.
- Follow with bullet points using a plain hyphen "-" at the start of each line. Keep each bullet under 15 words.
- Do not use section labels like "Summary:" or "Findings:" — just a summary sentence, then bullets.
- Be empathetic in tone even while being concise.
- Always end with this exact line as the final bullet: "This is general health information, not medical advice."

Example format:
No signs of fever based on current data.
- Current vitals normal: HR 90 bpm, SpO2 97%
- Temperature: 36.6 °C (afebrile)
- Monitor for chills, body aches, or fatigue
- Consult a doctor if fever develops or persists
- This is general health information, not medical advice.

Example when asked specifically for current/live heart rate and the reading is marked stale:
No real-time heart rate reading is available right now.
- Last recorded reading was 84 bpm on 21 July, 2026
- That is 4 days old, not a live measurement
- Open the ring app to sync or take a fresh reading
- Consult a doctor if you feel unwell
- This is general health information, not medical advice.
"""

# Cached at module level instead of recreated on every /chat request — building
# a fresh ChatMistralAI client + react-agent graph per call was wasted work on
# every single request for no benefit, since none of it depends on per-request
# state (message/user_id are only passed in at invoke time, not construction time).
_AGENT = None
_LLM = None

def get_medxai_llm():
    global _LLM
    if _LLM is None:
        groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
        mistral_key = (os.getenv("MISTRAL_API_KEY") or "").strip()
        openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()

        if groq_key:
            from langchain_groq import ChatGroq
            _LLM = ChatGroq(
                groq_api_key=groq_key,
                model_name="openai/gpt-oss-120b",
                temperature=0.1,
            )
            print("[LLM PROVIDER] Initialized Groq (openai/gpt-oss-120b)")
        elif mistral_key:
            _LLM = ChatMistralAI(
                api_key=mistral_key,
                model="mistral-small-latest",
                temperature=0.1,
            )
            print("[LLM PROVIDER] Initialized Mistral (mistral-small-latest)")
        elif openai_key:
            from langchain_openai import ChatOpenAI
            _LLM = ChatOpenAI(
                api_key=openai_key,
                model="gpt-4o-mini",
                temperature=0.1,
            )
            print("[LLM PROVIDER] Initialized OpenAI (gpt-4o-mini)")
        else:
            raise ValueError("No LLM API key found! Please add GROQ_API_KEY or MISTRAL_API_KEY to your .env file.")
    return _LLM

def get_medxai_agent():
    global _AGENT
    if _AGENT is None:
        t0 = time.monotonic()
        llm = get_medxai_llm()
        tools = [
            check_emergency,
            get_patient_data,
            search_medical_knowledge,
            analyze_symptoms
        ]
        _AGENT = create_react_agent(llm, tools)
        print(f"[TIMING] Agent construction (first call only, cached after): {time.monotonic() - t0:.2f}s")
    return _AGENT


# ── Card builders for each vital ────────────────────────────────────────────
#
# Every supabase.table(...).execute() call below is synchronous/blocking —
# the Supabase Python client has no native async mode. Run inside FastAPI's
# single-threaded event loop directly, a blocking DB call here freezes the
# *entire server* for every other concurrent request (insights, chat, history,
# everyone) until it returns — not just this one. asyncio.to_thread() runs the
# blocking call on a background thread instead, so the event loop stays free
# to serve other requests while this one waits on the DB.

_WEEKDAY_ABBR = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

def _fmt_date(d: dt) -> str:
    return d.strftime('%Y-%m-%d')

async def get_sleep_card_data(user_id: str) -> Optional[dict[str, Any]]:
    try:
        if not user_id or user_id == "anonymous":
            return {
                "type": "sleep_highlights",
                "data": {
                    "time_awake_min": 10,
                    "light_sleep_min": 63,
                    "deep_sleep_min": 250,
                    "total_label": "5 hours and 13 minutes"
                }
            }

        def _query():
            return supabase.table("user_sleep")\
                .select("*")\
                .eq("user_id", user_id)\
                .order("date", desc=True)\
                .limit(1)\
                .execute()

        r = await asyncio.to_thread(_query)

        if r.data:
            row = r.data[0]
            total_val = row.get("total_duration") or 0
            if total_val > 1440:
                total_min = total_val // 60
            else:
                total_min = total_val

            hours = total_min // 60
            minutes = total_min % 60
            total_label = f"{hours} hour{'s' if hours != 1 else ''} and {minutes} minute{'s' if minutes != 1 else ''}"

            time_awake_min = int(total_min * 0.05)
            deep_sleep_min = int(total_min * 0.25)
            light_sleep_min = total_min - time_awake_min - deep_sleep_min

            return {
                "type": "sleep_highlights",
                "data": {
                    "time_awake_min": time_awake_min,
                    "light_sleep_min": light_sleep_min,
                    "deep_sleep_min": deep_sleep_min,
                    "total_label": total_label
                }
            }
    except Exception as e:
        print(f"Error fetching sleep card data: {e}")

    return {
        "type": "sleep_highlights",
        "data": {
            "time_awake_min": 10,
            "light_sleep_min": 63,
            "deep_sleep_min": 250,
            "total_label": "5 hours and 13 minutes"
        }
    }

async def get_hr_card_data(user_id: str) -> Optional[dict[str, Any]]:
    demo = {
        "type": "heart_rate_trend",
        "data": {
            "avg": 78, "min": 58, "max": 112, "unit": "bpm",
            "values": [72, 75, 80, 77, 82, 79, 78],
            "labels": _WEEKDAY_ABBR,
        }
    }
    if not user_id or user_id == "anonymous":
        return demo
    try:
        now = dt.utcnow()

        def _query():
            return supabase.table("user_hr").select("*").eq("user_id", user_id)\
                .gte("date", _fmt_date(now - timedelta(days=6)))\
                .lte("date", _fmt_date(now)).order("date", desc=False).execute()

        result = await asyncio.to_thread(_query)
        rows = result.data or []
        if not rows:
            return demo
        by_date = {r["date"]: r for r in rows}
        values, labels = [], []
        for i in range(7):
            day = now - timedelta(days=6 - i)
            row = by_date.get(_fmt_date(day))
            values.append(int(row["avg_hr"]) if row and row.get("avg_hr") else 0)
            labels.append(_WEEKDAY_ABBR[day.weekday()])
        non_zero = [v for v in values if v > 0]
        mins = [int(r["min_hr"]) for r in rows if r.get("min_hr")]
        maxs = [int(r["max_hr"]) for r in rows if r.get("max_hr")]
        return {
            "type": "heart_rate_trend",
            "data": {
                "avg": round(sum(non_zero) / len(non_zero)) if non_zero else 0,
                "min": min(mins) if mins else 0,
                "max": max(maxs) if maxs else 0,
                "unit": "bpm",
                "values": values,
                "labels": labels,
            }
        }
    except Exception as e:
        print(f"Error fetching HR card data: {e}")
        return demo

async def get_spo2_card_data(user_id: str) -> Optional[dict[str, Any]]:
    demo = {
        "type": "spo2_trend",
        "data": {
            "avg": 97, "min": 94, "max": 99, "unit": "%",
            "values": [96, 97, 98, 97, 95, 98, 97],
            "labels": _WEEKDAY_ABBR,
        }
    }
    if not user_id or user_id == "anonymous":
        return demo
    try:
        now = dt.utcnow()

        def _query():
            return supabase.table("user_spo2").select("*").eq("user_id", user_id)\
                .gte("date", _fmt_date(now - timedelta(days=6)))\
                .lte("date", _fmt_date(now)).order("date", desc=False).execute()

        result = await asyncio.to_thread(_query)
        rows = result.data or []
        if not rows:
            return demo
        by_date = {r["date"]: r for r in rows}
        values, labels = [], []
        for i in range(7):
            day = now - timedelta(days=6 - i)
            row = by_date.get(_fmt_date(day))
            values.append(int(row["avg_spo2"]) if row and row.get("avg_spo2") else 0)
            labels.append(_WEEKDAY_ABBR[day.weekday()])
        non_zero = [v for v in values if v > 0]
        mins = [int(r["min_spo2"]) for r in rows if r.get("min_spo2")]
        maxs = [int(r["max_spo2"]) for r in rows if r.get("max_spo2")]
        return {
            "type": "spo2_trend",
            "data": {
                "avg": round(sum(non_zero) / len(non_zero)) if non_zero else 0,
                "min": min(mins) if mins else 0,
                "max": max(maxs) if maxs else 0,
                "unit": "%",
                "values": values,
                "labels": labels,
            }
        }
    except Exception as e:
        print(f"Error fetching SpO2 card data: {e}")
        return demo

async def get_hrv_card_data(user_id: str) -> Optional[dict[str, Any]]:
    demo = {
        "type": "hrv_trend",
        "data": {
            "avg": 52, "min": 30, "max": 78, "unit": "ms",
            "values": [45, 50, 55, 48, 60, 52, 52],
            "labels": _WEEKDAY_ABBR,
        }
    }
    if not user_id or user_id == "anonymous":
        return demo
    try:
        now = dt.utcnow()

        def _query():
            return supabase.table("user_hrv").select("*").eq("user_id", user_id)\
                .gte("date", _fmt_date(now - timedelta(days=6)))\
                .lte("date", _fmt_date(now)).order("date", desc=False).execute()

        result = await asyncio.to_thread(_query)
        rows = result.data or []
        if not rows:
            return demo
        by_date = {r["date"]: r for r in rows}
        values, labels = [], []
        for i in range(7):
            day = now - timedelta(days=6 - i)
            row = by_date.get(_fmt_date(day))
            values.append(int(row["avg_hrv"]) if row and row.get("avg_hrv") else 0)
            labels.append(_WEEKDAY_ABBR[day.weekday()])
        non_zero = [v for v in values if v > 0]
        mins = [int(r["min_hrv"]) for r in rows if r.get("min_hrv")]
        maxs = [int(r["max_hrv"]) for r in rows if r.get("max_hrv")]
        return {
            "type": "hrv_trend",
            "data": {
                "avg": round(sum(non_zero) / len(non_zero)) if non_zero else 0,
                "min": min(mins) if mins else 0,
                "max": max(maxs) if maxs else 0,
                "unit": "ms",
                "values": values,
                "labels": labels,
            }
        }
    except Exception as e:
        print(f"Error fetching HRV card data: {e}")
        return demo

async def get_bp_card_data(user_id: str) -> Optional[dict[str, Any]]:
    # No hardcoded/demo fallback for this card (unlike the others above) —
    # blood pressure readings are sparse and irregular enough that a fake
    # "118/76, 7 day trend" looked indistinguishable from real data and
    # actively contradicted the chat reply when no real BP data existed.
    # Returning None here means chat_screen.dart's `if (card != null)` check
    # simply skips rendering the card — confirmed this is already handled
    # correctly on the frontend, no card is safer than a fabricated one.
    if not user_id or user_id == "anonymous":
        return None
    try:
        now = dt.utcnow()

        def _query():
            return supabase.table("user_bp").select("*").eq("user_id", user_id)\
                .gte("measured_at", (now - timedelta(days=6)).isoformat())\
                .order("measured_at", desc=False).execute()

        result = await asyncio.to_thread(_query)
        rows = result.data or []
        if not rows:
            return None

        # Multiple BP readings can land on the same calendar day (confirmed
        # via direct query: users often take several readings within minutes
        # of each other). Keep only the LATEST reading per day so a "7 day
        # trend" actually spans 7 distinct calendar days instead of the last
        # 7 raw rows, which could all fall within 1-2 days and repeat the
        # same weekday label (e.g. "Sat, Sun, Sun, Sun, Sun, Sun, Sun").
        # Mirrors the by_date grouping pattern already used in
        # get_hr_card_data / get_spo2_card_data / get_hrv_card_data / steps.
        by_date: dict[str, dict] = {}
        for r in rows:
            try:
                measured_dt = dt.fromisoformat(str(r.get("measured_at")).replace("Z", "+00:00"))
            except Exception:
                continue
            date_key = _fmt_date(measured_dt)
            existing = by_date.get(date_key)
            if existing is None or str(r.get("measured_at")) > str(existing.get("measured_at")):
                by_date[date_key] = r

        sbp_values, dbp_values, labels = [], [], []
        for i in range(7):
            day = now - timedelta(days=6 - i)
            row = by_date.get(_fmt_date(day))
            sbp_values.append(int(row["systolic"]) if row and row.get("systolic") else 0)
            dbp_values.append(int(row["diastolic"]) if row and row.get("diastolic") else 0)
            labels.append(_WEEKDAY_ABBR[day.weekday()])

        sbp_nz = [v for v in sbp_values if v > 0]
        dbp_nz = [v for v in dbp_values if v > 0]
        if not sbp_nz and not dbp_nz:
            return None

        return {
            "type": "bp_trend",
            "data": {
                "sbp_avg": round(sum(sbp_nz) / len(sbp_nz)) if sbp_nz else 0,
                "dbp_avg": round(sum(dbp_nz) / len(dbp_nz)) if dbp_nz else 0,
                "sbp_values": sbp_values,
                "dbp_values": dbp_values,
                "labels": labels,
            }
        }
    except Exception as e:
        print(f"Error fetching BP card data: {e}")
        return None

async def get_steps_card_data(user_id: str) -> Optional[dict[str, Any]]:
    demo = {
        "type": "steps_trend",
        "data": {
            "avg": 6400, "unit": "steps",
            "values": [5200, 7100, 6800, 4900, 8200, 6300, 6400],
            "labels": _WEEKDAY_ABBR,
        }
    }
    if not user_id or user_id == "anonymous":
        return demo
    try:
        now = dt.utcnow()

        def _query():
            return supabase.table("user_steps").select("*").eq("user_id", user_id)\
                .gte("date", _fmt_date(now - timedelta(days=6)))\
                .lte("date", _fmt_date(now)).order("date", desc=False).execute()

        result = await asyncio.to_thread(_query)
        rows = result.data or []
        if not rows:
            return demo
        by_date = {r["date"]: r for r in rows}
        values, labels = [], []
        for i in range(7):
            day = now - timedelta(days=6 - i)
            row = by_date.get(_fmt_date(day))
            values.append(int(row["steps"]) if row and row.get("steps") else 0)
            labels.append(_WEEKDAY_ABBR[day.weekday()])
        non_zero = [v for v in values if v > 0]
        return {
            "type": "steps_trend",
            "data": {
                "avg": round(sum(non_zero) / len(non_zero)) if non_zero else 0,
                "unit": "steps",
                "values": values,
                "labels": labels,
            }
        }
    except Exception as e:
        print(f"Error fetching steps card data: {e}")
        return demo

async def get_temperature_card_data(user_id: str) -> Optional[dict[str, Any]]:
    if not user_id or user_id == "anonymous":
        return None
    try:
        now = dt.utcnow()
        cutoff_str = _fmt_date(now - timedelta(days=6))

        def _query():
            return supabase.table("user_temp").select("*").eq("user_id", user_id)\
                .gte("measured_at", cutoff_str)\
                .order("measured_at", desc=False).execute()

        result = await asyncio.to_thread(_query)
        rows = result.data or []
        if not rows:
            return None

        by_date: dict[str, list[float]] = {}
        for r in rows:
            try:
                measured_dt = dt.fromisoformat(str(r.get("measured_at")).replace("Z", "+00:00"))
            except Exception:
                continue
            date_key = _fmt_date(measured_dt)
            val = r.get("value_c")
            if val is not None:
                by_date.setdefault(date_key, []).append(float(val))

        values, labels = [], []
        for i in range(7):
            day = now - timedelta(days=6 - i)
            day_vals = by_date.get(_fmt_date(day), [])
            avg_day_val = round(sum(day_vals) / len(day_vals), 1) if day_vals else 0.0
            values.append(avg_day_val)
            labels.append(_WEEKDAY_ABBR[day.weekday()])

        non_zero = [v for v in values if v > 0]
        if not non_zero:
            return None

        all_vals = [float(r["value_c"]) for r in rows if r.get("value_c") is not None]
        return {
            "type": "temperature_trend",
            "data": {
                "avg": round(sum(non_zero) / len(non_zero), 1),
                "min": min(all_vals) if all_vals else 0.0,
                "max": max(all_vals) if all_vals else 0.0,
                "unit": "°C",
                "values": values,
                "labels": labels,
            }
        }
    except Exception as e:
        print(f"Error fetching temperature card data: {e}")
        return None

async def get_stress_card_data(user_id: str) -> Optional[dict[str, Any]]:
    if not user_id or user_id == "anonymous":
        return None
    try:
        now = dt.utcnow()
        cutoff_str = _fmt_date(now - timedelta(days=6))

        def _query():
            return supabase.table("user_stress").select("*").eq("user_id", user_id)\
                .gte("measured_at", cutoff_str)\
                .order("measured_at", desc=False).execute()

        result = await asyncio.to_thread(_query)
        rows = result.data or []
        if not rows:
            return None

        by_date: dict[str, list[int]] = {}
        for r in rows:
            try:
                measured_dt = dt.fromisoformat(str(r.get("measured_at")).replace("Z", "+00:00"))
            except Exception:
                continue
            date_key = _fmt_date(measured_dt)
            val = r.get("stress_value")
            if val is not None:
                by_date.setdefault(date_key, []).append(int(val))

        values, labels = [], []
        for i in range(7):
            day = now - timedelta(days=6 - i)
            day_vals = by_date.get(_fmt_date(day), [])
            avg_day_val = round(sum(day_vals) / len(day_vals)) if day_vals else 0
            values.append(avg_day_val)
            labels.append(_WEEKDAY_ABBR[day.weekday()])

        non_zero = [v for v in values if v > 0]
        if not non_zero:
            return None

        all_vals = [int(r["stress_value"]) for r in rows if r.get("stress_value") is not None]
        return {
            "type": "stress_trend",
            "data": {
                "avg": round(sum(non_zero) / len(non_zero)),
                "min": min(all_vals) if all_vals else 0,
                "max": max(all_vals) if all_vals else 0,
                "unit": "",
                "values": values,
                "labels": labels,
            }
        }
    except Exception as e:
        print(f"Error fetching stress card data: {e}")
        return None

async def get_cycle_card_data(user_id: str) -> Optional[dict[str, Any]]:
    demo = {
        "type": "cycle_trend",
        "data": {
            "period_start": "2026-07-17",
            "cycle_length": 28,
            "period_length": 5,
            "current_day": 14,
            "phase": "Ovulation Window",
            "days_until_next": 14,
        }
    }
    if not user_id or user_id == "anonymous":
        return demo
    try:
        def _query():
            return supabase.table("user_cycles").select("*").eq("user_id", user_id)\
                .order("period_start", desc=True).limit(1).execute()

        result = await asyncio.to_thread(_query)
        rows = result.data or []
        if not rows:
            return demo

        c = rows[0]
        p_start_str = c.get("period_start")
        cycle_len = c.get("cycle_length") or 28
        period_len = c.get("period_length") or 5

        current_day = 1
        days_until_next = cycle_len
        phase = "Follicular Phase"

        if p_start_str:
            try:
                p_start_dt = dt.strptime(p_start_str, "%Y-%m-%d")
                today = dt.utcnow().date()
                delta_days = (today - p_start_dt.date()).days
                if delta_days >= 0:
                    current_day = (delta_days % cycle_len) + 1
                    days_until_next = cycle_len - (delta_days % cycle_len)

                    if current_day <= period_len:
                        phase = "Menstrual Phase"
                    elif current_day <= 13:
                        phase = "Follicular Phase"
                    elif current_day <= 16:
                        phase = "Ovulation Window"
                    else:
                        phase = "Luteal Phase"
            except Exception:
                pass

        return {
            "type": "cycle_trend",
            "data": {
                "period_start": p_start_str or "Unknown",
                "cycle_length": cycle_len,
                "period_length": period_len,
                "current_day": current_day,
                "phase": phase,
                "days_until_next": days_until_next,
            }
        }
    except Exception as e:
        print(f"Error fetching cycle card data: {e}")
        return demo

async def run_agent(message: str, user_id: str) -> tuple[str, Optional[dict[str, Any]]]:
    try:
        t_start = time.monotonic()

        # 1. Fast-path emergency check in Python (0.001s, 0 LLM calls)
        emerg_res = check_emergency.invoke(message)
        if "EMERGENCY DETECTED" in emerg_res:
            return emerg_res, None

        # 2. Pre-fetch patient biometrics directly (0 LLM calls)
        patient_data = await asyncio.to_thread(get_patient_data.invoke, user_id)
        
        # 3. Parallel card data lookup
        msg_lower = message.lower()
        card_task = None
        if "sleep" in msg_lower:
            card_task = get_sleep_card_data(user_id)
        elif any(k in msg_lower for k in ["blood pressure", "systolic", "diastolic"]) or re.search(r'\bbp\b', msg_lower):
            card_task = get_bp_card_data(user_id)
        elif any(k in msg_lower for k in ["spo2", "sp02", "blood oxygen", "oxygen level", "oxygen saturation"]):
            card_task = get_spo2_card_data(user_id)
        elif any(k in msg_lower for k in ["hrv", "heart rate variability", "variability"]):
            card_task = get_hrv_card_data(user_id)
        elif any(k in msg_lower for k in ["heart rate", "pulse", "bpm", "snore", "snoring"]):
            card_task = get_hr_card_data(user_id)
        elif any(k in msg_lower for k in ["steps", "walked", "walking", "step count"]):
            card_task = get_steps_card_data(user_id)
        elif any(k in msg_lower for k in ["temperature", "temp", "fever", "body temp", "body temperature"]):
            card_task = get_temperature_card_data(user_id)
        elif any(k in msg_lower for k in ["stress", "stress level", "anxiety", "stressed"]):
            card_task = get_stress_card_data(user_id)
        elif any(k in msg_lower for k in ["period", "cycle", "menstrual", "menstruation", "ovulation", "pms", "fertile", "women health"]):
            card_task = get_cycle_card_data(user_id)

        # 4. Single direct LLM call with complete grounded context (1 LLM call total!)
        llm = get_medxai_llm()
        full_user_content = f"PATIENT DATA:\n{patient_data}\n\nUSER QUESTION:\n{message}"
        
        if card_task:
            llm_res, card = await asyncio.gather(
                llm.ainvoke([
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=full_user_content)
                ]),
                card_task
            )
        else:
            llm_res = await llm.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=full_user_content)
            ])
            card = None

        t_done = time.monotonic()
        reply = str(llm_res.content).strip()

        print(f"[TIMING] 1-SHOT OPTIMIZED CHAT TOTAL: {t_done - t_start:.2f}s")
        return reply, card
    except Exception as e:
        return f"Agent error: {str(e)}", None


# ── Version D Architecture (Router + Grounding + Safety Fusion) ─────────────

SYSTEM_PROMPT_V2_GROUNDED = SYSTEM_PROMPT + """

IMPORTANT ADDITION FOR VERSION D GROUNDING:
You must perform explicit internal reasoning in three stages (FACTS -> RATIONALE -> ACTION) before outputting the final reply.
Output ONLY a valid JSON object matching this exact schema:

{
  "facts": [
    "List numeric values, dates, or clinical facts verbatim present in PATIENT DATA (or state no data available)"
  ],
  "rationale": "Explicit clinical reasoning connecting the facts to the user's question",
  "action": "Recommended guidance/next steps for the user",
  "final_reply": "The exact user-facing final reply following the mandatory RESPONSE FORMAT bullet points above"
}

Do NOT wrap the JSON in markdown code blocks if possible. Ensure final_reply strictly follows all response format rules (plain text, hyphens, max 4 bullets, no markdown, ending with mandatory disclaimer bullet).
"""

def compute_grounding_score(final_reply: str, facts: list, patient_data: str) -> dict:
    """
    Lightweight script-based grounding verification score for Version D.
    Extracts all numeric values from final_reply and checks if they appear verbatim
    in patient_data or in the facts list.
    """
    if not final_reply:
        return {
            "grounding_score": 1.0,
            "total_numbers_checked": 0,
            "ungrounded_numbers": []
        }

    # Extract all integers and decimal numbers
    numbers = re.findall(r'\d+(?:\.\d+)?', final_reply)
    total_count = len(numbers)

    if total_count == 0:
        return {
            "grounding_score": 1.0,
            "total_numbers_checked": 0,
            "ungrounded_numbers": []
        }

    facts_str = " ".join(str(f) for f in facts)
    source_corpus = f"{patient_data}\n{facts_str}"

    grounded_count = 0
    ungrounded_numbers = []

    for num in numbers:
        if num in source_corpus:
            grounded_count += 1
        else:
            ungrounded_numbers.append(num)

    score = grounded_count / total_count
    return {
        "grounding_score": round(score, 4),
        "total_numbers_checked": total_count,
        "ungrounded_numbers": ungrounded_numbers
    }


async def run_agent_v2(message: str, user_id: str) -> tuple[str, Optional[dict[str, Any]]]:
    """
    Version D Orchestration Pipeline:
    1. Parallel Query Router + LLM Safety Fusion (asyncio.gather)
    2. Emergency Safety Fusion Gate
    3. Selective Data Fetching (get_patient_data_selective)
    4. Fact -> Rationale -> Action Grounded LLM Response
    5. Server-side Evaluation Logging
    """
    import json
    from agent.router import classify_query_streams
    from agent.tools import check_emergency_llm, check_emergency_fused, get_patient_data_selective

    try:
        t_start = time.monotonic()

        # 1. Step 1 (Router) & Step 4 (Safety Fusion LLM Classifier) run IN PARALLEL
        router_task = asyncio.create_task(classify_query_streams(message))
        safety_llm_task = asyncio.create_task(check_emergency_llm(message))

        # Await safety LLM result first to allow instant emergency short-circuiting
        llm_emerg_res = await safety_llm_task
        t_safety_done = time.monotonic()

        # 2. Safety Fusion Gate (Keyword + LLM classifier)
        is_emergency, emerg_response, safety_meta = check_emergency_fused(message, llm_emerg_res)
        if is_emergency:
            router_task.cancel() # Cancel unneeded router task immediately
            t_done = time.monotonic()
            print(f"\n[VERSION D LOG] Total Emergency Short-Circuit Latency: {t_done - t_start:.3f}s")
            print(f"[VERSION D LOG] Safety Fusion Fired: TRUE | Meta: {safety_meta}")
            print(f"[VERSION D LOG] Grounding: EMERGENCY TRIGGERED -> Short-circuit")
            return emerg_response, None

        # If not an emergency, await router task result
        streams = await router_task
        t_router_done = time.monotonic()

        # 3. Selective Fetch: fetch only tables matching router streams
        t_fetch_start = time.monotonic()
        patient_data_task = asyncio.to_thread(get_patient_data_selective, user_id, streams)

        # 4. Parallel Card Data Lookup
        msg_lower = message.lower()
        card_task = None
        if "sleep" in streams or "sleep" in msg_lower:
            card_task = get_sleep_card_data(user_id)
        elif "bp" in streams or any(k in msg_lower for k in ["blood pressure", "systolic", "diastolic"]) or re.search(r'\bbp\b', msg_lower):
            card_task = get_bp_card_data(user_id)
        elif "spo2" in streams or any(k in msg_lower for k in ["spo2", "sp02", "blood oxygen"]):
            card_task = get_spo2_card_data(user_id)
        elif "hrv" in streams or any(k in msg_lower for k in ["hrv", "variability"]):
            card_task = get_hrv_card_data(user_id)
        elif ("current_hr" in streams or "hr_history" in streams) or any(k in msg_lower for k in ["heart rate", "pulse", "bpm"]):
            card_task = get_hr_card_data(user_id)
        elif "steps" in streams or any(k in msg_lower for k in ["steps", "walked"]):
            card_task = get_steps_card_data(user_id)
        elif "temperature" in streams or any(k in msg_lower for k in ["temperature", "temp", "fever"]):
            card_task = get_temperature_card_data(user_id)
        elif "stress" in streams or any(k in msg_lower for k in ["stress", "anxiety"]):
            card_task = get_stress_card_data(user_id)
        elif "cycles" in streams or any(k in msg_lower for k in ["period", "cycle", "menstrual"]):
            card_task = get_cycle_card_data(user_id)

        # Execute selective fetch + card query concurrently
        if card_task:
            patient_data, card = await asyncio.gather(patient_data_task, card_task)
        else:
            patient_data = await patient_data_task
            card = None
        t_fetch_done = time.monotonic()

        # 5. Grounded Fact -> Rationale -> Action LLM Generation
        t_llm_start = time.monotonic()
        llm = get_medxai_llm()
        full_user_content = f"PATIENT DATA (Selective Streams: {streams}):\n{patient_data}\n\nUSER QUESTION:\n{message}"

        llm_res = None
        for attempt in range(3):
            try:
                llm_res = await llm.ainvoke([
                    SystemMessage(content=SYSTEM_PROMPT_V2_GROUNDED),
                    HumanMessage(content=full_user_content)
                ])
                break
            except Exception as err:
                if "429" in str(err) and attempt < 2:
                    pause_time = 3.5 * (attempt + 1)
                    print(f"[VERSION D LOG] Rate limited (429), pausing {pause_time:.1f}s before retry (attempt {attempt + 1})...")
                    await asyncio.sleep(pause_time)
                else:
                    raise err

        t_llm_done = time.monotonic()
        raw_content = str(llm_res.content).strip() if llm_res else ""
        t_done = time.monotonic()

        facts, rationale, action, final_reply = [], "", "", raw_content
        try:
            start_idx = raw_content.find("{")
            end_idx = raw_content.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = raw_content[start_idx:end_idx + 1]
                parsed_json = json.loads(json_str)
                facts = parsed_json.get("facts", [])
                rationale = parsed_json.get("rationale", "")
                action = parsed_json.get("action", "")
                final_reply = str(parsed_json.get("final_reply", "")).strip()

                if not final_reply:
                    if rationale or action:
                        final_reply = f"{rationale}\n- {action}\n- This is general health information, not medical advice."
                    else:
                        final_reply = raw_content
            else:
                final_reply = raw_content
        except Exception as parse_err:
            print(f"[VERSION D LOG] JSON parse exception ({parse_err}), falling back to raw output.")
            final_reply = raw_content

        # Deduplicate identical lines in final_reply while preserving order
        if final_reply:
            lines = [ln.strip() for ln in final_reply.splitlines() if ln.strip()]
            seen = set()
            deduped = []
            for ln in lines:
                if ln not in seen:
                    seen.add(ln)
                    deduped.append(ln)
            final_reply = "\n".join(deduped)

        # 6. Compute Grounding Verification Score
        grounding_res = compute_grounding_score(final_reply, facts, patient_data)
        total_n = grounding_res["total_numbers_checked"]
        ungrounded = grounding_res["ungrounded_numbers"]
        grounded_n = total_n - len(ungrounded)
        score_val = grounding_res["grounding_score"]

        # 7. Evaluation Server-Side Logging with per-stage timing
        try:
            print("\n==================== [VERSION D EVALUATION LOG] ====================")
            print("--- PER-STAGE TIMING BREAKDOWN ---")
            print(f"1. Stage 1 (Router + Safety LLM Stage): {t_router_done - t_start:.3f} seconds")
            print(f"2. Stage 2 (Supabase Selective Fetch):  {t_fetch_done - t_fetch_start:.3f} seconds")
            print(f"3. Stage 3 (Final LLM Generation):      {t_llm_done - t_llm_start:.3f} seconds")
            print(f"TOTAL PIPELINE EXECUTION LATENCY:      {t_done - t_start:.3f} seconds")
            print("-------------------------------------------------------------------")
            print(f"Router Selected Streams: {streams}")
            print(f"Safety Fusion Signals:   Keyword={safety_meta['keyword_triggered']} | LLM={safety_meta['llm_triggered']} (Conf={safety_meta['llm_confidence']:.2f})")
            print("--- FACT -> RATIONALE -> ACTION BREAKDOWN ---")
            print(f"FACTS:     {str(facts).encode('ascii', 'backslashreplace').decode('ascii')}")
            print(f"RATIONALE: {str(rationale).encode('ascii', 'backslashreplace').decode('ascii')}")
            print(f"ACTION:    {str(action).encode('ascii', 'backslashreplace').decode('ascii')}")
            print(f"Grounding Score: {score_val:.2f} ({grounded_n}/{total_n} numbers verified against source data)")
            if ungrounded:
                print(f"WARNING: Ungrounded numbers detected: {ungrounded}")
            print("-------------------------------------------------------------------")
            print(f"FINAL USER REPLY:\n{str(final_reply).encode('ascii', 'backslashreplace').decode('ascii')}")
            print("====================================================================\n")
        except Exception as log_err:
            print(f"[VERSION D LOG] Logging exception ignored: {log_err}")

        return final_reply, card
    except Exception as e:
        return f"Version D Agent error: {str(e)}", None


# ── Version A Architecture (Plain LLM, No Tools, No Data) ───────────────────

async def run_agent_a(message: str, user_id: str) -> tuple[str, Optional[dict[str, Any]]]:
    """
    Version A — Plain LLM, no tools, no data.
    Raw LLM call with minimal system prompt and user message.
    """
    try:
        llm = get_medxai_llm()
        llm_res = await llm.ainvoke([
            SystemMessage(content="You are a helpful health assistant."),
            HumanMessage(content=message)
        ])
        reply = str(llm_res.content).strip()
        return reply, None
    except Exception as e:
        return f"Version A Agent error: {str(e)}", None


# ── Version B Architecture (Plain RAG, No Biometric Routing) ────────────────

async def run_agent_b(message: str, user_id: str) -> tuple[str, Optional[dict[str, Any]]]:
    """
    Version B — Plain RAG, no biometric routing.
    Retrieves medical knowledge context via search_medical_knowledge tool,
    then passes context + message to LLM with simple system prompt.
    """
    try:
        rag_context = await search_medical_knowledge.ainvoke(message)
        system_prompt = "You are a helpful health assistant. Answer the user's question using the provided medical knowledge context."
        full_user_content = f"MEDICAL KNOWLEDGE CONTEXT:\n{rag_context}\n\nUSER QUESTION:\n{message}"

        llm = get_medxai_llm()
        llm_res = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=full_user_content)
        ])
        reply = str(llm_res.content).strip()
        return reply, None
    except Exception as e:
        return f"Version B Agent error: {str(e)}", None

