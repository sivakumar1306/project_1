from langchain_core.tools import tool
from datetime import datetime, timezone, timedelta
import re
from zoneinfo import ZoneInfo
from db.supabase import supabase

# All measured_at / date-ish timestamps from Supabase are stored in UTC.
# Convert to IST before handing them to the LLM so replies show the user's
# actual local time instead of raw UTC (e.g. "15:51" showing when it's
# really 21:21 for the user).
def _to_ist(iso_str) -> str:
    if not iso_str:
        return "unknown time"
    try:
        dt = datetime.fromisoformat(str(iso_str).replace("Z", "+00:00"))
        return dt.astimezone(ZoneInfo("Asia/Kolkata")).strftime("%H:%M on %d %B %Y")
    except Exception:
        return str(iso_str)

# IMPORTANT: this must stay `async def`, not a sync function wrapped in
# asyncio.run(). rag.py's retrieve_context_with_expansion() fires off a
# background asyncio.create_task() to expand the knowledge base without
# blocking the user's reply. asyncio.run() creates a brand-new event loop,
# runs the coroutine to completion, and immediately DESTROYS that loop the
# instant it returns — which cancels any task scheduled on it that hasn't
# finished yet. That would silently kill the background expansion task
# before it ever did any work, completely defeating that fix (and likely
# print "Task was destroyed but it is pending!" warnings in the logs).
# By making this an async tool, LangChain's agent.ainvoke() awaits it
# directly on the real, persistent FastAPI/uvicorn event loop instead — so
# the background task actually survives and completes after this call returns.
@tool
async def search_medical_knowledge(query: str) -> str:
    """Search the medical knowledge base for information about symptoms, conditions, treatments, and medications. Automatically expands knowledge base if needed."""
    try:
        from services.rag import retrieve_context_with_expansion
        results = await retrieve_context_with_expansion(query, match_count=3)
        if not results:
            return "No relevant medical information found."
        output = ""
        for r in results:
            output += f"[Source: {r['source']}]\n{r['content']}\n\n"
        return output.strip()
    except Exception as e:
        return f"Medical knowledge search failed: {str(e)}"

# Note on this file's blocking supabase.table(...).execute() calls (below):
# get_patient_data is a synchronous @tool. When the agent is invoked via
# agent.ainvoke() (see graph.py's run_agent), LangChain automatically runs
# sync tool functions in a background thread pool rather than on the main
# event loop — so these blocking DB calls do NOT freeze other concurrent
# FastAPI requests the way the un-wrapped calls in health.py's /insights
# route previously did (those were called directly inside an async route
# handler, with nothing offloading them to a thread). No asyncio wrapping is
# needed here; left as plain synchronous Supabase calls, matching how @tool
# functions are meant to be written. (search_medical_knowledge above is the
# one exception that needed to be async, for the reason explained there.)
def _sleep_line(day: dict) -> str:
    """One RECENT SLEEP line. Some rows store total_duration in seconds; the
    fall-risk engine and the sleep card treat values > 1440 as seconds, so do
    the same here instead of printing e.g. "450h 0m"."""
    total = day.get("total_duration") or 0
    total_min = int(round(total / 60 if total > 1440 else total))
    return f"- {day.get('date')}: {total_min // 60}h {total_min % 60}m, score {day.get('sleep_score')}/100\n"


def _fall_risk_context(user_id: str) -> str:
    try:
        from agent.fall_risk import get_fall_risk
        fr = get_fall_risk(user_id)
        r = fr["result"]
        if r["latest_data_date"] is None and r["falls_365"] == 0 and r["near_falls_14"] == 0:
            return "FALL RISK ASSESSMENT: NO ring data found for this user; a fall-risk score cannot be computed."
        return fr["context"]
    except Exception as e:
        print(f"[FALL RISK] context error: {e}")
        return "FALL RISK ASSESSMENT: unavailable (engine error)."

@tool
def get_patient_data(user_id: str) -> str:
    """Get the patient's health profile and recent smart ring biometric data. Input should be the user's UUID string."""
    try:
        user_id = user_id.strip().strip('"').strip("'")
        context = ""

        from concurrent.futures import ThreadPoolExecutor

        def _fetch_profile():
            try:
                return supabase.table("user_profiles").select("*").eq("id", user_id).maybe_single().execute()
            except Exception:
                return None

        def _fetch_curr_hr():
            try:
                return supabase.table("user_hr_readings").select("*").eq("user_id", user_id).order("measured_at", desc=True).limit(1).execute()
            except Exception:
                return None

        def _fetch_table(table, order_col="date", limit=3):
            try:
                r = supabase.table(table).select("*").eq("user_id", user_id).order(order_col, desc=True).limit(limit).execute()
                return r.data or []
            except Exception:
                return []

        with ThreadPoolExecutor(max_workers=10) as executor:
            f_profile = executor.submit(_fetch_profile)
            f_curr_hr = executor.submit(_fetch_curr_hr)
            f_sleep = executor.submit(_fetch_table, "user_sleep", "date", 3)
            f_hr = executor.submit(_fetch_table, "user_hr", "date", 3)
            f_hrv = executor.submit(_fetch_table, "user_hrv", "date", 3)
            f_spo2 = executor.submit(_fetch_table, "user_spo2", "date", 3)
            f_steps = executor.submit(_fetch_table, "user_steps", "date", 3)
            f_bp = executor.submit(_fetch_table, "user_bp", "measured_at", 1)
            f_temp = executor.submit(_fetch_table, "user_temp", "measured_at", 1)
            f_stress = executor.submit(_fetch_table, "user_stress", "measured_at", 3)
            f_cycles = executor.submit(_fetch_table, "user_cycles", "period_start", 2)
            f_fall = executor.submit(_fall_risk_context, user_id)

            profile = f_profile.result()
            current_hr = f_curr_hr.result()
            sleep = f_sleep.result()
            hr = f_hr.result()
            hrv = f_hrv.result()
            spo2 = f_spo2.result()
            steps = f_steps.result()
            bp = f_bp.result()
            temp = f_temp.result()
            stress = f_stress.result()
            cycles = f_cycles.result()

        if profile and profile.data:
            p = profile.data
            context += f"""
PATIENT PROFILE:
- Name: {p.get('full_name', 'unknown')}
- Age: {p.get('age', 'unknown')}
- Gender: {p.get('gender', 'unknown')}
- Height: {p.get('height_cm', 'unknown')} cm
- Weight: {p.get('weight_kg', 'unknown')} kg
"""

        # Live/current reading
        if current_hr and current_hr.data:
            c = current_hr.data[0]
            measured_at_str = c.get("measured_at")
            staleness_note = ""
            try:
                measured_dt = datetime.fromisoformat(measured_at_str.replace("Z", "+00:00"))
                age_hours = (datetime.now(timezone.utc) - measured_dt).total_seconds() / 3600
                if age_hours > 3:
                    staleness_note = f" [STALE: this reading is {age_hours:.1f} hours old, NOT real-time]"
            except Exception:
                pass
            measured_at_local = _to_ist(measured_at_str)
            context += f"\nCURRENT HEART RATE (most recent single reading in DB): {c.get('value_bpm')} bpm, measured at {measured_at_local}{staleness_note}\n"
        else:
            context += "\nCURRENT HEART RATE: NO reading found in database for this user.\n"

        if sleep:
            context += "\nRECENT SLEEP:\n"
            for day in reversed(sleep):
                context += _sleep_line(day)

        if hr:
            context += "\nHISTORICAL DAILY HEART RATE (NOT the current/live reading):\n"
            for day in reversed(hr):
                context += f"- {day.get('date')}: avg {day.get('avg_hr')} bpm (min {day.get('min_hr')}, max {day.get('max_hr')})\n"

        if hrv:
            context += "\nRECENT HRV:\n"
            for day in reversed(hrv):
                context += f"- {day.get('date')}: avg {day.get('avg_hrv')} ms\n"

        if spo2:
            context += "\nRECENT SPO2:\n"
            for day in reversed(spo2):
                context += f"- {day.get('date')}: avg {day.get('avg_spo2')}%\n"

        if steps:
            context += "\nRECENT STEPS:\n"
            for day in reversed(steps):
                context += f"- {day.get('date')}: {day.get('steps')} steps, {day.get('calories')} kcal\n"

        if bp:
            b = bp[0]
            context += f"\nLATEST BLOOD PRESSURE: {b.get('systolic')}/{b.get('diastolic')} (measured {_to_ist(b.get('measured_at'))})\n"

        if temp:
            t = temp[0]
            context += f"\nLATEST TEMPERATURE: {t.get('value_c')} °C (measured {_to_ist(t.get('measured_at'))})\n"
        else:
            context += "\nLATEST TEMPERATURE: NO reading found in database for this user.\n"

        if stress:
            context += "\nRECENT STRESS LEVEL:\n"
            for s in reversed(stress):
                lbl = f" ({s.get('label')})" if s.get('label') else ""
                context += f"- {_to_ist(s.get('measured_at'))}: level {s.get('stress_value')}{lbl}\n"
        else:
            context += "\nRECENT STRESS LEVEL: NO reading found in database for this user.\n"

        if cycles:
            context += "\nMENSTRUAL CYCLE LOGS:\n"
            for cy in reversed(cycles):
                p_start = cy.get("period_start") or "unknown"
                p_end = cy.get("period_end") or "ongoing"
                c_len = cy.get("cycle_length") or 28
                p_len = cy.get("period_length") or 5
                est_next_str = "unknown"
                days_until = "unknown"
                curr_day_str = "unknown"
                if p_start != "unknown":
                    try:
                        p_start_dt = datetime.strptime(p_start, "%Y-%m-%d")
                        today = datetime.utcnow().date()
                        delta_days = (today - p_start_dt.date()).days
                        if delta_days >= 0:
                            curr_day = (delta_days % c_len) + 1
                            days_until = c_len - (delta_days % c_len)
                            next_dt = today + timedelta(days=days_until)
                            est_next_str = next_dt.strftime("%d %B %Y")
                            curr_day_str = f"Day {curr_day}"
                    except Exception:
                        pass
                context += f"- Period start: {p_start}, period end: {p_end}, cycle length: {c_len} days, period length: {p_len} days. Currently at {curr_day_str}. Estimated next period: {est_next_str} (in {days_until} days).\n"

        fall_ctx = f_fall.result()
        if fall_ctx:
            context += "\n" + fall_ctx + "\n"

        result = context.strip() if context else "No biometric or ring data found for this user. The ring is not connected or has not synced readings. Tell the user: Please connect your ring to view analysis."
        print(f"[get_patient_data] user_id={user_id}\n---TOOL OUTPUT SENT TO LLM---\n{result}\n---END TOOL OUTPUT---")
        return result
    except Exception as e:
        error_msg = f"Failed to fetch patient data: {str(e)}"
        print(f"[get_patient_data] ERROR for user_id={user_id}: {error_msg}")
        return error_msg

def get_patient_data_selective(user_id: str, streams: list[str]) -> str:
    """Get ONLY requested biometric data streams for a patient. Used for selective fetching in Version D."""
    try:
        user_id = user_id.strip().strip('"').strip("'")
        if not streams:
            return "No personal biometric data requested for this general inquiry."

        context = ""
        from concurrent.futures import ThreadPoolExecutor

        def _fetch_profile():
            try:
                return supabase.table("user_profiles").select("*").eq("id", user_id).maybe_single().execute()
            except Exception:
                return None

        def _fetch_curr_hr():
            try:
                return supabase.table("user_hr_readings").select("*").eq("user_id", user_id).neq("source", "demo_seed").order("measured_at", desc=True).limit(1).execute()
            except Exception:
                return None

        def _fetch_table(table, order_col="date", limit=3):
            try:
                r = supabase.table(table).select("*").eq("user_id", user_id).order(order_col, desc=True).limit(limit).execute()
                return r.data or []
            except Exception:
                return []

        futures = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            if "profile" in streams:
                futures["profile"] = executor.submit(_fetch_profile)
            if "current_hr" in streams:
                futures["current_hr"] = executor.submit(_fetch_curr_hr)
            if "sleep" in streams:
                futures["sleep"] = executor.submit(_fetch_table, "user_sleep", "date", 3)
            if "hr_history" in streams:
                futures["hr"] = executor.submit(_fetch_table, "user_hr", "date", 3)
            if "hrv" in streams:
                futures["hrv"] = executor.submit(_fetch_table, "user_hrv", "date", 3)
            if "spo2" in streams:
                futures["spo2"] = executor.submit(_fetch_table, "user_spo2", "date", 3)
            if "steps" in streams:
                futures["steps"] = executor.submit(_fetch_table, "user_steps", "date", 3)
            if "bp" in streams:
                futures["bp"] = executor.submit(_fetch_table, "user_bp", "measured_at", 1)
            if "temperature" in streams:
                futures["temp"] = executor.submit(_fetch_table, "user_temp", "measured_at", 1)
            if "stress" in streams:
                futures["stress"] = executor.submit(_fetch_table, "user_stress", "measured_at", 3)
            if "cycles" in streams:
                futures["cycles"] = executor.submit(_fetch_table, "user_cycles", "period_start", 2)
            if "fall_risk" in streams:
                futures["fall_risk"] = executor.submit(_fall_risk_context, user_id)

            results = {k: f.result() for k, f in futures.items()}

        profile = results.get("profile")
        if profile and profile.data:
            p = profile.data
            context += f"""
PATIENT PROFILE:
- Name: {p.get('full_name', 'unknown')}
- Age: {p.get('age', 'unknown')}
- Gender: {p.get('gender', 'unknown')}
- Height: {p.get('height_cm', 'unknown')} cm
- Weight: {p.get('weight_kg', 'unknown')} kg
"""

        current_hr = results.get("current_hr")
        if "current_hr" in streams:
            if current_hr and current_hr.data:
                c = current_hr.data[0]
                measured_at_str = c.get("measured_at")
                staleness_note = ""
                try:
                    measured_dt = datetime.fromisoformat(measured_at_str.replace("Z", "+00:00"))
                    age_hours = (datetime.now(timezone.utc) - measured_dt).total_seconds() / 3600
                    if age_hours > 3:
                        staleness_note = f" [STALE: this reading is {age_hours:.1f} hours old, NOT real-time]"
                except Exception:
                    pass
                measured_at_local = _to_ist(measured_at_str)
                context += f"\nCURRENT HEART RATE (most recent single reading in DB): {c.get('value_bpm')} bpm, measured at {measured_at_local}{staleness_note}\n"
            else:
                context += "\nCURRENT HEART RATE: NO reading found in database for this user.\n"

        sleep = results.get("sleep")
        if sleep:
            context += "\nRECENT SLEEP:\n"
            for day in reversed(sleep):
                context += _sleep_line(day)

        hr = results.get("hr")
        if hr:
            context += "\nHISTORICAL DAILY HEART RATE (NOT the current/live reading):\n"
            for day in reversed(hr):
                context += f"- {day.get('date')}: avg {day.get('avg_hr')} bpm (min {day.get('min_hr')}, max {day.get('max_hr')})\n"

        hrv = results.get("hrv")
        if hrv:
            context += "\nRECENT HRV:\n"
            for day in reversed(hrv):
                context += f"- {day.get('date')}: avg {day.get('avg_hrv')} ms\n"

        spo2 = results.get("spo2")
        if spo2:
            context += "\nRECENT SPO2:\n"
            for day in reversed(spo2):
                context += f"- {day.get('date')}: avg {day.get('avg_spo2')}%\n"

        steps = results.get("steps")
        if steps:
            context += "\nRECENT STEPS:\n"
            for day in reversed(steps):
                context += f"- {day.get('date')}: {day.get('steps')} steps, {day.get('calories')} kcal\n"

        bp = results.get("bp")
        if bp:
            b = bp[0]
            context += f"\nLATEST BLOOD PRESSURE: {b.get('systolic')}/{b.get('diastolic')} (measured {_to_ist(b.get('measured_at'))})\n"

        temp = results.get("temp")
        if "temperature" in streams:
            if temp:
                t = temp[0]
                context += f"\nLATEST TEMPERATURE: {t.get('value_c')} °C (measured {_to_ist(t.get('measured_at'))})\n"
            else:
                context += "\nLATEST TEMPERATURE: NO reading found in database for this user.\n"

        stress = results.get("stress")
        if "stress" in streams:
            if stress:
                context += "\nRECENT STRESS LEVEL:\n"
                for s in reversed(stress):
                    lbl = f" ({s.get('label')})" if s.get('label') else ""
                    context += f"- {_to_ist(s.get('measured_at'))}: level {s.get('stress_value')}{lbl}\n"
            else:
                context += "\nRECENT STRESS LEVEL: NO reading found in database for this user.\n"

        cycles = results.get("cycles")
        if cycles:
            context += "\nMENSTRUAL CYCLE LOGS:\n"
            for cy in reversed(cycles):
                p_start = cy.get("period_start") or "unknown"
                p_end = cy.get("period_end") or "ongoing"
                c_len = cy.get("cycle_length") or 28
                p_len = cy.get("period_length") or 5
                est_next_str = "unknown"
                days_until = "unknown"
                curr_day_str = "unknown"
                if p_start != "unknown":
                    try:
                        p_start_dt = datetime.strptime(p_start, "%Y-%m-%d")
                        today = datetime.utcnow().date()
                        delta_days = (today - p_start_dt.date()).days
                        if delta_days >= 0:
                            curr_day = (delta_days % c_len) + 1
                            days_until = c_len - (delta_days % c_len)
                            next_dt = today + timedelta(days=days_until)
                            est_next_str = next_dt.strftime("%d %B %Y")
                            curr_day_str = f"Day {curr_day}"
                    except Exception:
                        pass
                context += f"- Period start: {p_start}, period end: {p_end}, cycle length: {c_len} days, period length: {p_len} days. Currently at {curr_day_str}. Estimated next period: {est_next_str} (in {days_until} days).\n"

        fall_ctx = results.get("fall_risk")
        if fall_ctx:
            context += "\n" + fall_ctx + "\n"

        result = context.strip() if context else "No biometric or ring data found for the requested streams."
        print(f"[get_patient_data_selective] user_id={user_id}, streams={streams}")
        return result
    except Exception as e:
        error_msg = f"Failed to fetch selective patient data: {str(e)}"
        print(f"[get_patient_data_selective] ERROR for user_id={user_id}: {error_msg}")
        return error_msg

# ── Shared emergency vocabulary (used by baseline keyword check AND fusion) ──
EMERGENCY_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
    "heart attack", "stroke", "unconscious", "unresponsive", "seizure",
    "severe bleeding", "overdose", "suicidal", "suicide", "kill myself",
    "severe headache", "sudden confusion", "face drooping", "arm weakness",
    "slurred speech", "severe allergic", "anaphylaxis", "stopped breathing",
    # fall-related (Review-II)
    "can't get up", "cannot get up", "can not get up", "hit my head", "fainted",
    "passed out", "blacked out", "fell down",
]
# Regex patterns for fall phrases that need guarding against benign uses
# (e.g. "I fell asleep", "I fell behind").
FALL_PATTERNS = [
    (r"\bi(?:'ve|ve| have)?(?: just)? (?:fell|fallen|had a fall)\b(?!\s+(?:asleep|behind|for|in love|ill|sick|off track))", "i fell"),
    (r"\b(?:on|lying on) the (?:\w+ )?floor\b", "on the floor"),
]
# Phone keyboards often send typographic apostrophes ("can’t get up");
# normalise them so the straight-apostrophe vocabulary above still matches.
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "′": "'"})


def match_emergency_keywords(message: str) -> list[str]:
    msg_lower = message.lower().translate(_APOSTROPHES)
    hits = [kw for kw in EMERGENCY_KEYWORDS if kw in msg_lower]
    for pattern, label in FALL_PATTERNS:
        if re.search(pattern, msg_lower) and label not in hits:
            hits.append(label)
    return hits


@tool
def check_emergency(message: str) -> str:
    """Check if the message contains emergency or life-threatening symptoms that require immediate medical attention."""
    triggered = match_emergency_keywords(message)
    if triggered:
        return f"EMERGENCY DETECTED: {', '.join(triggered)}. This requires IMMEDIATE medical attention. Call emergency services (112 in India) or go to the nearest emergency room NOW. Do not wait."
    return "No emergency detected."

async def check_emergency_llm(message: str) -> tuple[bool, str, float]:
    """
    Fast LLM emergency classification evaluating whether a message describes or implies
    a potential medical emergency (including non-obvious/paraphrased phrasing).
    Returns (is_emergency, reason, confidence).
    """
    import os
    import json
    import re
    from langchain_mistralai import ChatMistralAI
    from langchain_core.messages import SystemMessage, HumanMessage

    try:
        from agent.graph import get_medxai_llm
        llm = get_medxai_llm()

        sys_prompt = """You are a medical safety emergency triage classifier.
Evaluate if the user message describes or implies a potential medical emergency (such as heart attack, stroke, severe respiratory distress, acute anaphylaxis, severe head injury, uncontrollable bleeding, self-harm, or a fall with inability to get up, head impact, fainting or loss of consciousness).
Informational questions about a condition (e.g. "what are the symptoms of a stroke?", "why do older people fall?") and benign phrases (e.g. "I fell asleep") are NOT emergencies.

Output ONLY valid JSON matching this structure:
{
  "is_emergency": true,
  "confidence": 0.95,
  "reason": "crushing chest pressure and arm numbness imply acute coronary event"
}
"""
        import asyncio
        for attempt in range(3):
            try:
                res = await llm.ainvoke([
                    SystemMessage(content=sys_prompt),
                    HumanMessage(content=message)
                ])
                raw = res.content.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
                    raw = re.sub(r"```$", "", raw).strip()

                parsed = json.loads(raw)
                is_emerg = bool(parsed.get("is_emergency", False))
                conf = float(parsed.get("confidence", 0.0))
                reason = str(parsed.get("reason", ""))
                return is_emerg, reason, conf
            except Exception as err:
                if "429" in str(err) and attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                else:
                    raise err

        return False, "Emergency LLM check failed", 0.0
    except Exception as e:
        return False, f"LLM emergency check error: {e}", 0.0

def check_emergency_fused(message: str, llm_res: tuple[bool, str, float], biometric_event: dict | None = None) -> tuple[bool, str, dict]:
    """
    Fused safety classification combining deterministic keyword matching with LLM classification.
    Returns (is_emergency: bool, response_text: str, metadata: dict).
    """
    msg_lower = message.lower()
    triggered_kw = match_emergency_keywords(message)

    is_llm_emerg, llm_reason, llm_conf = llm_res
    llm_triggered = is_llm_emerg and llm_conf >= 0.7

    # Disagreement override check:
    # If keyword matched BUT LLM confidently disagrees (is_llm_emerg is False and llm_conf >= 0.85),
    # treat keyword match as false positive (e.g. informational query "What are the symptoms of a stroke?")
    keyword_overridden = False
    if triggered_kw and (not is_llm_emerg) and (llm_conf >= 0.85):
        keyword_overridden = True
        print(f"[SAFETY] Keyword match overridden — LLM confidently classified as non-emergency (conf={llm_conf:.2f})")
        effective_kw_triggered = []
    else:
        effective_kw_triggered = triggered_kw

    # Third signal (Review-II): physiological evidence independent of the text.
    # An uncancelled fall detected by the ring/phone in the last 30 minutes
    # escalates ANY message, even a calm-sounding one ("I'm fine").
    biometric_triggered = bool(biometric_event)

    is_emergency = bool(effective_kw_triggered) or llm_triggered or biometric_triggered

    meta = {
        "keyword_triggered": triggered_kw,
        "effective_keyword_triggered": effective_kw_triggered,
        "keyword_overridden": keyword_overridden,
        "llm_triggered": llm_triggered,
        "llm_reason": llm_reason,
        "llm_confidence": llm_conf,
        "biometric_triggered": biometric_triggered,
        "biometric_event": {
            "detected_at": biometric_event.get("detected_at"),
            "peak_g": biometric_event.get("peak_g"),
        } if biometric_event else None,
    }

    if is_emergency:
        triggers = []
        if effective_kw_triggered:
            triggers.append(", ".join(effective_kw_triggered))
        if llm_triggered:
            triggers.append(llm_reason)
        if biometric_triggered:
            triggers.append(f"a fall was detected by your ring/phone at {_to_ist(biometric_event.get('detected_at'))} and has not been cancelled")
        desc = "; ".join(triggers)
        if biometric_triggered and not effective_kw_triggered and not llm_triggered:
            response_msg = (f"EMERGENCY DETECTED: {desc}. Even if you feel fine, a fall can cause injuries that are not obvious straight away. "
                            "If you hit your head, feel dizzy or confused, are in pain, or cannot get up, call emergency services (112 in India) now. "
                            "If you are safe, cancel the fall alert in the SUNDR app.")
        else:
            response_msg = f"EMERGENCY DETECTED: {desc}. This requires IMMEDIATE medical attention. Call emergency services (112 in India) or go to the nearest emergency room NOW. Do not wait."
        return True, response_msg, meta

    return False, "No emergency detected.", meta

@tool
def analyze_symptoms(symptoms: str) -> str:
    """Analyze a list of symptoms and return a structured breakdown with possible conditions to investigate."""
    try:
        common_patterns = {
            "thirst,urination,fatigue,blurry vision": "Pattern suggests possible blood sugar issues — consider diabetes screening",
            "chest pain,shortness of breath,sweating": "Pattern suggests possible cardiac issue — seek immediate evaluation",
            "fever,cough,sore throat,runny nose": "Pattern consistent with upper respiratory infection",
            "headache,fever,stiff neck,sensitivity to light": "Pattern warrants urgent evaluation — possible meningitis",
            "fatigue,weight gain,cold intolerance,dry skin": "Pattern suggests possible thyroid dysfunction",
            "anxiety,rapid heartbeat,sweating,trembling": "Pattern consistent with anxiety or panic disorder",
        }
        symptom_lower = symptoms.lower()
        analysis = f"Symptoms reported: {symptoms}\n\n"
        matched = False
        for pattern, suggestion in common_patterns.items():
            pattern_words = pattern.split(",")
            if sum(1 for word in pattern_words if word in symptom_lower) >= 2:
                analysis += f"⚠️ {suggestion}\n"
                matched = True
        if not matched:
            analysis += "No strong pattern match found. Recommend consulting a healthcare provider for proper evaluation.\n"
        analysis += "\n⚠️ This is not a diagnosis. Always consult a qualified medical professional."
        return analysis
    except Exception as e:
        return f"Symptom analysis failed: {str(e)}"