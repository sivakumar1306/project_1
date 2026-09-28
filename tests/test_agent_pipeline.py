"""
Offline end-to-end test of the Review-II Version D pipeline.
A scripted fake LLM stands in for Groq and the synthetic cohort stands in for
Supabase, so this checks the wiring (router -> tri-modal safety -> selective
fetch -> fall engine -> grounded generation -> self-verification) without any
network access or API quota.

Run:  python -m pytest tests/test_agent_pipeline.py -q
"""
import asyncio
import json
import os
import re
import sys
from datetime import date, datetime, timezone
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent.fall_risk as fall_risk
import agent.graph as graph
from scripts.fall_demo_data import DEMO_USERS, USERS_BY_KEY, history_for

TODAY = datetime.utcnow().date()
NOW = datetime.now(timezone.utc)
BY_ID = {u["id"]: u for u in DEMO_USERS}


class FakeLLM:
    """Scripted responses keyed on the system prompt of each call."""

    def __init__(self, fabricate_first=True):
        self.fabricate_first = fabricate_first
        self.calls = []

    async def ainvoke(self, messages):
        sys_prompt = messages[0].content
        user_msg = messages[1].content if len(messages) > 1 else ""
        self.calls.append(sys_prompt[:40])
        if sys_prompt.startswith("You are a precise data-stream routing"):
            if "dizzy" in user_msg:
                out = {"streams": ["hrv"], "confidence": 0.55, "safety_relevant": True}
            elif "older people" in user_msg:
                out = {"streams": [], "confidence": 0.95, "safety_relevant": False}
            else:
                out = {"streams": ["fall_risk"], "confidence": 0.93, "safety_relevant": False}
            return SimpleNamespace(content=json.dumps(out))
        if sys_prompt.startswith("You are a medical safety emergency triage"):
            emerg = "can't get up" in user_msg or "on the floor" in user_msg
            return SimpleNamespace(content=json.dumps({"is_emergency": emerg, "confidence": 0.95, "reason": "test"}))
        # grounded generation
        data = user_msg
        m = re.search(r"FALL RISK SCORE \(24h\): (\d+)/100", data)
        score = m.group(1) if m else None
        is_correction = len(messages) > 2
        if self.fabricate_first and not is_correction:
            reply = f"Your fall risk is moderate at {int(score) + 7 if score else 88}.\n- Your HRV dropped 35 percent"
        else:
            reply = (f"Your fall risk score is {score}/100 today." if score else "No fall risk data available.") + \
                    "\n- Prioritise sleep and stand up slowly"
        return SimpleNamespace(content=json.dumps({"facts": [], "rationale": "r", "action": "a", "final_reply": reply}))


def _patch(monkeypatch, llm, recent_fall_for=None):
    def fake_fetch(user_id, today, days=45):
        return history_for(BY_ID[user_id], today, NOW, include_recent_fall=False)
    monkeypatch.setattr(fall_risk, "fetch_fall_history", fake_fetch)
    monkeypatch.setattr(fall_risk, "_log_result", lambda *a, **k: None)
    monkeypatch.setattr(fall_risk, "get_recent_fall_event",
                        lambda uid, window_minutes=30: ({"detected_at": NOW.isoformat(), "peak_g": 3.6}
                                                        if uid == recent_fall_for else None))
    fall_risk._CACHE.clear()
    monkeypatch.setattr(graph, "get_medxai_llm", lambda: llm)


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_fall_query_routes_grounds_and_self_corrects(monkeypatch):
    llm = FakeLLM(fabricate_first=True)
    _patch(monkeypatch, llm)
    reply, card, meta = run(graph.run_agent_v2("What's my fall risk today?", USERS_BY_KEY["luteal_strain"],
                                               verbose=True, suppress_internal_log=True))
    assert meta["streams"] == ["fall_risk"]
    assert card and card["type"] == "fall_risk"
    sv = meta["self_verification"]
    assert sv["triggered"] and sv["score_before"] < 1.0 and sv["score_after"] == 1.0
    assert f"{card['data']['score']}/100" in reply
    assert "FALL RISK SCORE (24h)" in meta["patient_data"]


def test_biometric_only_emergency(monkeypatch):
    llm = FakeLLM()
    uid = USERS_BY_KEY["recent_fall"]
    _patch(monkeypatch, llm, recent_fall_for=uid)
    reply, card, meta = run(graph.run_agent_v2("I'm fine, just a bit shaken", uid, verbose=True, suppress_internal_log=True))
    assert meta["is_emergency"] and meta["safety_meta"]["biometric_triggered"]
    assert not meta["safety_meta"]["keyword_triggered"] and not meta["safety_meta"]["llm_triggered"]
    assert "EMERGENCY DETECTED" in reply


def test_text_emergency_short_circuits(monkeypatch):
    llm = FakeLLM()
    _patch(monkeypatch, llm)
    reply, _, meta = run(graph.run_agent_v2("I fell and can't get up", USERS_BY_KEY["low_risk"],
                                            verbose=True, suppress_internal_log=True))
    assert meta["is_emergency"] and meta["safety_meta"]["keyword_triggered"]


def test_low_confidence_safety_query_widens_fetch(monkeypatch):
    llm = FakeLLM(fabricate_first=False)
    _patch(monkeypatch, llm)
    reply, card, meta = run(graph.run_agent_v2("I feel dizzy when I stand up", USERS_BY_KEY["luteal_strain"],
                                               verbose=True, suppress_internal_log=True))
    rm = meta["router_meta"]
    assert rm["widened"] and rm["raw_streams"] == ["hrv"]
    assert "fall_risk" in meta["streams"] and "spo2" in meta["streams"]


def test_general_question_fetches_nothing_and_no_alarm(monkeypatch):
    llm = FakeLLM(fabricate_first=False)
    _patch(monkeypatch, llm)
    reply, card, meta = run(graph.run_agent_v2("Why do older people fall more often?", USERS_BY_KEY["low_risk"],
                                               verbose=True, suppress_internal_log=True))
    assert not meta["is_emergency"] and meta["streams"] == [] and card is None


class GeneralNumbersLLM(FakeLLM):
    """Answers a general-knowledge question with legitimate numbers that are not patient data."""

    async def ainvoke(self, messages):
        sys_prompt = messages[0].content
        if sys_prompt.startswith("You are a precise data-stream routing") or \
                sys_prompt.startswith("You are a medical safety emergency triage"):
            return await super().ainvoke(messages)
        self.calls.append("generation")
        reply = ("Falls become more common with age.\n- About 1 in 4 adults over 65 falls each year\n"
                 "- Muscle strength drops about 30 percent by age 80")
        return SimpleNamespace(content=json.dumps({"facts": [], "rationale": "r", "action": "a", "final_reply": reply}))


def test_general_question_with_numbers_skips_self_verification(monkeypatch):
    llm = GeneralNumbersLLM()
    _patch(monkeypatch, llm)
    reply, card, meta = run(graph.run_agent_v2("Why do older people fall more often?", USERS_BY_KEY["low_risk"],
                                               verbose=True, suppress_internal_log=True))
    assert meta["streams"] == []
    assert meta["self_verification"] == {"triggered": False, "skipped": "no patient data"}
    assert "65" in reply and "30 percent" in reply                    # general-knowledge numbers kept
    assert llm.calls.count("generation") == 1                          # no corrective regeneration


def test_fetched_data_without_numbers_skips_self_verification(monkeypatch):
    import agent.tools as tools
    llm = GeneralNumbersLLM()
    _patch(monkeypatch, llm)
    monkeypatch.setattr(tools, "get_patient_data_selective",
                        lambda uid, streams: "FALL RISK ASSESSMENT: NO ring data found for this user.")
    reply, card, meta = run(graph.run_agent_v2("What's my fall risk today?", USERS_BY_KEY["low_risk"],
                                               verbose=True, suppress_internal_log=True))
    assert meta["streams"] == ["fall_risk"]
    assert meta["self_verification"]["skipped"] == "no patient data"
    assert llm.calls.count("generation") == 1
