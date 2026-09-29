"""
Offline tests for the demo page endpoints (no Supabase, no LLM API).
The synthetic cohort stands in for Supabase (fetch_fall_history is patched with
scripts.fall_demo_data.history_for) and a scripted fake LLM stands in for Groq.

Run:  python -m pytest tests/test_demo_api.py -q
"""
import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import agent.fall_risk as fall_risk
import agent.graph as graph
import main
import routers.demo as demo
from scripts.fall_demo_data import DEMO_USERS, USERS_BY_KEY, history_for
from tests.test_agent_pipeline import FakeLLM

BY_ID = {u["id"]: u for u in DEMO_USERS}
OTHER_ID = "00000000-0000-0000-0000-000000000000"


@pytest.fixture
def client(monkeypatch):
    def fake_fetch(user_id, today, days=45):
        if user_id not in BY_ID:
            return {"profile": {}, "daily": {}, "cycles": [], "events": []}
        return history_for(BY_ID[user_id], today, datetime.now(timezone.utc), include_recent_fall=False)
    monkeypatch.setattr(fall_risk, "fetch_fall_history", fake_fetch)
    monkeypatch.setattr(fall_risk, "_log_result", lambda *a, **k: None)
    monkeypatch.setattr(fall_risk, "get_recent_fall_event", lambda uid, window_minutes=30: None)
    fall_risk._CACHE.clear()
    return TestClient(main.app)


def _no_brand(payload):
    assert "medxai" not in str(payload).lower()


# ── GET /api/v1/demo/users ────────────────────────────────────────────────
def test_demo_users_lists_the_five_demo_users(client):
    r = client.get("/api/v1/demo/users")
    assert r.status_code == 200
    users = r.json()
    assert [u["id"] for u in users] == [u["id"] for u in DEMO_USERS]
    assert {u["scenario"] for u in users} == {u["scenario"] for u in DEMO_USERS}
    _no_brand(users)


# ── GET /api/v1/fall-risk/{user_id} ───────────────────────────────────────
def test_fall_risk_payload_matches_the_engine(client):
    uid = USERS_BY_KEY["luteal_strain"]
    r = client.get(f"/api/v1/fall-risk/{uid}")
    assert r.status_code == 200
    d = r.json()
    engine = fall_risk.get_fall_risk(uid)["result"]              # cached result the endpoint used
    assert d["enough_history"] and d["message"] is None
    assert d["score"] == engine["frs"] and d["tier"] == engine["tier"] and d["coverage"] == engine["coverage_q"]
    assert [l["name"] for l in d["layers"]] == ["Fall history", "Motion", "Physiology", "Recovery"]
    assert d["layers"][0]["weight"] == round(engine["weights"]["FH"], 2)
    assert d["cycle"]["applied"] and d["cycle"]["phase"] == "luteal"
    assert d["cycle"]["score_without_correction"] == engine["cycle"]["frs_without_correction"]
    assert len(d["series"]) == 14 and d["series"][-1]["score"] == d["score"]
    assert abs(sum(c["points"] for c in d["contributors"]) - d["points_total"]) < 0.1 * len(d["contributors"])
    assert {"sentence", "r", "n"} <= set(d["correlations"][0])
    assert [t["name"] for t in d["tiers"]] == ["LOW", "MODERATE", "HIGH"]
    _no_brand(d)


def test_fall_risk_flags_stale_data(client):
    d = client.get(f"/api/v1/fall-risk/{USERS_BY_KEY['stale']}").json()
    assert d["stale"] and d["data_age_days"] > 2 and d["enough_history"]


def test_fall_risk_without_data_returns_friendly_message(client):
    d = client.get(f"/api/v1/fall-risk/{OTHER_ID}").json()
    assert d["enough_history"] is False and d["message"] and d["series"] == []


def test_fall_risk_engine_error_returns_502(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("db down")
    monkeypatch.setattr(fall_risk, "get_fall_risk", boom)
    r = client.get(f"/api/v1/fall-risk/{USERS_BY_KEY['low_risk']}")
    assert r.status_code == 502 and "error" in r.json()


# ── POST /api/v1/demo/chat ────────────────────────────────────────────────
def test_demo_chat_runs_the_pipeline_and_returns_a_sanitized_summary(client, monkeypatch):
    monkeypatch.setattr(graph, "get_medxai_llm", lambda: FakeLLM(fabricate_first=True))
    r = client.post("/api/v1/demo/chat", json={"message": "What's my fall risk today?", "user_id": USERS_BY_KEY["luteal_strain"]})
    assert r.status_code == 200
    d = r.json()
    assert set(d) == {"reply", "card", "pipeline"}
    assert d["card"]["type"] == "fall_risk"
    p = d["pipeline"]
    assert p["streams"] == ["fall_risk"] and not p["is_emergency"]
    assert p["self_verification"]["triggered"] and p["self_verification"]["score_after"] == 1.0
    assert p["grounding_score"] == 1.0 and p["timing_total_s"] is not None
    assert set(p["safety"]) == {"keyword_triggered", "llm_triggered", "llm_confidence", "biometric_triggered", "keyword_overridden"}
    assert "patient_data" not in str(d) and "FALL RISK ASSESSMENT" not in str(d) and "facts" not in p
    _no_brand(d)


def test_demo_chat_biometric_emergency(client, monkeypatch):
    uid = USERS_BY_KEY["recent_fall"]
    monkeypatch.setattr(graph, "get_medxai_llm", lambda: FakeLLM())
    monkeypatch.setattr(fall_risk, "get_recent_fall_event",
                        lambda u, window_minutes=30: {"detected_at": datetime.now(timezone.utc).isoformat(), "peak_g": 3.6} if u == uid else None)
    d = client.post("/api/v1/demo/chat", json={"message": "I'm fine, just a bit shaken", "user_id": uid}).json()
    assert d["reply"].startswith("EMERGENCY DETECTED")
    s = d["pipeline"]["safety"]
    assert d["pipeline"]["is_emergency"] and s["biometric_triggered"] and not s["llm_triggered"] and not s["keyword_triggered"]
    assert d["pipeline"]["grounding_score"] is None


def test_demo_chat_strips_brand_and_never_leaks_meta(client, monkeypatch):
    async def fake_v2(message, user_id, verbose=False, suppress_internal_log=False):
        meta = {"is_emergency": False, "streams": ["sleep"], "patient_data": "SECRET PATIENT DATA 123",
                "router_meta": {"confidence": 0.9, "safety_relevant": False, "widened": False, "raw_streams": ["sleep"]},
                "safety_meta": {"keyword_triggered": [], "llm_triggered": False, "llm_confidence": 0.8,
                                "llm_reason": "internal reasoning", "biometric_triggered": False, "keyword_overridden": False},
                "grounding_res": {"grounding_score": 1.0, "strict_score": 1.0, "total_numbers_checked": 2},
                "self_verification": {"triggered": False, "score_before": 1.0, "score_after": 1.0},
                "facts": ["x"], "rationale": "prompt text", "timing": {"total": 1.234}}
        return "I am MedXAI. You slept well.", None, meta
    monkeypatch.setattr(graph, "run_agent_v2", fake_v2)
    d = client.post("/api/v1/demo/chat", json={"message": "How did I sleep?", "user_id": USERS_BY_KEY["low_risk"]}).json()
    assert d["reply"] == "I am the assistant. You slept well."
    for leaked in ("SECRET PATIENT DATA", "internal reasoning", "prompt text", "raw_streams"):
        assert leaked not in str(d)
    assert d["pipeline"]["timing_total_s"] == 1.23


def test_demo_chat_rejects_non_demo_users_and_reports_agent_errors(client, monkeypatch):
    r = client.post("/api/v1/demo/chat", json={"message": "hi", "user_id": OTHER_ID})
    assert r.status_code == 403

    async def failing(*a, **k):
        return "Version D Agent error: No LLM API key found!", None, {}
    monkeypatch.setattr(graph, "run_agent_v2", failing)
    r = client.post("/api/v1/demo/chat", json={"message": "hi", "user_id": USERS_BY_KEY["low_risk"]})
    assert r.status_code == 502 and "error" in r.json() and "Version D" not in r.text


def test_demo_chat_timeout(client, monkeypatch):
    import asyncio

    async def slow(*a, **k):
        await asyncio.sleep(5)
    monkeypatch.setattr(graph, "run_agent_v2", slow)
    monkeypatch.setattr(demo, "CHAT_TIMEOUT_S", 0.05)
    r = client.post("/api/v1/demo/chat", json={"message": "hi", "user_id": USERS_BY_KEY["low_risk"]})
    assert r.status_code == 504


# ── POST /api/v1/demo/fall-now/{user_id} ──────────────────────────────────
class RecordingSupabase:
    def __init__(self):
        self.inserted = []

    def table(self, name):
        outer = self

        class Q:
            def insert(self, row):
                outer.inserted.append((name, row))
                return self

            def execute(self):
                return SimpleNamespace(data=[])
        return Q()


def test_fall_now_inserts_one_uncancelled_fall_for_a_demo_user(client, monkeypatch):
    fake = RecordingSupabase()
    monkeypatch.setattr(demo, "supabase", fake)
    uid = USERS_BY_KEY["recent_fall"]
    r = client.post(f"/api/v1/demo/fall-now/{uid}")
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] and d["valid_minutes"] == 30
    assert len(fake.inserted) == 1
    table, row = fake.inserted[0]
    assert table == "user_fall_events" and row["user_id"] == uid
    assert row["event_type"] == "fall" and row["user_cancelled"] is False and row["dispatched"] is False
    detected = datetime.fromisoformat(row["detected_at"])
    assert abs((datetime.now(timezone.utc) - detected).total_seconds()) < 60


def test_fall_now_refuses_other_users_and_writes_nothing(client, monkeypatch):
    fake = RecordingSupabase()
    monkeypatch.setattr(demo, "supabase", fake)
    r = client.post(f"/api/v1/demo/fall-now/{OTHER_ID}")
    assert r.status_code == 403 and fake.inserted == []


# ── GET /demo ─────────────────────────────────────────────────────────────
def test_demo_page_is_served_without_brand(client):
    r = client.get("/demo")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "Fall-Risk Prediction — Live Demo" in r.text
    assert "medxai" not in r.text.lower()
    for path in ("/api/v1/demo/users", "/api/v1/fall-risk/", "/api/v1/demo/chat", "/api/v1/demo/fall-now/"):
        assert path.replace("/api/v1", "") in r.text
