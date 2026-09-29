"""
Live-demo endpoints for the five synthetic demo users (scripts/fall_demo_data.py).

  GET  /api/v1/demo/users              the demo users
  POST /api/v1/demo/chat               Version D agent reply + a small, sanitized pipeline summary
  POST /api/v1/demo/fall-now/{user_id} log one fresh uncancelled fall for a demo user (nothing is sent anywhere)

All three only accept the demo user ids, so they cannot touch real users' data.
"""

import asyncio
import inspect
import re
import time
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import agent.fall_risk as fall_risk
import agent.graph as graph
from db.supabase import supabase
from scripts.fall_demo_data import DEMO_USERS, recent_fall_event

router = APIRouter()

DEMO_IDS = {u["id"] for u in DEMO_USERS}
CHAT_TIMEOUT_S = 55                       # the page gives up at 60 s
FALL_WINDOW_MIN = inspect.signature(fall_risk.get_recent_fall_event).parameters["window_minutes"].default
_BRAND = re.compile(r"\bMedXAI\b", re.IGNORECASE)
_AGENT_ERROR = ("Version D Agent error", "Agent error")


def _forbidden():
    return JSONResponse(status_code=403, content={"error": "Only the demo users can be used on this page."})


def _clean(text: str) -> str:
    return _BRAND.sub("the assistant", text or "")


def _r(v, ndp=2):
    return None if v is None else round(float(v), ndp)


def summarize_pipeline(meta: dict) -> dict:
    """Small, display-safe summary of run_agent_v2's verbose meta (no patient data, no prompts)."""
    meta = meta or {}
    router_meta = meta.get("router_meta") or {}
    safety = meta.get("safety_meta") or {}
    sv = meta.get("self_verification") or {}
    grounding = meta.get("grounding_res") or {}
    is_emergency = bool(meta.get("is_emergency"))
    return {
        "is_emergency": is_emergency,
        "streams": list(meta.get("streams") or []),
        "router": {"confidence": _r(router_meta.get("confidence")),
                   "safety_relevant": router_meta.get("safety_relevant"),
                   "widened": bool(router_meta.get("widened"))},
        "safety": {"keyword_triggered": list(safety.get("effective_keyword_triggered", safety.get("keyword_triggered")) or []),
                   "llm_triggered": bool(safety.get("llm_triggered")),
                   "llm_confidence": _r(safety.get("llm_confidence")),
                   "biometric_triggered": bool(safety.get("biometric_triggered")),
                   "keyword_overridden": bool(safety.get("keyword_overridden"))},
        # strict score = numbers in the reply found in the patient data; not applicable to emergency replies
        "grounding_score": None if is_emergency else _r(grounding.get("strict_score", grounding.get("grounding_score"))),
        "numbers_checked": None if is_emergency else grounding.get("total_numbers_checked"),
        "self_verification": {"triggered": bool(sv.get("triggered")), "skipped": sv.get("skipped"),
                              "score_before": _r(sv.get("score_before")), "score_after": _r(sv.get("score_after"))},
        "timing_total_s": _r((meta.get("timing") or {}).get("total")),
    }


@router.get("/demo/users")
async def demo_users():
    return [{"id": u["id"], "name": u["full_name"], "scenario": u["scenario"]} for u in DEMO_USERS]


class DemoChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    user_id: str


@router.post("/demo/chat")
async def demo_chat(req: DemoChatRequest):
    if req.user_id not in DEMO_IDS:
        return _forbidden()
    t0 = time.monotonic()
    try:
        reply, card, meta = await asyncio.wait_for(
            graph.run_agent_v2(req.message, req.user_id, verbose=True, suppress_internal_log=True), CHAT_TIMEOUT_S)
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={"error": "The assistant took too long to answer. Please try again."})
    except Exception as e:
        print(f"[DEMO CHAT] {e}")
        return JSONResponse(status_code=502, content={"error": "The assistant is unavailable right now. Please try again."})
    if str(reply).startswith(_AGENT_ERROR):
        print(f"[DEMO CHAT] {reply}")
        return JSONResponse(status_code=502, content={
            "error": "The assistant could not answer (check the LLM API key in .env and your connection)."})
    pipeline = summarize_pipeline(meta)
    if pipeline["timing_total_s"] is None:
        pipeline["timing_total_s"] = _r(time.monotonic() - t0)
    return {"reply": _clean(str(reply)), "card": card if isinstance(card, dict) else None, "pipeline": pipeline}


@router.post("/demo/fall-now/{user_id}")
async def demo_fall_now(user_id: str):
    """Logs a simulated ring fall event. It only writes one row; no SMS, call or notification is sent."""
    if user_id not in DEMO_IDS:
        return _forbidden()
    row = recent_fall_event(user_id, minutes_ago=0)
    row["source"] = "demo_page"
    try:
        await asyncio.to_thread(lambda: supabase.table("user_fall_events").insert(row).execute())
    except Exception as e:
        print(f"[DEMO FALL-NOW] {e}")
        return JSONResponse(status_code=502, content={
            "error": "Could not log the fall event. Is the database reachable and schema_fall.sql applied?"})
    fall_risk._CACHE.clear()               # the next score reflects the new event
    return {"ok": True, "user_id": user_id, "detected_at": row["detected_at"], "valid_minutes": FALL_WINDOW_MIN}
