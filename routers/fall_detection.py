"""
Live-demo endpoints for Experiment 4: trained ML fall DETECTION on PIFv3.

The page only shows what the committed files contain; nothing is trained or recomputed here.
  pifv3_results.json       pre-registered evaluation (run_pifv3_detection.py)
  pifv3_demo_samples.json  windows from the 4 held-out participants (make_pifv3_demo.py)

  GET  /api/v1/detection/results                         the evaluation
  GET  /api/v1/detection/chart                           evaluation_charts/pifv3_detection.png
  GET  /api/v1/detection/samples                         [{sample_id, pid}] (no labels, no predictions)
  GET  /api/v1/detection/sample/{sample_id}              the full sample record
  POST /api/v1/detection/sample/{sample_id}/send-to-safety
       only for a FALL prediction: logs one uncancelled fall for the Lakshmi demo user.
       It only writes one row; no SMS, call or notification is sent.

If a file is missing the endpoints return {"available": false} and the rest of the page keeps working.
"""

import asyncio
import json
import os

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

import agent.fall_risk as fall_risk
from db.supabase import supabase
from scripts.fall_demo_data import USERS_BY_KEY, recent_fall_event

router = APIRouter()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(ROOT, "pifv3_results.json")
SAMPLES_PATH = os.path.join(ROOT, "pifv3_demo_samples.json")
CHART_PATH = os.path.join(ROOT, "evaluation_charts", "pifv3_detection.png")
SOURCE = "ml_detector_demo"
UNAVAILABLE = {"available": False}


def _load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        print(f"[DETECTION] cannot read {os.path.basename(path)}: {e}")
        return None


def _find(sample_id):
    data = _load(SAMPLES_PATH)
    if data is None:
        return None, None
    return data, next((s for s in data.get("samples", []) if s.get("sample_id") == sample_id), None)


def _not_found():
    return JSONResponse(status_code=404, content={"error": "Unknown sample."})


@router.get("/detection/results")
async def detection_results():
    data = _load(RESULTS_PATH)
    return data if data is not None else UNAVAILABLE


@router.get("/detection/chart", include_in_schema=False)
async def detection_chart():
    if not os.path.exists(CHART_PATH):
        return JSONResponse(status_code=404, content=UNAVAILABLE)
    return FileResponse(CHART_PATH, media_type="image/png")


@router.get("/detection/samples")
async def detection_samples():
    data = _load(SAMPLES_PATH)
    if data is None:
        return UNAVAILABLE
    return {"available": True,
            "trained_on_participants": data.get("trained_on_participants"),
            "held_out_participants": data.get("held_out_participants"),
            "held_out_windows": data.get("held_out_windows"),
            "held_out_accuracy": data.get("held_out_accuracy"),
            "model": data.get("model"),
            "samples": [{"sample_id": s["sample_id"], "pid": s["pid"]} for s in data.get("samples", [])]}


@router.get("/detection/sample/{sample_id}")
async def detection_sample(sample_id: str):
    data, sample = _find(sample_id)
    if data is None:
        return UNAVAILABLE
    return sample if sample is not None else _not_found()


@router.post("/detection/sample/{sample_id}/send-to-safety")
async def detection_send_to_safety(sample_id: str):
    data, sample = _find(sample_id)
    if data is None:
        return UNAVAILABLE
    if sample is None:
        return _not_found()
    if sample.get("predicted") != "FALL":
        return JSONResponse(status_code=400, content={"error": "Only a FALL prediction can be sent to the safety system."})
    uid = USERS_BY_KEY["recent_fall"]
    row = recent_fall_event(uid, minutes_ago=0)
    row["source"] = SOURCE
    try:
        await asyncio.to_thread(lambda: supabase.table("user_fall_events").insert(row).execute())
    except Exception as e:
        print(f"[DETECTION SEND-TO-SAFETY] {e}")
        return JSONResponse(status_code=502, content={
            "error": "Could not log the fall event. Is the database reachable and schema_fall.sql applied?"})
    fall_risk._CACHE.clear()               # the next score reflects the new event
    return {"ok": True, "user_id": uid, "sample_id": sample_id, "detected_at": row["detected_at"]}
