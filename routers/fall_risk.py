"""
GET /api/v1/fall-risk/{user_id}

Fall-risk score for one user, straight from the deterministic engine
(agent/fall_risk.py). Every number in the response is produced by the engine;
this router only reshapes and rounds for display.
"""

import asyncio
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

import agent.fall_risk as fall_risk

router = APIRouter()

LAYER_ORDER = ("FH", "MO", "P", "RC")
SERIES_DAYS = 14
NO_DATA_MESSAGE = "No ring data found for this user yet. Sync the ring, then refresh to see a fall-risk score."


def _r(v, ndp=1):
    return None if v is None else round(float(v), ndp)


def _fmt(v, ndp):
    """Engine value rounded the same way the engine formats it for the LLM context."""
    if v is None:
        return None
    return round(float(v), ndp) if ndp else int(round(float(v)))


def build_fall_risk_payload(user_id: str) -> dict:
    fr = fall_risk.get_fall_risk(user_id)
    res = fr["result"]
    today = datetime.utcnow().date()

    enough = not (res["latest_data_date"] is None and res["falls_365"] == 0 and res["near_falls_14"] == 0)

    layers = [{"key": k, "name": fall_risk.LAYER_NAMES[k], "risk": _r(res["layers"][k]["risk"]),
               "weight": _r(res["weights"][k], 2), "coverage": _r(res["layers"][k]["coverage"], 2)}
              for k in LAYER_ORDER]

    contributors = []
    for c in res["contributions"]:
        if c["points"] <= 0:
            continue
        ndp = c.get("ndp", 0)
        contributors.append({
            "feature": c["feature"], "label": c["label"], "layer": fall_risk.LAYER_NAMES[c["layer"]],
            "points": _r(c["points"]), "value": _fmt(c.get("value"), ndp),
            "baseline": _fmt(c.get("baseline"), ndp), "unit": c.get("unit", ""),
        })

    cycle = None
    if res.get("cycle"):
        cy = res["cycle"]
        cycle = {"phase": cy["phase"], "cycle_day": cy["cycle_day"], "cycle_length": cy["cycle_length"],
                 "applied": cy["applied"], "score_without_correction": cy.get("frs_without_correction")}

    series = []
    if enough:
        history = fall_risk.fetch_fall_history(user_id, today)
        series = [{"date": p["date"], "score": p["frs"], "tier": p["tier"]}
                  for p in fall_risk.compute_fall_risk_series(history, today, SERIES_DAYS)]

    return {
        "user_id": user_id,
        "enough_history": enough,
        "message": None if enough else NO_DATA_MESSAGE,
        "score": res["frs"],
        "tier": res["tier"],
        "tiers": [{"name": name, "below": upper} for upper, name in fall_risk.TIERS],
        "coverage": res["coverage_q"],
        "low_confidence": res["low_confidence"],
        "stale": res["stale"],
        "latest_data_date": res["latest_data_date"],
        "data_age_days": res["data_age_days"],
        "layers": layers,
        "contributors": contributors,
        "points_total": _r(res["frs_raw"]),
        "falls_365": res["falls_365"],
        "near_falls_14": res["near_falls_14"],
        "cycle": cycle,
        "trend_7d": fr["trend_7d"],
        "series": series,
        "correlations": [{"sentence": c["sentence"], "r": _r(c["r"], 2), "n": c["n"]} for c in fr["correlations"]],
    }


@router.get("/fall-risk/{user_id}")
async def get_fall_risk_endpoint(user_id: str):
    try:
        # The engine's Supabase calls are blocking; keep them off the event loop.
        return await asyncio.to_thread(build_fall_risk_payload, user_id)
    except Exception as e:
        print(f"[FALL RISK API] {user_id}: {e}")
        return JSONResponse(status_code=502, content={"error": "Could not compute the fall-risk score right now. Please retry."})
