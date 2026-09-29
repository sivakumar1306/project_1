"""
GET /api/v1/fall-risk/{user_id}

Fall-risk score for one user, straight from the deterministic engine
(agent/fall_risk.py). Every number in the response is produced by the engine;
this router only reshapes and rounds for display.
"""

import asyncio
import math
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


# ── Step-by-step breakdown for the "How this score is calculated" walkthrough ──
# Every number comes from the engine result or the engine constants; this only reshapes.
LAYER_SOURCES = {                     # citation text only; the ratios come from EVIDENCE_RATIOS
    "FH": {"source": "Deandrea et al. 2010 (history of falls)", "ratio_type": "OR"},
    "MO": {"source": "Deandrea et al. 2010 (gait problems)", "ratio_type": "OR"},
    "P": {"source": "Mol et al. 2019 (orthostatic hypotension)", "ratio_type": "OR"},
    "RC": {"source": "SWAN cohort (poor sleep)", "ratio_type": "RR"},
}


def build_breakdown_payload(user_id: str) -> dict:
    res = fall_risk.get_fall_risk(user_id)["result"]
    enough = not (res["latest_data_date"] is None and res["falls_365"] == 0 and res["near_falls_14"] == 0)
    layers_res, weights = res["layers"], res["weights"]
    points = {c["feature"]: c["points"] for c in res["contributions"]}

    features = []
    for key, spec in fall_risk.FEATURES.items():
        fr = res["features"][key]
        L = spec["layer"]
        f = {"key": key, "label": spec["label"], "unit": spec["unit"], "ndp": spec["ndp"],
             "layer": L, "layer_name": fall_risk.LAYER_NAMES[L], "risk_direction": spec["dir"],
             "nominal_weight": spec["w"], "present": fr["present"]}
        if fr["present"]:
            z_risk = -fr["z"] if spec["dir"] == "low" else fr["z"]
            pw = layers_res[L]["present_weight"]
            f.update({"value": fr["value"], "date": fr["date"], "baseline": fr["baseline"],
                      "z": fr["z"], "z_risk": z_risk, "strain": fr["strain"],
                      # strain above the straight-line value = the SpO2 < 95 % absolute floor
                      "strain_floor_applied": fr["strain"] > fall_risk._strain_from_z(z_risk) + 1e-9,
                      "weight_in_layer": spec["w"] / pw if pw else 0.0,
                      "points": points.get(key, 0.0), "baseline_mode": fr["baseline_mode"],
                      "n_baseline_days": fr["n_baseline"], "reason": None})
        else:
            f.update({"value": None, "baseline": None, "z": None, "z_risk": None, "strain": None,
                      "strain_floor_applied": False, "weight_in_layer": 0.0, "points": 0.0,
                      "baseline_mode": None, "n_baseline_days": None, "reason": fr.get("reason")})
        features.append(f)

    # Motion layer = weighted mean of daily steps and the near-fall component; recover the
    # near-fall strain from the engine's own layer numbers rather than re-implementing it.
    mo_pw = layers_res["MO"]["present_weight"]
    steps_part = sum(f["nominal_weight"] * f["strain"] for f in features if f["layer"] == "MO" and f["present"])
    nf_strain = (layers_res["MO"]["risk"] * mo_pw - steps_part) / fall_risk.NEAR_FALL_WEIGHT
    near_falls = {"count": res["near_falls_14"], "strain": max(0.0, nf_strain),
                  "weight_in_layer": fall_risk.NEAR_FALL_WEIGHT / mo_pw if mo_pw else 0.0,
                  "points": points.get("near_falls", 0.0)}

    layers = []
    for L in LAYER_ORDER:
        ratio = fall_risk.EVIDENCE_RATIOS[L]
        layers.append({"key": L, "name": fall_risk.LAYER_NAMES[L], "evidence_ratio": ratio,
                       "ratio_type": LAYER_SOURCES[L]["ratio_type"], "source": LAYER_SOURCES[L]["source"],
                       "ln_ratio": math.log(ratio), "base_weight": fall_risk.BASE_LAYER_WEIGHTS[L],
                       "coverage_pct": layers_res[L]["coverage"] * 100, "adjusted_weight": weights[L],
                       "risk": layers_res[L]["risk"], "points": weights[L] * layers_res[L]["risk"]})

    cycle = None
    if res.get("cycle"):
        cy = res["cycle"]
        wo = cy.get("frs_without_correction")
        cycle = {"phase": cy["phase"], "day": cy["cycle_day"], "cycle_length": cy["cycle_length"],
                 "applied": cy["applied"], "score_without_correction": wo,
                 "tier_without_correction": fall_risk.tier_for(wo) if wo is not None else None}

    return {
        "user_id": user_id,
        "enough_history": enough,
        "message": None if enough else NO_DATA_MESSAGE,
        "stale": res["stale"],
        "data_age_days": res["data_age_days"],
        "latest_data_date": res["latest_data_date"],
        "features": features,
        "near_falls": near_falls,
        "layers": layers,
        "fall_history": {"falls_365": res["falls_365"], "near_falls_14": res["near_falls_14"],
                         "fh_risk": layers_res["FH"]["risk"], "points": points.get("fall_history", 0.0)},
        "total": {"frs_raw": res["frs_raw"], "frs": res["frs"], "tier": res["tier"],
                  "thresholds": [upper for upper, _ in fall_risk.TIERS[:-1]],
                  "tiers": [{"name": name, "below": upper} for upper, name in fall_risk.TIERS]},
        "cycle": cycle,
        "constants": {"dead_zone_z": fall_risk.DEAD_ZONE_Z, "full_strain_z": fall_risk.FULL_STRAIN_Z,
                      "baseline_days": fall_risk.BASELINE_DAYS, "min_baseline_days": fall_risk.MIN_BASELINE_POINTS},
    }


@router.get("/fall-risk/{user_id}/breakdown")
async def get_fall_risk_breakdown_endpoint(user_id: str):
    try:
        return await asyncio.to_thread(build_breakdown_payload, user_id)
    except Exception as e:
        print(f"[FALL RISK BREAKDOWN] {user_id}: {e}")
        return JSONResponse(status_code=502, content={"error": "Could not load the score breakdown right now. Please retry."})
