"""
MedXAI Fall-Risk Engine (Review-II)
===================================

Deterministic "compute" half of the compute-then-explain design: this module
computes the fall-risk score; the LLM only explains the numbers it produces.

Pipeline
  1. Personal baselines   - robust median / MAD over the previous 28 days
  2. Trend deviation      - z-score of today's value vs the personal baseline,
                            in the risk direction (e.g. HRV DOWN, resting HR UP)
  3. Cycle-aware baseline - for users with cycle logs, physiology features are
                            compared against same-phase days (phase-matched
                            baseline), falling back to literature offsets
  4. Four layers          - Physiology (P), Recovery (RC), Motion (MO),
                            Fall history (FH)
  5. Evidence weights     - layer weight proportional to ln(odds/risk ratio)
                            from published meta-analyses, renormalised by
                            layer data coverage (adaptive weighting)
  6. Outputs              - FRS (0-100), tier, coverage/confidence Q,
                            additive per-feature attribution, trend

The pure functions (compute_fall_risk, compute_fall_risk_series) take plain
dicts, so they are unit-testable and usable offline without Supabase.
"""

from __future__ import annotations

import math
import statistics
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

# ── Evidence-derived layer weights ──────────────────────────────────────────
# weight_i ∝ ln(ratio_i), normalised to sum to 1.
#   FH: history of falls        OR 2.8   (Deandrea et al., Epidemiology 2010)
#   MO: gait problems           OR 2.1   (Deandrea et al., Epidemiology 2010)
#   P : orthostatic hypotension OR 1.73  (OH & falls meta-analysis, JAMDA)
#   RC: poor sleep              RR 1.27  (SWAN cohort, Innov Aging 2024)
EVIDENCE_RATIOS = {"FH": 2.8, "MO": 2.1, "P": 1.73, "RC": 1.27}
_LN = {k: math.log(v) for k, v in EVIDENCE_RATIOS.items()}
BASE_LAYER_WEIGHTS = {k: v / sum(_LN.values()) for k, v in _LN.items()}

LAYER_NAMES = {"P": "Physiology", "RC": "Recovery", "MO": "Motion", "FH": "Fall history"}

# Feature definitions. "dir" is the risk direction: "low" = a drop below
# baseline adds risk, "high" = a rise above baseline adds risk.
# "min_scale" is a floor on the robust spread so tiny natural variation does
# not produce huge z-scores.
FEATURES: dict[str, dict[str, Any]] = {
    "hrv":         {"layer": "P",  "w": 0.35, "dir": "low",  "label": "HRV",                "unit": "ms",    "min_scale": 3.0,  "ndp": 0},
    "rhr":         {"layer": "P",  "w": 0.25, "dir": "high", "label": "resting heart rate", "unit": "bpm",   "min_scale": 1.5,  "ndp": 0},
    "spo2":        {"layer": "P",  "w": 0.25, "dir": "low",  "label": "SpO2",               "unit": "%",     "min_scale": 1.0,  "ndp": 0},
    "temp":        {"layer": "P",  "w": 0.15, "dir": "high", "label": "skin temperature",   "unit": "°C",    "min_scale": 0.1,  "ndp": 1},
    "sleep_min":   {"layer": "RC", "w": 0.40, "dir": "low",  "label": "sleep duration",     "unit": "min",   "min_scale": 20.0, "ndp": 0},
    "sleep_score": {"layer": "RC", "w": 0.30, "dir": "low",  "label": "sleep score",        "unit": "/100",  "min_scale": 3.0,  "ndp": 0},
    "stress":      {"layer": "RC", "w": 0.30, "dir": "high", "label": "stress level",       "unit": "",      "min_scale": 3.0,  "ndp": 0},
    "steps":       {"layer": "MO", "w": 0.50, "dir": "low",  "label": "daily steps",        "unit": "steps", "min_scale": 500., "ndp": 0},
}
NEAR_FALL_WEIGHT = 0.50          # second Motion component (impact events)
PHYSIO_CYCLE_FEATURES = ("hrv", "rhr", "temp")

# Literature-informed luteal-phase offsets used only when too few same-phase
# history days exist for a phase-matched baseline. Luteal phase: skin temp
# ~+0.3-0.4 °C, resting HR +2-5 bpm, vagal HRV lower (Oura-ring study,
# IJWH 2022; wearable HRV living systematic review, Sports Med 2026).
LUTEAL_OFFSETS = {"temp": 0.3, "rhr": 2.0, "hrv_frac": -0.07}

BASELINE_DAYS = 28
MIN_BASELINE_POINTS = 7
MIN_PHASE_POINTS = 5
STALE_AFTER_DAYS = 2           # latest data older than this -> [STALE]
MISSING_AFTER_DAYS = 7         # feature value older than this -> treated missing
DEAD_ZONE_Z = 0.5              # |z| below this = normal day-to-day noise
FULL_STRAIN_Z = 3.0            # z at which strain reaches 100

TIERS = [(40, "LOW"), (70, "MODERATE"), (101, "HIGH")]


# ── Helpers ────────────────────────────────────────────────────────────────

def _to_date(d: Any) -> Optional[date]:
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    try:
        return datetime.fromisoformat(str(d).replace("Z", "+00:00")).date()
    except Exception:
        try:
            return datetime.strptime(str(d)[:10], "%Y-%m-%d").date()
        except Exception:
            return None


def _to_dt(d: Any) -> Optional[datetime]:
    if d is None:
        return None
    if isinstance(d, datetime):
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    try:
        v = datetime.fromisoformat(str(d).replace("Z", "+00:00"))
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _robust_baseline(values: list[float], min_scale: float) -> tuple[float, float]:
    """Median and MAD-based spread (1.4826*MAD ≈ SD for normal data)."""
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values])
    return med, max(1.4826 * mad, min_scale)


def _strain_from_z(z_risk: float) -> float:
    """Map a risk-direction z-score to 0-100 strain with a noise dead-zone."""
    s = (z_risk - DEAD_ZONE_Z) / (FULL_STRAIN_Z - DEAD_ZONE_Z)
    return max(0.0, min(1.0, s)) * 100.0


def _fmt(v: float, ndp: int) -> str:
    return f"{v:.{ndp}f}" if ndp else f"{int(round(v))}"


def tier_for(score: float) -> str:
    for upper, name in TIERS:
        if score < upper:
            return name
    return "HIGH"


# ── Menstrual cycle phase ──────────────────────────────────────────────────

def cycle_phase(d: date, cycles: list[dict]) -> Optional[dict]:
    """
    Phase for date d from the cycle log. Uses the latest logged period_start
    on or before d; dates before the first log are projected backwards using
    that cycle's length. Returns None when the phase can't be determined.
    """
    starts = []
    for c in cycles or []:
        ps = _to_date(c.get("period_start"))
        if ps:
            starts.append((ps, int(c.get("cycle_length") or 28), int(c.get("period_length") or 5)))
    if not starts:
        return None
    starts.sort()

    prior = [s for s in starts if s[0] <= d]
    if prior:
        start, clen, plen = prior[-1]
    else:
        start, clen, plen = starts[0]
        while start > d:
            start -= timedelta(days=clen)

    day = (d - start).days + 1
    if day > clen + 10:          # missed logs: phase unreliable
        return None
    ov = max(clen - 14, plen + 2)  # luteal phase ~14 days
    if day <= plen:
        phase = "menstrual"
    elif day < ov - 1:
        phase = "follicular"
    elif day <= ov + 1:
        phase = "ovulatory"
    else:
        phase = "luteal"
    return {"phase": phase, "cycle_day": day, "cycle_length": clen}


def _cycle_eligible(profile: dict, cycles: list[dict]) -> bool:
    gender = str((profile or {}).get("gender") or "").strip().lower()
    return bool(cycles) and gender.startswith("f")


# ── Core computation (pure) ────────────────────────────────────────────────

def compute_fall_risk(history: dict, today: date, apply_cycle_correction: bool = True,
                      _include_uncorrected: bool = True) -> dict:
    """
    history = {
      "profile": {...},
      "daily":   {"YYYY-MM-DD": {"hrv":.., "rhr":.., "spo2":.., "temp":..,
                                 "sleep_min":.., "sleep_score":.., "stress":.., "steps":..}},
      "cycles":  [{"period_start":.., "cycle_length":.., "period_length":..}],
      "events":  [{"detected_at":.., "event_type": "fall"|"near_fall", "user_cancelled": bool}],
    }
    """
    daily = {}
    for k, v in (history.get("daily") or {}).items():
        dk = _to_date(k)
        if dk and dk <= today:
            daily[dk] = v
    profile = history.get("profile") or {}
    cycles = history.get("cycles") or []
    events = history.get("events") or []

    use_cycle = apply_cycle_correction and _cycle_eligible(profile, cycles)
    today_phase = cycle_phase(today, cycles) if _cycle_eligible(profile, cycles) else None

    feature_results: dict[str, dict] = {}
    latest_dates: list[date] = []

    for key, spec in FEATURES.items():
        # current value = most recent value on/before today
        cur_date, cur_val = None, None
        for dk in sorted(daily.keys(), reverse=True):
            v = daily[dk].get(key)
            if v is not None:
                cur_date, cur_val = dk, float(v)
                break
        if cur_val is None or (today - cur_date).days > MISSING_AFTER_DAYS:
            feature_results[key] = {"present": False, "reason": "no recent data"}
            continue
        latest_dates.append(cur_date)

        window_start = cur_date - timedelta(days=BASELINE_DAYS)
        hist = [(dk, float(daily[dk][key])) for dk in daily
                if window_start <= dk < cur_date and daily[dk].get(key) is not None]
        if len(hist) < MIN_BASELINE_POINTS:
            feature_results[key] = {"present": False, "reason": "insufficient baseline"}
            continue

        baseline_mode = "personal"
        values = [v for _, v in hist]
        offset = 0.0
        if use_cycle and key in PHYSIO_CYCLE_FEATURES and today_phase:
            same = []
            for dk, v in hist:
                ph = cycle_phase(dk, cycles)
                if ph and ph["phase"] == today_phase["phase"]:
                    same.append(v)
            if len(same) >= MIN_PHASE_POINTS:
                values = same
                baseline_mode = f"phase-matched ({today_phase['phase']})"
            elif today_phase["phase"] == "luteal":
                baseline_mode = "luteal offset"
                if key == "temp":
                    offset = LUTEAL_OFFSETS["temp"]
                elif key == "rhr":
                    offset = LUTEAL_OFFSETS["rhr"]

        med, scale = _robust_baseline(values, spec["min_scale"])
        if baseline_mode == "luteal offset" and key == "hrv":
            med = med * (1 + LUTEAL_OFFSETS["hrv_frac"])
        med += offset

        z = (cur_val - med) / scale
        z_risk = -z if spec["dir"] == "low" else z
        strain = _strain_from_z(z_risk)

        if key == "spo2" and cur_val < 95:        # absolute soft threshold
            strain = max(strain, min(100.0, (95 - cur_val) / 5 * 100))

        feature_results[key] = {
            "present": True, "value": cur_val, "date": cur_date.isoformat(),
            "baseline": med, "z": z, "strain": strain, "baseline_mode": baseline_mode,
            "n_baseline": len(values),
        }

    # Motion component 2: near-falls / cancelled fall alerts in last 14 days
    now_dt = datetime.combine(today, datetime.max.time()).replace(tzinfo=timezone.utc)
    near_falls_14 = 0
    falls_365 = 0
    for e in events:
        edt = _to_dt(e.get("detected_at"))
        if not edt:
            continue
        age_days = (now_dt - edt).total_seconds() / 86400
        if age_days < 0:
            continue
        etype = str(e.get("event_type") or "fall")
        cancelled = bool(e.get("user_cancelled"))
        if etype == "fall" and not cancelled and age_days <= 365:
            falls_365 += 1
        elif (etype == "near_fall" or cancelled) and age_days <= 14:
            near_falls_14 += 1
    near_fall_strain = min(100.0, 50.0 * near_falls_14)
    fh_norm = 0.0 if falls_365 == 0 else (60.0 if falls_365 == 1 else 100.0)

    # Layer aggregation
    layers: dict[str, dict] = {}
    for L in ("P", "RC", "MO"):
        feats = [k for k, s in FEATURES.items() if s["layer"] == L]
        total_w = sum(FEATURES[k]["w"] for k in feats) + (NEAR_FALL_WEIGHT if L == "MO" else 0)
        present = [k for k in feats if feature_results[k]["present"]]
        pw = sum(FEATURES[k]["w"] for k in present)
        strain_sum = sum(FEATURES[k]["w"] * feature_results[k]["strain"] for k in present)
        if L == "MO":
            pw += NEAR_FALL_WEIGHT
            strain_sum += NEAR_FALL_WEIGHT * near_fall_strain
        coverage = pw / total_w if total_w else 0.0
        risk = strain_sum / pw if pw else 0.0
        layers[L] = {"risk": risk, "stability": 100.0 - risk, "coverage": coverage, "present_weight": pw}
    layers["FH"] = {"risk": fh_norm, "stability": 100.0 - fh_norm, "coverage": 1.0, "present_weight": 1.0}

    # Adaptive weights: base evidence weight x coverage, renormalised to sum 1
    raw = {L: BASE_LAYER_WEIGHTS[L] * layers[L]["coverage"] for L in layers}
    s = sum(raw.values())
    weights = {L: (raw[L] / s if s else 0.0) for L in layers}

    frs_raw = sum(weights[L] * layers[L]["risk"] for L in layers)

    # Additive attribution: contributions sum exactly to frs_raw
    contributions = []
    for key, spec in FEATURES.items():
        fr = feature_results[key]
        if not fr["present"]:
            continue
        L = spec["layer"]
        pts = weights[L] * (spec["w"] / layers[L]["present_weight"]) * fr["strain"]
        contributions.append({
            "feature": key, "label": spec["label"], "layer": L, "points": pts,
            "value": fr["value"], "baseline": fr["baseline"], "z": fr["z"],
            "unit": spec["unit"], "ndp": spec["ndp"], "baseline_mode": fr["baseline_mode"],
        })
    if near_falls_14:
        contributions.append({"feature": "near_falls", "label": "near-fall events (14 days)", "layer": "MO",
                              "points": weights["MO"] * (NEAR_FALL_WEIGHT / layers["MO"]["present_weight"]) * near_fall_strain,
                              "value": near_falls_14, "unit": "events"})
    if falls_365:
        contributions.append({"feature": "fall_history", "label": "falls in past 365 days", "layer": "FH",
                              "points": weights["FH"] * fh_norm, "value": falls_365, "unit": "falls"})
    contributions.sort(key=lambda c: c["points"], reverse=True)

    # Coverage / confidence Q
    q = sum(BASE_LAYER_WEIGHTS[L] * layers[L]["coverage"] for L in layers) * 100
    latest = max(latest_dates) if latest_dates else None
    data_age = (today - latest).days if latest else None
    stale = data_age is None or data_age > STALE_AFTER_DAYS
    if data_age is None:
        q = 0.0
    elif data_age > 1:
        q -= min(40.0, 10.0 * (data_age - 1))
    q = max(0.0, min(100.0, q))

    frs = int(round(frs_raw))
    result = {
        "date": today.isoformat(),
        "frs": frs,
        "frs_raw": frs_raw,
        "tier": tier_for(frs),
        "coverage_q": int(round(q)),
        "low_confidence": q < 60,
        "stale": stale,
        "latest_data_date": latest.isoformat() if latest else None,
        "data_age_days": data_age,
        "layers": layers,
        "weights": weights,
        "features": feature_results,
        "contributions": contributions,
        "falls_365": falls_365,
        "near_falls_14": near_falls_14,
        "cycle": None,
    }

    if today_phase:
        cyc = {"phase": today_phase["phase"], "cycle_day": today_phase["cycle_day"],
               "cycle_length": today_phase["cycle_length"], "applied": use_cycle}
        if use_cycle and _include_uncorrected:
            unc = compute_fall_risk(history, today, apply_cycle_correction=False, _include_uncorrected=False)
            cyc["frs_without_correction"] = unc["frs"]
            cyc["physiology_risk_without_correction"] = unc["layers"]["P"]["risk"]
        result["cycle"] = cyc
    return result


def compute_fall_risk_series(history: dict, end: date, days: int = 30,
                             apply_cycle_correction: bool = True) -> list[dict]:
    """Daily FRS for the `days` days ending at `end` (uses only data up to each day)."""
    out = []
    for i in range(days - 1, -1, -1):
        d = end - timedelta(days=i)
        r = compute_fall_risk(history, d, apply_cycle_correction, _include_uncorrected=False)
        out.append({"date": d.isoformat(), "frs": r["frs"], "physiology_risk": r["layers"]["P"]["risk"],
                    "tier": r["tier"], "coverage_q": r["coverage_q"],
                    "phase": (r["cycle"] or {}).get("phase")})
    return out


# ── LLM context formatting (grounding-friendly: every number is literal) ────

def format_fall_risk_context(result: dict, trend_7d: Optional[int] = None,
                             correlations: Optional[list[dict]] = None) -> str:
    lines = ["FALL RISK ASSESSMENT (computed by the deterministic MedXAI fall-risk engine; do NOT recompute or adjust these values):"]
    lines.append(f"- FALL RISK SCORE (24h): {result['frs']}/100, tier: {result['tier']}")
    conf = "LOW CONFIDENCE" if result["low_confidence"] else "adequate"
    lines.append(f"- DATA COVERAGE (confidence): {result['coverage_q']}% ({conf})")
    if result["latest_data_date"]:
        stale_note = f" [STALE: latest ring data is {result['data_age_days']} days old, NOT current]" if result["stale"] else ""
        lines.append(f"- LATEST RING DATA USED: {result['latest_data_date']}{stale_note}")
    else:
        lines.append("- LATEST RING DATA USED: none found [NO DATA]")

    L = result["layers"]
    lines.append(
        "- LAYER RISK (0-100, higher = more risk): "
        f"Physiology {int(round(L['P']['risk']))}, Recovery {int(round(L['RC']['risk']))}, "
        f"Motion {int(round(L['MO']['risk']))}, Fall history {int(round(L['FH']['risk']))}"
    )
    w = result["weights"]
    lines.append(
        "- LAYER WEIGHTS (evidence-derived, coverage-adjusted): "
        f"Fall history {w['FH']:.2f}, Motion {w['MO']:.2f}, Physiology {w['P']:.2f}, Recovery {w['RC']:.2f}"
    )
    lines.append(f"- FALLS IN PAST 365 DAYS: {result['falls_365']}; NEAR-FALL EVENTS IN PAST 14 DAYS: {result['near_falls_14']}")

    top = [c for c in result["contributions"] if c["points"] >= 0.5][:4]
    if top:
        lines.append("- TOP CONTRIBUTORS (points added to the score):")
        for c in top:
            pts = int(round(c["points"]))
            if c["feature"] in ("near_falls", "fall_history"):
                lines.append(f"  - {c['label']}: {c['value']}, +{pts} points")
            else:
                ndp = c["ndp"]
                unit = f" {c['unit']}" if c["unit"] else ""
                lines.append(
                    f"  - {c['label']}: {_fmt(c['value'], ndp)}{unit} vs personal baseline "
                    f"{_fmt(c['baseline'], ndp)}{unit} (z = {c['z']:.1f}), +{pts} points"
                )
    else:
        lines.append("- TOP CONTRIBUTORS: none; all metrics are within the personal baseline range")

    cyc = result.get("cycle")
    if cyc:
        if cyc.get("applied"):
            extra = ""
            if "frs_without_correction" in cyc:
                extra = f"; score without cycle adjustment would be {cyc['frs_without_correction']}"
            lines.append(
                f"- CYCLE PHASE: {cyc['phase']} (day {cyc['cycle_day']} of {cyc['cycle_length']}); "
                f"cycle-aware baseline applied to HRV, resting heart rate and temperature{extra}"
            )
        else:
            lines.append(f"- CYCLE PHASE: {cyc['phase']} (day {cyc['cycle_day']} of {cyc['cycle_length']}); cycle adjustment not applied")

    if trend_7d is not None:
        lines.append(f"- 7-DAY AVERAGE FALL RISK SCORE: {trend_7d}")

    if correlations:
        lines.append("- PERSONAL CORRELATIONS (from this user's own history):")
        for c in correlations[:3]:
            lines.append(f"  - {c['sentence']} (r = {c['r']:.2f}, n = {c['n']} days)")
    return "\n".join(lines)


def build_card(result: dict, trend_7d: Optional[int] = None) -> dict:
    """Payload for the Flutter chat card (type 'fall_risk')."""
    return {
        "type": "fall_risk",
        "data": {
            "score": result["frs"],
            "tier": result["tier"],
            "coverage": result["coverage_q"],
            "stale": result["stale"],
            "latest_data_date": result["latest_data_date"],
            "trend_7d": trend_7d,
            "layers": {LAYER_NAMES[k]: int(round(v["risk"])) for k, v in result["layers"].items()},
            "top_contributors": [
                {"label": c["label"], "points": int(round(c["points"]))}
                for c in result["contributions"][:3] if c["points"] >= 0.5
            ],
            "cycle_phase": (result.get("cycle") or {}).get("phase"),
        },
    }


# ── Supabase fetch (lazy import so the pure engine works offline) ───────────

def _daily_mean(rows: list[dict], ts_col: str, val_col: str) -> dict[date, float]:
    buckets: dict[date, list[float]] = {}
    for r in rows:
        d = _to_date(r.get(ts_col))
        v = r.get(val_col)
        if d and v is not None:
            buckets.setdefault(d, []).append(float(v))
    return {d: sum(v) / len(v) for d, v in buckets.items()}


def fetch_fall_history(user_id: str, today: date, days: int = 45) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    from db.supabase import supabase

    since = (today - timedelta(days=days)).isoformat()
    since_365 = (datetime.now(timezone.utc) - timedelta(days=366)).isoformat()

    def q_date(table: str):
        try:
            return supabase.table(table).select("*").eq("user_id", user_id).gte("date", since).execute().data or []
        except Exception:
            return []

    def q_ts(table: str):
        try:
            return supabase.table(table).select("*").eq("user_id", user_id).gte("measured_at", since).execute().data or []
        except Exception:
            return []

    def q_profile():
        try:
            r = supabase.table("user_profiles").select("*").eq("id", user_id).maybe_single().execute()
            return (r.data if r else None) or {}
        except Exception:
            return {}

    def q_cycles():
        try:
            return supabase.table("user_cycles").select("*").eq("user_id", user_id).order("period_start", desc=True).limit(6).execute().data or []
        except Exception:
            return []

    def q_events():
        try:
            return supabase.table("user_fall_events").select("*").eq("user_id", user_id).gte("detected_at", since_365).execute().data or []
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {
            "hr": ex.submit(q_date, "user_hr"), "hrv": ex.submit(q_date, "user_hrv"),
            "spo2": ex.submit(q_date, "user_spo2"), "sleep": ex.submit(q_date, "user_sleep"),
            "steps": ex.submit(q_date, "user_steps"), "temp": ex.submit(q_ts, "user_temp"),
            "stress": ex.submit(q_ts, "user_stress"), "profile": ex.submit(q_profile),
            "cycles": ex.submit(q_cycles), "events": ex.submit(q_events),
        }
        res = {k: f.result() for k, f in futs.items()}

    return assemble_history(res)


def assemble_history(res: dict) -> dict:
    """
    Convert raw Supabase-shaped rows into the engine's history format.
    res keys: hr, hrv, spo2, sleep, steps, temp, stress (row lists),
              profile (dict), cycles (list), events (list).
    Shared by the live fetch and the offline demo generator so both paths
    produce identical engine inputs.
    """
    daily: dict[date, dict] = {}

    def put(d, k, v):
        if d is not None and v is not None:
            daily.setdefault(d, {})[k] = float(v)

    for r in res.get("hr", []):
        d = _to_date(r.get("date"))
        # min daily HR is the closest available proxy for resting HR
        put(d, "rhr", r.get("min_hr") if r.get("min_hr") is not None else r.get("avg_hr"))
    for r in res.get("hrv", []):
        put(_to_date(r.get("date")), "hrv", r.get("avg_hrv"))
    for r in res.get("spo2", []):
        put(_to_date(r.get("date")), "spo2", r.get("avg_spo2"))
    for r in res.get("sleep", []):
        d = _to_date(r.get("date"))
        tot = r.get("total_duration")
        if tot is not None and tot > 1440:     # stored in seconds in some rows
            tot = tot / 60
        put(d, "sleep_min", tot)
        put(d, "sleep_score", r.get("sleep_score"))
    for r in res.get("steps", []):
        put(_to_date(r.get("date")), "steps", r.get("steps"))
    for d, v in _daily_mean(res.get("temp", []), "measured_at", "value_c").items():
        put(d, "temp", v)
    for d, v in _daily_mean(res.get("stress", []), "measured_at", "stress_value").items():
        put(d, "stress", v)

    return {
        "profile": res.get("profile") or {},
        "daily": {d.isoformat(): v for d, v in daily.items()},
        "cycles": res.get("cycles") or [],
        "events": res.get("events") or [],
    }


_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_TTL_S = 60.0


def get_fall_risk(user_id: str, today: Optional[date] = None, use_cache: bool = True) -> dict:
    """
    Fetch history, compute today's score, 7-day trend and correlations.
    Returns {"result", "trend_7d", "correlations", "context", "card"}.
    """
    today = today or datetime.utcnow().date()
    key = f"{user_id}:{today.isoformat()}"
    if use_cache and key in _CACHE and time.monotonic() - _CACHE[key][0] < _CACHE_TTL_S:
        return _CACHE[key][1]

    history = fetch_fall_history(user_id, today)
    result = compute_fall_risk(history, today)
    series = compute_fall_risk_series(history, today - timedelta(days=1), days=7)
    trend_7d = int(round(sum(s["frs"] for s in series) / len(series))) if series else None

    try:
        from agent.correlations import compute_correlations
        correlations = compute_correlations(history.get("daily") or {})
    except Exception as e:
        print(f"[FALL RISK] correlation step skipped: {e}")
        correlations = []

    out = {
        "result": result,
        "trend_7d": trend_7d,
        "correlations": correlations,
        "context": format_fall_risk_context(result, trend_7d, correlations),
        "card": build_card(result, trend_7d),
    }
    _CACHE[key] = (time.monotonic(), out)
    _log_result(user_id, result)
    return out


def _log_result(user_id: str, result: dict) -> None:
    """Best-effort audit log to user_fall_risk (never breaks the chat path)."""
    try:
        from db.supabase import supabase
        supabase.table("user_fall_risk").insert({
            "user_id": user_id,
            "frs_24": result["frs"],
            "tier": result["tier"],
            "coverage_q": result["coverage_q"],
            "layer_p": int(round(result["layers"]["P"]["risk"])),
            "layer_rc": int(round(result["layers"]["RC"]["risk"])),
            "layer_mo": int(round(result["layers"]["MO"]["risk"])),
            "layer_fh": int(round(result["layers"]["FH"]["risk"])),
            "weights": {k: round(v, 3) for k, v in result["weights"].items()},
            "top_contributors": [{"label": c["label"], "points": round(c["points"], 1)} for c in result["contributions"][:5]],
            "cycle_phase": (result.get("cycle") or {}).get("phase"),
        }).execute()
    except Exception as e:
        print(f"[FALL RISK] audit log skipped: {e}")


def get_recent_fall_event(user_id: str, window_minutes: int = 30) -> Optional[dict]:
    """Biometric safety signal: an uncancelled fall detected in the last N minutes."""
    try:
        from db.supabase import supabase
        since = (datetime.now(timezone.utc) - timedelta(minutes=window_minutes)).isoformat()
        r = supabase.table("user_fall_events").select("*").eq("user_id", user_id)\
            .eq("event_type", "fall").eq("user_cancelled", False)\
            .gte("detected_at", since).order("detected_at", desc=True).limit(1).execute()
        return (r.data or [None])[0]
    except Exception as e:
        print(f"[FALL RISK] recent fall lookup failed: {e}")
        return None
