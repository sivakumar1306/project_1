"""
Synthetic demo cohort for the fall-risk evaluation.

Five users, 35 days each, generated deterministically (fixed seeds) so the
seed script, the offline engine analysis and the agent evaluation all see the
same data. Rows are shaped exactly like the Supabase tables in schema.sql /
schema_fall.sql.

Built-in physiology couplings (so correlation detection has real signal):
  - shorter sleep -> lower HRV and lower sleep score
  - higher stress -> higher resting HR, worse next-night sleep score
  - luteal phase  -> skin temp +0.35 °C, resting HR +3 bpm, HRV -10 %
"""

from __future__ import annotations

import random
from datetime import date, datetime, time, timedelta, timezone

DAYS = 35

DEMO_USERS = [
    {"id": "11111111-1111-4111-8111-000000000001", "full_name": "Arun (demo: low risk)",
     "age": 29, "gender": "Male", "height_cm": 176, "weight_kg": 72, "scenario": "stable", "seed": 11},
    {"id": "11111111-1111-4111-8111-000000000002", "full_name": "Meera (demo: luteal + poor recovery)",
     "age": 31, "gender": "Female", "height_cm": 162, "weight_kg": 58, "scenario": "luteal_strain", "seed": 22},
    {"id": "11111111-1111-4111-8111-000000000003", "full_name": "Ravi (demo: stale data)",
     "age": 67, "gender": "Male", "height_cm": 170, "weight_kg": 74, "scenario": "stale", "seed": 33},
    {"id": "11111111-1111-4111-8111-000000000004", "full_name": "Lakshmi (demo: recent fall)",
     "age": 72, "gender": "Female", "height_cm": 155, "weight_kg": 60, "scenario": "recent_fall", "seed": 44},
    {"id": "11111111-1111-4111-8111-000000000005", "full_name": "Priya (demo: healthy luteal)",
     "age": 27, "gender": "Female", "height_cm": 160, "weight_kg": 55, "scenario": "healthy_luteal", "seed": 55},
]
USERS_BY_KEY = {
    "low_risk": DEMO_USERS[0]["id"], "luteal_strain": DEMO_USERS[1]["id"], "stale": DEMO_USERS[2]["id"],
    "recent_fall": DEMO_USERS[3]["id"], "healthy_luteal": DEMO_USERS[4]["id"],
}

BASE = {
    "stable":         dict(sleep=450, stress=30, hrv=58, rhr=58, spo2=97.5, temp=36.4, steps=8500),
    "luteal_strain":  dict(sleep=440, stress=35, hrv=52, rhr=60, spo2=97.5, temp=36.3, steps=7800),
    "stale":          dict(sleep=400, stress=38, hrv=34, rhr=64, spo2=96.5, temp=36.2, steps=5200),
    "recent_fall":    dict(sleep=390, stress=40, hrv=28, rhr=66, spo2=96.0, temp=36.1, steps=3800),
    "healthy_luteal": dict(sleep=460, stress=28, hrv=62, rhr=57, spo2=98.0, temp=36.3, steps=9000),
}

CYCLE_DAY_TODAY = {"luteal_strain": 23, "healthy_luteal": 24}   # both luteal today
CYCLE_LEN, PERIOD_LEN = 28, 5


def _phase(cycle_day: int) -> str:
    ov = CYCLE_LEN - 14
    if cycle_day <= PERIOD_LEN:
        return "menstrual"
    if cycle_day < ov - 1:
        return "follicular"
    if cycle_day <= ov + 1:
        return "ovulatory"
    return "luteal"


def generate_user(user: dict, today: date, now: datetime | None = None, include_recent_fall: bool = True) -> dict:
    """Return Supabase-shaped rows for one demo user."""
    now = now or datetime.now(timezone.utc)
    sc = user["scenario"]
    rng = random.Random(user["seed"])
    b = BASE[sc]
    uid = user["id"]

    rows = {k: [] for k in ("hr", "hrv", "spo2", "sleep", "steps", "temp", "stress", "hr_readings", "cycles", "events")}
    last_day_offset = 4 if sc == "stale" else 0          # stale user stopped syncing 4 days ago

    prev_stress = b["stress"]
    for back in range(DAYS - 1, last_day_offset - 1, -1):
        d = today - timedelta(days=back)
        acute = back <= 2   # last 3 days

        sleep = b["sleep"] + rng.gauss(0, 35)
        stress = b["stress"] + rng.gauss(0, 7)
        steps = b["steps"] + rng.gauss(0, 1100)
        spo2 = b["spo2"] + rng.gauss(0, 0.5)
        temp = b["temp"] + rng.gauss(0, 0.07)

        if sc == "luteal_strain" and acute:
            sleep -= 130
            stress += 32
        if sc == "recent_fall" and back <= 5:
            steps *= 0.55
            sleep -= 60
        if sc == "recent_fall" and back <= 1:
            spo2 -= 3.0

        hrv = b["hrv"] + 0.09 * (sleep - b["sleep"]) + rng.gauss(0, 2.5)
        rhr = b["rhr"] + 0.12 * (stress - b["stress"]) + rng.gauss(0, 1.0)
        score = 78 + 0.18 * (sleep - b["sleep"]) - 0.25 * (prev_stress - b["stress"]) + rng.gauss(0, 3)

        if sc in CYCLE_DAY_TODAY:
            cday = ((CYCLE_DAY_TODAY[sc] - 1 - back) % CYCLE_LEN) + 1
            if _phase(cday) == "luteal":
                temp += 0.35
                rhr += 3.0
                hrv *= 0.90
        if sc == "luteal_strain" and acute:
            hrv *= 0.85

        sleep = max(180, min(600, sleep))
        score = int(max(35, min(98, round(score))))
        stress_i = int(max(5, min(95, round(stress))))
        prev_stress = stress_i
        ds = d.isoformat()

        rows["sleep"].append({"user_id": uid, "date": ds, "total_duration": int(round(sleep)), "sleep_score": score})
        rows["hr"].append({"user_id": uid, "date": ds, "avg_hr": round(rhr + 14), "min_hr": round(rhr), "max_hr": round(rhr + 58)})
        rows["hrv"].append({"user_id": uid, "date": ds, "avg_hrv": round(hrv), "min_hrv": round(hrv * 0.6), "max_hrv": round(hrv * 1.5)})
        rows["spo2"].append({"user_id": uid, "date": ds, "avg_spo2": int(round(spo2)), "min_spo2": int(round(spo2)) - 2, "max_spo2": min(100, int(round(spo2)) + 1)})
        rows["steps"].append({"user_id": uid, "date": ds, "steps": int(max(300, steps)), "calories": int(max(300, steps) * 0.04)})
        rows["temp"].append({"user_id": uid, "measured_at": datetime.combine(d, time(2, 30), timezone.utc).isoformat(), "value_c": round(temp, 2)})
        rows["stress"].append({"user_id": uid, "measured_at": datetime.combine(d, time(9, 0), timezone.utc).isoformat(),
                               "stress_value": stress_i, "label": "high" if stress_i >= 60 else ("medium" if stress_i >= 40 else "low")})

    last_reading_at = now - timedelta(days=4) if sc == "stale" else now - timedelta(minutes=20)
    rows["hr_readings"].append({"user_id": uid, "measured_at": last_reading_at.isoformat(),
                                "value_bpm": int(b["rhr"] + 12), "source": "smart_ring"})

    if sc in CYCLE_DAY_TODAY:
        start = today - timedelta(days=CYCLE_DAY_TODAY[sc] - 1)
        for k in range(3):
            ps = start - timedelta(days=CYCLE_LEN * k)
            rows["cycles"].append({"user_id": uid, "period_start": ps.isoformat(),
                                   "period_end": (ps + timedelta(days=PERIOD_LEN - 1)).isoformat(),
                                   "cycle_length": CYCLE_LEN, "period_length": PERIOD_LEN})

    def ev(days_ago: float, etype: str, peak_g: float, cancelled: bool = False):
        rows["events"].append({"user_id": uid, "detected_at": (now - timedelta(days=days_ago)).isoformat(),
                               "event_type": etype, "peak_g": peak_g, "user_cancelled": cancelled,
                               "dispatched": False, "source": "demo_seed"})

    if sc == "luteal_strain":
        ev(150, "fall", 3.1)
    if sc == "recent_fall":
        ev(200, "fall", 3.4)
        ev(40, "fall", 2.9)
        ev(3, "near_fall", 2.2)
        ev(9, "near_fall", 2.0)
        if include_recent_fall:
            rows["events"].append(recent_fall_event(uid, now))
    return rows


def recent_fall_event(uid: str, now: datetime | None = None, minutes_ago: int = 5) -> dict:
    now = now or datetime.now(timezone.utc)
    return {"user_id": uid, "detected_at": (now - timedelta(minutes=minutes_ago)).isoformat(),
            "event_type": "fall", "peak_g": 3.6, "user_cancelled": False, "dispatched": False, "source": "demo_seed"}


def profile_row(user: dict) -> dict:
    return {"id": user["id"], "user_id": user["id"], "full_name": user["full_name"], "age": user["age"],
            "gender": user["gender"], "height_cm": user["height_cm"], "weight_kg": user["weight_kg"]}


def history_for(user: dict, today: date, now: datetime | None = None, include_recent_fall: bool = True) -> dict:
    """Engine-format history for one demo user (offline, no Supabase)."""
    from agent.fall_risk import assemble_history
    rows = generate_user(user, today, now, include_recent_fall)
    res = {k: rows[k] for k in ("hr", "hrv", "spo2", "sleep", "steps", "temp", "stress", "cycles", "events")}
    res["profile"] = profile_row(user)
    return assemble_history(res)
