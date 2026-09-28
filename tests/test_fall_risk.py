"""
Unit tests for the Review-II deterministic components (no LLM, no network).
Run:  python -m pytest tests/test_fall_risk.py -q
"""
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.fall_risk import (BASE_LAYER_WEIGHTS, compute_fall_risk, compute_fall_risk_series,
                             cycle_phase, format_fall_risk_context, tier_for)
from agent.correlations import compute_correlations
from scripts.fall_demo_data import DEMO_USERS, USERS_BY_KEY, history_for

TODAY = date(2026, 9, 30)
NOW = datetime(2026, 9, 30, 10, 0, tzinfo=timezone.utc)
USERS = {u["scenario"]: u for u in DEMO_USERS}


def _hist(scenario, include_recent_fall=True):
    return history_for(USERS[scenario], TODAY, NOW, include_recent_fall)


# ── weights ────────────────────────────────────────────────────────────────
def test_evidence_weights_sum_to_one_and_follow_evidence_order():
    assert math.isclose(sum(BASE_LAYER_WEIGHTS.values()), 1.0, abs_tol=1e-9)
    w = BASE_LAYER_WEIGHTS
    assert w["FH"] > w["MO"] > w["P"] > w["RC"]
    assert round(w["FH"], 2) == 0.40 and round(w["RC"], 2) == 0.09


def test_adaptive_weights_renormalise_when_data_missing():
    h = _hist("stable")
    for d in h["daily"].values():
        d.pop("steps", None)
    r = compute_fall_risk(h, TODAY)
    assert math.isclose(sum(r["weights"].values()), 1.0, abs_tol=1e-9)
    assert r["layers"]["MO"]["coverage"] == 0.5
    assert r["weights"]["MO"] < BASE_LAYER_WEIGHTS["MO"]
    assert r["coverage_q"] < 100


# ── scoring ────────────────────────────────────────────────────────────────
def test_contributions_sum_to_score():
    for sc in USERS:
        r = compute_fall_risk(_hist(sc), TODAY)
        total = sum(c["points"] for c in r["contributions"])
        assert math.isclose(total, r["frs_raw"], abs_tol=1e-6), sc


def test_scenarios_rank_as_designed():
    frs = {sc: compute_fall_risk(_hist(sc), TODAY)["frs"] for sc in USERS}
    assert tier_for(frs["stable"]) == "LOW"
    assert tier_for(frs["recent_fall"]) == "HIGH"
    assert frs["recent_fall"] > frs["luteal_strain"] > frs["stable"]


def test_stale_data_is_flagged_and_lowers_confidence():
    r = compute_fall_risk(_hist("stale"), TODAY)
    assert r["stale"] and r["data_age_days"] == 4
    assert r["coverage_q"] < 100
    assert "[STALE" in format_fall_risk_context(r)


def test_no_data_gives_zero_score_and_zero_confidence():
    r = compute_fall_risk({"profile": {}, "daily": {}, "cycles": [], "events": []}, TODAY)
    assert r["frs"] == 0 and r["coverage_q"] == 0 and r["low_confidence"]


def test_fall_history_counts_only_uncancelled_falls():
    h = {"profile": {}, "daily": {}, "cycles": [], "events": [
        {"detected_at": (NOW - timedelta(days=10)).isoformat(), "event_type": "fall", "user_cancelled": False},
        {"detected_at": (NOW - timedelta(days=5)).isoformat(), "event_type": "fall", "user_cancelled": True},
        {"detected_at": (NOW - timedelta(days=400)).isoformat(), "event_type": "fall", "user_cancelled": False},
    ]}
    r = compute_fall_risk(h, TODAY)
    assert r["falls_365"] == 1          # old fall excluded, cancelled one is a near-fall
    assert r["near_falls_14"] == 1


# ── cycle awareness ────────────────────────────────────────────────────────
def test_cycle_phase_boundaries():
    cycles = [{"period_start": "2026-09-01", "cycle_length": 28, "period_length": 5}]
    assert cycle_phase(date(2026, 9, 3), cycles)["phase"] == "menstrual"
    assert cycle_phase(date(2026, 9, 8), cycles)["phase"] == "follicular"
    assert cycle_phase(date(2026, 9, 14), cycles)["phase"] == "ovulatory"
    assert cycle_phase(date(2026, 9, 23), cycles)["phase"] == "luteal"
    assert cycle_phase(date(2026, 8, 25), cycles)["phase"] == "luteal"   # projected backwards


def test_cycle_correction_only_for_female_users_with_logs():
    r = compute_fall_risk(_hist("stable"), TODAY)
    assert r["cycle"] is None
    r2 = compute_fall_risk(_hist("healthy_luteal"), TODAY)
    assert r2["cycle"]["applied"] and r2["cycle"]["phase"] == "luteal"


def test_cycle_correction_reduces_luteal_physiology_strain_for_healthy_user():
    h = _hist("healthy_luteal")
    on = compute_fall_risk_series(h, TODAY, 21, True)
    off = compute_fall_risk_series(h, TODAY, 21, False)
    lut_on = [d["physiology_risk"] for d in on if d["phase"] == "luteal"]
    lut_off = [d["physiology_risk"] for d, x in zip(off, on) if x["phase"] == "luteal"]
    assert lut_on and sum(lut_on) / len(lut_on) < sum(lut_off) / len(lut_off)


# ── correlations ───────────────────────────────────────────────────────────
def test_correlation_detects_built_in_sleep_hrv_coupling():
    cors = compute_correlations(_hist("stable")["daily"])
    pairs = {(c["x"], c["y"]) for c in cors}
    assert ("sleep_min", "hrv") in pairs
    c = [c for c in cors if (c["x"], c["y"]) == ("sleep_min", "hrv")][0]
    assert c["r"] > 0.4 and c["p"] < 0.05 and c["n"] >= 14


def test_correlation_needs_enough_days():
    daily = {f"2026-09-{d:02d}": {"sleep_min": 400 + d, "hrv": 50 + d} for d in range(1, 8)}
    assert compute_correlations(daily) == []


# ── safety vocabulary & fusion ─────────────────────────────────────────────
def test_fall_keywords_guard_benign_phrases():
    from agent.tools import match_emergency_keywords
    assert match_emergency_keywords("I fell and can't get up")
    assert match_emergency_keywords("i just fell in the bathroom")
    assert match_emergency_keywords("I'm lying on the floor")
    assert not match_emergency_keywords("I fell asleep on the couch")
    assert not match_emergency_keywords("my fellow student")
    assert not match_emergency_keywords("why do older people fall more often?")


def test_biometric_signal_alone_triggers_fusion():
    from agent.tools import check_emergency_fused
    ev = {"detected_at": NOW.isoformat(), "peak_g": 3.6}
    is_e, msg, meta = check_emergency_fused("I'm fine, just a bit shaken", (False, "calm message", 0.9), ev)
    assert is_e and meta["biometric_triggered"] and "EMERGENCY DETECTED" in msg
    is_e2, _, meta2 = check_emergency_fused("I'm fine, just a bit shaken", (False, "calm message", 0.9), None)
    assert not is_e2 and not meta2["biometric_triggered"]


def test_llm_override_still_suppresses_informational_keyword():
    from agent.tools import check_emergency_fused
    is_e, _, meta = check_emergency_fused("What are the symptoms of a stroke?", (False, "informational", 0.99))
    assert not is_e and meta["keyword_overridden"]


# ── routing rules ──────────────────────────────────────────────────────────
def test_safety_widening_rule():
    from agent.router import apply_safety_widening, SAFETY_STREAMS
    s, w = apply_safety_widening(["sleep"], 0.5, True)
    assert w and all(x in s for x in SAFETY_STREAMS)
    s, w = apply_safety_widening(["hrv"], 0.9, True)
    assert w and "fall_risk" in s and "spo2" not in s
    s, w = apply_safety_widening(["bp"], 0.3, False)
    assert not w and s == ["bp"]


def test_router_output_parsing_accepts_both_formats():
    from agent.router import _parse_router_output
    assert _parse_router_output('["bp", "nonsense"]')[0] == ["bp"]
    st, conf, sr = _parse_router_output('{"streams": ["fall_risk"], "confidence": 0.6, "safety_relevant": true}')
    assert st == ["fall_risk"] and conf == 0.6 and sr is True


def test_keyword_safety_relevance_requires_personal_context():
    from agent.router import keyword_safety_relevant
    assert keyword_safety_relevant("I feel dizzy when I stand up")
    assert not keyword_safety_relevant("why do older people fall more often?")


# ── strict grounding ───────────────────────────────────────────────────────
def test_strict_grounding_uses_digit_boundaries_and_ignores_facts():
    from agent.graph import compute_grounding_score_strict
    data = "FALL RISK SCORE (24h): 56/100\n- HRV: 26 ms"
    assert compute_grounding_score_strict("Your score is 56 and HRV 26 ms.", data)["grounding_score"] == 1.0
    r = compute_grounding_score_strict("Your score is 5 out of 10.", data)
    assert r["ungrounded_numbers"] == ["5", "10"]
    assert compute_grounding_score_strict("Call 112 now.", data)["total_numbers_checked"] == 0


# ── gait signal processing (Motion layer) ──────────────────────────────────
def _synthetic_walk(step_s=0.55, cv=0.02, seed=0, fs=100, dur=70):
    import numpy as np
    rng = np.random.default_rng(seed)
    t, cur = [], 0.0
    while cur < dur:
        cur += step_s * (1 + rng.normal(0, cv))
        t.append(cur)
    tt = np.arange(int(dur * fs)) / fs
    v = np.ones_like(tt)
    for ti in t:
        v += 0.3 * np.exp(-((tt - ti) / 0.06) ** 2)
    v += rng.normal(0, 0.02, len(tt))
    ml = 0.1 * np.sin(2 * np.pi * tt / (2 * step_s)) + rng.normal(0, 0.03, len(tt))
    ap = 0.15 * np.sin(2 * np.pi * tt / step_s) + rng.normal(0, 0.03, len(tt))
    return v, ml, ap, fs


def test_gait_features_recover_known_cadence_and_variability():
    from agent.gait_features import extract_gait_features
    steady = extract_gait_features(*_synthetic_walk(0.55, 0.02, seed=1))
    irregular = extract_gait_features(*_synthetic_walk(0.55, 0.08, seed=2))
    assert abs(steady["cadence"] - 60 / 0.55) < 3
    assert irregular["step_time_cv"] > steady["step_time_cv"]
    assert irregular["stride_regularity"] < steady["stride_regularity"]


def test_gait_instability_index_orders_by_risk():
    from agent.gait_features import extract_gait_features, gait_instability_index
    rows = [extract_gait_features(*_synthetic_walk(0.55, 0.02, seed=s)) for s in range(5)] + \
           [extract_gait_features(*_synthetic_walk(0.65, 0.08, seed=10 + s)) for s in range(5)]
    idx = gait_instability_index(rows)
    assert idx[5:].mean() > idx[:5].mean()


def test_v2_detector_recovers_true_variability_despite_turns():
    import numpy as np
    from scripts.gait_synthetic_bench import run_bench
    r = run_bench(n=12)
    err_v1 = np.abs(r[:, 1] - r[:, 0]).mean()
    err_v2 = np.abs(r[:, 2] - r[:, 0]).mean()
    assert err_v2 < 1.0 and err_v2 < err_v1 / 3
    assert np.corrcoef(r[:, 0], r[:, 2])[0, 1] > 0.9


def test_daily_walking_detector_finds_walks_and_ignores_rest():
    import numpy as np
    from scripts.gait_synthetic_bench import simulate_walk
    from agent.daily_gait import walking_windows, detect_bouts, daily_features_for_chunks
    rng = np.random.default_rng(0)
    fs = 100
    rest = lambda s: (np.ones(int(s * fs)) + rng.normal(0, 0.01, int(s * fs)))
    wv, wml, wap, wyaw, _, _ = simulate_walk(0.02, seed=4, dur=60, turns=False)
    v = np.concatenate([rest(120), wv, rest(120)])
    ml = np.concatenate([rng.normal(0, .01, 12000), wml, rng.normal(0, .01, 12000)])
    ap = np.concatenate([rng.normal(0, .01, 12000), wap, rng.normal(0, .01, 12000)])
    walk = walking_windows(v, fs)
    bouts = detect_bouts(walk, fs)
    assert len(bouts) == 1
    a, b = bouts[0]
    assert 115 * fs <= a <= 130 * fs and 170 * fs <= b <= 185 * fs      # found the 60 s walk
    assert walk[:20].sum() == 0 and walk[-20:].sum() == 0               # rest is not walking
    f = daily_features_for_chunks([{"v": v, "ml": ml, "ap": ap, "yaw": None}] * 3, fs)
    assert f["d_n_bouts"] == 3 and abs(f["d_cadence"] - 60 / (wv.size and 0.55)) < 25


def test_fall_keywords_handle_contractions_and_typographic_apostrophes():
    from agent.tools import match_emergency_keywords
    assert match_emergency_keywords("I can’t get up")                  # curly apostrophe from phone keyboards
    assert match_emergency_keywords("I can’t breathe")
    assert match_emergency_keywords("I've fallen and hurt my hip")
    assert match_emergency_keywords("I’ve just fallen in the kitchen")
    assert match_emergency_keywords("ive had a fall")
    assert match_emergency_keywords("I have fallen")
    assert not match_emergency_keywords("I've fallen asleep twice today")
    assert not match_emergency_keywords("I’ve fallen behind on my walks")
