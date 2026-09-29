"""
Offline tests for GET /api/v1/fall-risk/{user_id}/breakdown (no Supabase).
The synthetic demo cohort stands in for Supabase, as in tests/test_demo_api.py.

Run:  python -m pytest tests/test_fall_risk_breakdown.py -q
"""
import math
import os
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import agent.fall_risk as fall_risk
import main
from scripts.fall_demo_data import DEMO_USERS, USERS_BY_KEY, history_for

BY_ID = {u["id"]: u for u in DEMO_USERS}
OTHER_ID = "00000000-0000-0000-0000-000000000000"
ALL_IDS = [u["id"] for u in DEMO_USERS]


@pytest.fixture
def client(monkeypatch):
    def fake_fetch(user_id, today, days=45):
        if user_id not in BY_ID:
            return {"profile": {}, "daily": {}, "cycles": [], "events": []}
        return history_for(BY_ID[user_id], today, datetime.now(timezone.utc))
    monkeypatch.setattr(fall_risk, "fetch_fall_history", fake_fetch)
    monkeypatch.setattr(fall_risk, "_log_result", lambda *a, **k: None)
    fall_risk._CACHE.clear()
    return TestClient(main.app)


def _get(client, uid):
    r = client.get(f"/api/v1/fall-risk/{uid}/breakdown")
    assert r.status_code == 200
    return r.json()


@pytest.mark.parametrize("uid", ALL_IDS)
def test_points_add_up_to_the_unrounded_score(client, uid):
    d = _get(client, uid)
    total = sum(f["points"] for f in d["features"]) + d["near_falls"]["points"] + d["fall_history"]["points"]
    assert abs(total - d["total"]["frs_raw"]) < 0.01
    assert abs(sum(l["points"] for l in d["layers"]) - d["total"]["frs_raw"]) < 0.01
    assert d["total"]["frs"] == int(round(d["total"]["frs_raw"]))
    assert d["total"]["tier"] == fall_risk.tier_for(d["total"]["frs"])


@pytest.mark.parametrize("uid", ALL_IDS)
def test_adjusted_weights_sum_to_one_and_layers_are_complete(client, uid):
    d = _get(client, uid)
    assert [l["key"] for l in d["layers"]] == ["FH", "MO", "P", "RC"]
    assert abs(sum(l["adjusted_weight"] for l in d["layers"]) - 1) < 1e-9
    assert abs(sum(l["base_weight"] for l in d["layers"]) - 1) < 1e-9
    for l in d["layers"]:
        assert l["evidence_ratio"] == fall_risk.EVIDENCE_RATIOS[l["key"]]
        assert abs(l["ln_ratio"] - math.log(l["evidence_ratio"])) < 1e-12
        assert l["source"] and l["ratio_type"] in ("OR", "RR")
        assert 0 <= l["coverage_pct"] <= 100


@pytest.mark.parametrize("uid", ALL_IDS)
def test_every_present_feature_has_z_and_strain(client, uid):
    d = _get(client, uid)
    assert [f["key"] for f in d["features"]] == list(fall_risk.FEATURES)
    for f in d["features"]:
        if f["present"]:
            assert f["z"] is not None and f["z_risk"] is not None and 0 <= f["strain"] <= 100
            assert f["baseline"] is not None and f["n_baseline_days"] >= d["constants"]["min_baseline_days"]
            assert f["baseline_mode"] and f["risk_direction"] in ("low", "high")
            assert f["z_risk"] == pytest.approx(-f["z"] if f["risk_direction"] == "low" else f["z"])
        else:
            assert f["reason"]
    # within a layer the feature shares (plus the near-fall share for Motion) sum to 1
    for L in ("P", "RC", "MO"):
        share = sum(f["weight_in_layer"] for f in d["features"] if f["layer"] == L and f["present"])
        if L == "MO":
            share += d["near_falls"]["weight_in_layer"]
        assert share == pytest.approx(1.0)


def test_meera_cycle_block_and_constants(client):
    d = _get(client, USERS_BY_KEY["luteal_strain"])
    cy = d["cycle"]
    assert cy["applied"] and cy["phase"] == "luteal" and cy["day"] > 0
    assert isinstance(cy["score_without_correction"], int)
    assert cy["tier_without_correction"] == fall_risk.tier_for(cy["score_without_correction"])
    assert any(f["baseline_mode"] == "phase-matched (luteal)" for f in d["features"])
    assert d["constants"] == {"dead_zone_z": 0.5, "full_strain_z": 3.0, "baseline_days": 28, "min_baseline_days": 7}
    assert d["total"]["thresholds"] == [40, 70]


def test_lakshmi_fall_history_and_near_falls(client):
    d = _get(client, USERS_BY_KEY["recent_fall"])
    fh = d["fall_history"]
    assert fh["falls_365"] >= 2 and fh["fh_risk"] == 100 and fh["points"] > 0
    assert d["near_falls"]["count"] == 2 and d["near_falls"]["strain"] == pytest.approx(100.0)


def test_breakdown_matches_the_score_endpoint(client):
    uid = USERS_BY_KEY["luteal_strain"]
    score = client.get(f"/api/v1/fall-risk/{uid}").json()
    d = _get(client, uid)
    assert d["total"]["frs"] == score["score"] and d["total"]["tier"] == score["tier"]
    assert round(d["total"]["frs_raw"], 1) == score["points_total"]


def test_stale_user_is_flagged(client):
    d = _get(client, USERS_BY_KEY["stale"])
    assert d["stale"] and d["data_age_days"] > 2 and d["enough_history"]


def test_unknown_user_gets_friendly_message(client):
    d = _get(client, OTHER_ID)
    assert d["enough_history"] is False and d["message"]
    assert d["total"]["frs_raw"] == 0 and all(not f["present"] for f in d["features"])


def test_engine_error_returns_502(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("db down")
    monkeypatch.setattr(fall_risk, "get_fall_risk", boom)
    r = client.get(f"/api/v1/fall-risk/{USERS_BY_KEY['low_risk']}/breakdown")
    assert r.status_code == 502 and "error" in r.json()


def test_no_brand_or_ml_wording(client):
    text = client.get(f"/api/v1/fall-risk/{USERS_BY_KEY['luteal_strain']}/breakdown").text.lower()
    for word in ("medxai", "machine learning", "neural", "model training"):
        assert word not in text
