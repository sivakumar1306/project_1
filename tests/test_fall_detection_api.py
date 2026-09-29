"""
Offline tests for the PIFv3 fall-detection demo endpoints (no Supabase, no dataset).
Run:  python -m pytest tests/test_fall_detection_api.py -q
"""
import json
import os
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import agent.fall_risk as fall_risk
import main
import routers.fall_detection as fd
from scripts.fall_demo_data import USERS_BY_KEY
from tests.test_demo_api import RecordingSupabase

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _committed(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return json.load(f)


SAMPLES = _committed("pifv3_demo_samples.json")
FALL_ID = next(s["sample_id"] for s in SAMPLES["samples"] if s["predicted"] == "FALL")
NO_FALL_ID = next(s["sample_id"] for s in SAMPLES["samples"] if s["predicted"] == "NO FALL")


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def fake_db(monkeypatch):
    fake = RecordingSupabase()
    monkeypatch.setattr(fd, "supabase", fake)
    return fake


@pytest.fixture
def missing_files(monkeypatch, tmp_path):
    for attr in ("RESULTS_PATH", "SAMPLES_PATH", "CHART_PATH"):
        monkeypatch.setattr(fd, attr, str(tmp_path / "missing"))


def test_results_are_the_committed_file(client):
    d = client.get("/api/v1/detection/results").json()
    assert d == _committed("pifv3_results.json")
    assert set(d["results"]) == {"primary_falls_vs_daily_activities", "secondary_falls_vs_all"}


def test_chart_is_served(client):
    r = client.get("/api/v1/detection/chart")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"


def test_samples_list_hides_labels_and_predictions(client):
    d = client.get("/api/v1/detection/samples").json()
    assert d["available"] is True
    assert d["held_out_accuracy"] == SAMPLES["held_out_accuracy"]
    assert d["trained_on_participants"] == SAMPLES["trained_on_participants"]
    assert [s["sample_id"] for s in d["samples"]] == [s["sample_id"] for s in SAMPLES["samples"]]
    for s in d["samples"]:
        assert set(s) == {"sample_id", "pid"}
        assert s["pid"] in SAMPLES["held_out_participants"]
    for leaked in ("is_fall", "predicted", "fall_probability", "label", "correct", "Fall while"):
        assert leaked not in json.dumps(d["samples"])


def test_sample_returns_the_full_record(client):
    want = next(s for s in SAMPLES["samples"] if s["sample_id"] == FALL_ID)
    assert client.get(f"/api/v1/detection/sample/{FALL_ID}").json() == want
    assert client.get("/api/v1/detection/sample/S99").status_code == 404


def test_send_to_safety_logs_one_fall_for_lakshmi(client, fake_db):
    fall_risk._CACHE["x"] = 1
    r = client.post(f"/api/v1/detection/sample/{FALL_ID}/send-to-safety")
    assert r.status_code == 200 and r.json()["ok"]
    assert len(fake_db.inserted) == 1
    table, row = fake_db.inserted[0]
    assert table == "user_fall_events" and row["user_id"] == USERS_BY_KEY["recent_fall"]
    assert row["event_type"] == "fall" and row["user_cancelled"] is False and row["dispatched"] is False
    assert row["source"] == "ml_detector_demo"
    detected = datetime.fromisoformat(row["detected_at"])
    assert abs((datetime.now(timezone.utc) - detected).total_seconds()) < 60
    assert not fall_risk._CACHE


def test_send_to_safety_refuses_no_fall_and_unknown_samples(client, fake_db):
    r = client.post(f"/api/v1/detection/sample/{NO_FALL_ID}/send-to-safety")
    assert r.status_code == 400 and "error" in r.json()
    assert client.post("/api/v1/detection/sample/S99/send-to-safety").status_code == 404
    assert fake_db.inserted == []


def test_send_to_safety_database_error_returns_502(client, monkeypatch):
    class Broken:
        def table(self, name):
            raise RuntimeError("db down")
    monkeypatch.setattr(fd, "supabase", Broken())
    r = client.post(f"/api/v1/detection/sample/{FALL_ID}/send-to-safety")
    assert r.status_code == 502 and "error" in r.json()


def test_missing_files_return_available_false(client, fake_db, missing_files):
    assert client.get("/api/v1/detection/results").json() == {"available": False}
    assert client.get("/api/v1/detection/samples").json() == {"available": False}
    assert client.get(f"/api/v1/detection/sample/{FALL_ID}").json() == {"available": False}
    assert client.post(f"/api/v1/detection/sample/{FALL_ID}/send-to-safety").json() == {"available": False}
    assert client.get("/api/v1/detection/chart").status_code == 404
    assert fake_db.inserted == []
    # the rest of the demo keeps working
    assert client.get("/api/v1/demo/users").status_code == 200
    assert client.get("/demo").status_code == 200


def test_demo_page_has_the_detection_section_and_no_brand(client):
    html = client.get("/demo").text
    assert "Fall Detection — trained ML model (PIFv3, 32 real participants)" in html
    assert ("Detects whether a fall just happened from motion signals. Different from the daily "
            "fall-risk score above, which estimates risk before a fall.") in html
    assert "Send to safety system (as Lakshmi&#39;s ring)" in html or "Send to safety system (as Lakshmi's ring)" in html
    assert "medxai" not in html.lower()
    # the section sits below the AI assistant
    assert html.index("<h2>AI assistant</h2>") < html.index("Fall Detection — trained ML model")
