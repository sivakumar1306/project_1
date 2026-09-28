"""
Chat-card builders must never show a value that is not in the user's rows
(offline: a fake Supabase client stands in for the database).
Run:  python -m pytest tests/test_cards.py -q
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent.graph as graph

UID = "11111111-1111-4111-8111-00000000abcd"
NOW = datetime.utcnow()
DAY = lambda back: (NOW - timedelta(days=back)).strftime("%Y-%m-%d")
TS = lambda back, h=9: (NOW - timedelta(days=back)).replace(hour=h, minute=0, second=0, microsecond=0).isoformat()

ALL_CARDS = {
    "sleep": graph.get_sleep_card_data, "hr": graph.get_hr_card_data, "spo2": graph.get_spo2_card_data,
    "hrv": graph.get_hrv_card_data, "bp": graph.get_bp_card_data, "steps": graph.get_steps_card_data,
    "temperature": graph.get_temperature_card_data, "stress": graph.get_stress_card_data,
    "cycle": graph.get_cycle_card_data,
}


class FakeQuery:
    def __init__(self, rows, fail):
        self.rows, self.fail = rows, fail

    def __getattr__(self, name):          # select / eq / gte / lte / order / limit ... all chain
        return lambda *a, **k: self

    def execute(self):
        if self.fail:
            raise RuntimeError("db down")
        return SimpleNamespace(data=list(self.rows))


class FakeSupabase:
    def __init__(self, tables=None, fail=False):
        self.tables, self.fail = tables or {}, fail

    def table(self, name):
        return FakeQuery(self.tables.get(name, []), self.fail)


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _card(monkeypatch, key, tables=None, fail=False, uid=UID):
    monkeypatch.setattr(graph, "supabase", FakeSupabase(tables, fail))
    return run(ALL_CARDS[key](uid))


# ── no data / failure / anonymous -> no card ───────────────────────────────
@pytest.mark.parametrize("key", list(ALL_CARDS))
def test_user_with_no_rows_gets_no_card(monkeypatch, key):
    assert _card(monkeypatch, key) is None


@pytest.mark.parametrize("key", list(ALL_CARDS))
def test_failed_query_gets_no_card(monkeypatch, key):
    assert _card(monkeypatch, key, fail=True) is None


@pytest.mark.parametrize("key", list(ALL_CARDS))
def test_anonymous_user_gets_no_demo_card(monkeypatch, key):
    assert _card(monkeypatch, key, uid="anonymous") is None
    assert _card(monkeypatch, key, uid="") is None


# ── every value in a card traces back to a row ─────────────────────────────
def _assert_trend_from_rows(data, row_values):
    present = [v for v in data["values"] if v is not None]
    assert present and set(present) <= set(row_values)
    assert data["avg"] == round(sum(present) / len(present))          # mean of the shown rows, nothing else
    assert 0 not in data["values"] or 0 in row_values                  # missing days are None, never a fake 0


@pytest.mark.parametrize("key,table,col,mn,mx", [
    ("hr", "user_hr", "avg_hr", "min_hr", "max_hr"),
    ("spo2", "user_spo2", "avg_spo2", "min_spo2", "max_spo2"),
    ("hrv", "user_hrv", "avg_hrv", "min_hrv", "max_hrv"),
    ("steps", "user_steps", "steps", None, None),
])
def test_daily_trend_cards_only_show_row_values(monkeypatch, key, table, col, mn, mx):
    rows = [{"date": DAY(0), col: 71.5, **({mn: 55, mx: 120} if mn else {})},
            {"date": DAY(3), col: 64, **({mn: 50, mx: 101} if mn else {})}]
    card = _card(monkeypatch, key, {table: rows})
    d = card["data"]
    assert len(d["values"]) == 7 and d["values"].count(None) == 5
    _assert_trend_from_rows(d, [71.5, 64])                              # 71.5 kept, not truncated to 71
    if mn:
        assert d["min"] == 50 and d["max"] == 120


def test_daily_trend_min_max_absent_when_columns_absent(monkeypatch):
    d = _card(monkeypatch, "hr", {"user_hr": [{"date": DAY(0), "avg_hr": 70}]})["data"]
    assert d["min"] is None and d["max"] is None


def test_bp_card_uses_latest_reading_per_day_and_no_fake_zeros(monkeypatch):
    rows = [{"measured_at": TS(1, 8), "systolic": 130, "diastolic": 85},
            {"measured_at": TS(1, 20), "systolic": 118, "diastolic": 78}]
    d = _card(monkeypatch, "bp", {"user_bp": rows})["data"]
    assert [v for v in d["sbp_values"] if v is not None] == [118]
    assert [v for v in d["dbp_values"] if v is not None] == [78]
    assert d["sbp_values"].count(None) == 6 and d["sbp_avg"] == 118 and d["dbp_avg"] == 78


@pytest.mark.parametrize("key,table,col", [("temperature", "user_temp", "value_c"), ("stress", "user_stress", "stress_value")])
def test_reading_trend_cards_only_show_row_values(monkeypatch, key, table, col):
    rows = [{"measured_at": TS(0, 3), col: 36.4 if key == "temperature" else 30},
            {"measured_at": TS(0, 9), col: 36.6 if key == "temperature" else 50},
            {"measured_at": TS(2, 3), col: 36.2 if key == "temperature" else 20}]
    d = _card(monkeypatch, key, {table: rows})["data"]
    raw = [r[col] for r in rows]
    assert d["min"] == min(raw) and d["max"] == max(raw)
    assert d["values"].count(None) == 5                                 # 2 days with readings, 5 without
    per_day = [v for v in d["values"] if v is not None]
    if key == "temperature":
        assert per_day == [36.2, 36.5]                                  # daily means of the rows
    else:
        assert per_day == [20, 40]


def test_sleep_card_omits_stages_the_row_does_not_have(monkeypatch):
    d = _card(monkeypatch, "sleep", {"user_sleep": [{"date": DAY(0), "total_duration": 450, "sleep_score": 80}]})["data"]
    assert d == {"total_label": "7 hours and 30 minutes"}               # no invented awake/light/deep minutes


def test_sleep_card_copies_stages_when_present_and_handles_seconds(monkeypatch):
    row = {"date": DAY(0), "total_duration": 27000, "deep_sleep_min": 95, "light_sleep_min": 250, "time_awake_min": 12}
    d = _card(monkeypatch, "sleep", {"user_sleep": [row]})["data"]
    assert d["total_label"] == "7 hours and 30 minutes"
    assert (d["deep_sleep_min"], d["light_sleep_min"], d["time_awake_min"]) == (95, 250, 12)


def test_sleep_card_without_duration_is_none(monkeypatch):
    assert _card(monkeypatch, "sleep", {"user_sleep": [{"date": DAY(0), "sleep_score": 80}]}) is None


def test_cycle_card_copies_logged_values_and_assumes_no_defaults(monkeypatch):
    start = (NOW - timedelta(days=3)).strftime("%Y-%m-%d")
    d = _card(monkeypatch, "cycle", {"user_cycles": [{"period_start": start, "cycle_length": 30, "period_length": 6}]})["data"]
    assert (d["period_start"], d["cycle_length"], d["period_length"]) == (start, 30, 6)
    assert d["current_day"] == 4 and d["days_until_next"] == 27 and d["phase"] == "Menstrual Phase"

    d2 = _card(monkeypatch, "cycle", {"user_cycles": [{"period_start": start}]})["data"]
    assert d2["cycle_length"] is None and d2["period_length"] is None          # no assumed 28 / 5
    assert d2["current_day"] is None and d2["days_until_next"] is None and d2["phase"] is None
