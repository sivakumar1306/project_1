"""
Checks that keep docs/METHODS.md consistent with the code it describes (offline).
Run:  python -m pytest tests/test_docs.py -q
"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _methods() -> str:
    with open(os.path.join(ROOT, "docs", "METHODS.md"), encoding="utf-8") as f:
        return f.read()


def test_version_c_d_hr_filter_note_holds_for_demo_cohort():
    """METHODS.md says the fall evaluation is unaffected by Version D's current-HR source filter
    because every demo heart-rate reading passes it (non-NULL source, not "demo_seed")."""
    from scripts.fall_demo_data import DEMO_USERS, generate_user
    now = datetime.now(timezone.utc)
    for u in DEMO_USERS:
        readings = generate_user(u, now.date(), now)["hr_readings"]
        assert readings, u["scenario"]
        for r in readings:
            assert r.get("source") is not None and r["source"] != "demo_seed", (u["scenario"], r)
    assert "### Known limitations of the prototype" in _methods()


def test_version_d_still_uses_the_documented_filter():
    with open(os.path.join(ROOT, "agent", "tools.py"), encoding="utf-8") as f:
        src = f.read()
    # If this filter changes, update "Known limitations of the prototype" in docs/METHODS.md.
    assert src.count('.neq("source", "demo_seed")') == 1
