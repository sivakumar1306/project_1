"""
Seed the Review-II fall-risk demo cohort into Supabase.

Usage:
  python seed_fall_demo.py              # (re)seed all 5 demo users, 35 days each
  python seed_fall_demo.py --fall-now   # only insert a fresh uncancelled fall for
                                        # the "recent fall" user (run right before
                                        # the demo; the biometric signal looks back 30 min)
  python seed_fall_demo.py --list       # print demo user ids

Re-run the full seed on the morning of the review: the data is generated
relative to "today", so seeding on Sunday would look 3 days stale on Wednesday.
Only the five demo user ids below are touched; real users are never modified.

Requires schema_fall.sql to have been run once in the Supabase SQL editor.
"""

import argparse
import sys
from datetime import datetime, timezone

from db.supabase import supabase
from scripts.fall_demo_data import DEMO_USERS, USERS_BY_KEY, generate_user, profile_row, recent_fall_event

TABLES = {
    "hr": "user_hr", "hrv": "user_hrv", "spo2": "user_spo2", "sleep": "user_sleep",
    "steps": "user_steps", "temp": "user_temp", "stress": "user_stress",
    "hr_readings": "user_hr_readings", "cycles": "user_cycles", "events": "user_fall_events",
}


def _insert(table: str, rows: list[dict]) -> None:
    for i in range(0, len(rows), 200):
        supabase.table(table).insert(rows[i:i + 200]).execute()


def seed_all() -> None:
    today = datetime.utcnow().date()
    now = datetime.now(timezone.utc)
    print(f"Seeding {len(DEMO_USERS)} demo users for {today} ...")
    for user in DEMO_USERS:
        uid = user["id"]
        rows = generate_user(user, today, now, include_recent_fall=True)
        try:
            supabase.table("user_profiles").upsert(profile_row(user)).execute()
        except Exception as e:
            print(f"  [WARN] profile upsert failed for {user['full_name']}: {e}")
        for key, table in TABLES.items():
            try:
                supabase.table(table).delete().eq("user_id", uid).execute()
                if rows[key]:
                    _insert(table, rows[key])
            except Exception as e:
                print(f"  [ERROR] {table} for {user['full_name']}: {e}")
                if table == "user_fall_events":
                    print("          -> did you run schema_fall.sql in the Supabase SQL editor?")
        print(f"  [OK] {user['full_name']:<40} {uid}")
    print("Done. The recent-fall event expires from the 30-minute safety window; "
          "run `python seed_fall_demo.py --fall-now` just before the demo.")


def fall_now() -> None:
    uid = USERS_BY_KEY["recent_fall"]
    supabase.table("user_fall_events").insert(recent_fall_event(uid, minutes_ago=1)).execute()
    print(f"[OK] Fresh uncancelled fall inserted for {uid} (valid for 30 minutes).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fall-now", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for u in DEMO_USERS:
            print(f"{u['id']}  {u['full_name']}")
        sys.exit(0)
    fall_now() if a.fall_now else seed_all()
