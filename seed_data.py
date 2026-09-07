import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(url, key)

TEST_USER_ID = "00000000-0000-0000-0000-000000000000"

def seed_database():
    print(f"Seeding demo biometric data for user_id={TEST_USER_ID}...")

    # 1. User Profile
    try:
        supabase.table("user_profiles").upsert({
            "id": TEST_USER_ID,
            "user_id": TEST_USER_ID,
            "full_name": "John Doe",
            "age": 34,
            "gender": "Male",
            "height_cm": 178,
            "weight_kg": 75
        }).execute()
        print("  [OK] user_profiles seeded")
    except Exception as e:
        print("  [WARN] user_profiles seed error:", e)

    # 2. Live Heart Rate
    try:
        supabase.table("user_hr_readings").insert({
            "user_id": TEST_USER_ID,
            "measured_at": datetime.utcnow().isoformat(),
            "value_bpm": 74,
            "source": "smart_ring"
        }).execute()
        print("  [OK] user_hr_readings seeded")
    except Exception as e:
        print("  [WARN] user_hr_readings seed error:", e)

    # 3. Sleep (last 3 days)
    try:
        today = datetime.utcnow().date()
        sleep_data = [
            {"user_id": TEST_USER_ID, "date": str(today - timedelta(days=2)), "total_duration": 420, "sleep_score": 82},
            {"user_id": TEST_USER_ID, "date": str(today - timedelta(days=1)), "total_duration": 450, "sleep_score": 88},
            {"user_id": TEST_USER_ID, "date": str(today), "total_duration": 465, "sleep_score": 90},
        ]
        supabase.table("user_sleep").upsert(sleep_data).execute()
        print("  [OK] user_sleep seeded")
    except Exception as e:
        print("  [WARN] user_sleep seed error:", e)

    # 4. Daily HR
    try:
        hr_data = [
            {"user_id": TEST_USER_ID, "date": str(today - timedelta(days=2)), "avg_hr": 72, "min_hr": 58, "max_hr": 110},
            {"user_id": TEST_USER_ID, "date": str(today - timedelta(days=1)), "avg_hr": 70, "min_hr": 56, "max_hr": 115},
            {"user_id": TEST_USER_ID, "date": str(today), "avg_hr": 68, "min_hr": 55, "max_hr": 105},
        ]
        supabase.table("user_hr").upsert(hr_data).execute()
        print("  [OK] user_hr seeded")
    except Exception as e:
        print("  [WARN] user_hr seed error:", e)

    # 5. SpO2
    try:
        spo2_data = [
            {"user_id": TEST_USER_ID, "date": str(today - timedelta(days=2)), "avg_spo2": 98, "min_spo2": 96, "max_spo2": 99},
            {"user_id": TEST_USER_ID, "date": str(today - timedelta(days=1)), "avg_spo2": 97, "min_spo2": 95, "max_spo2": 99},
            {"user_id": TEST_USER_ID, "date": str(today), "avg_spo2": 98, "min_spo2": 96, "max_spo2": 100},
        ]
        supabase.table("user_spo2").upsert(spo2_data).execute()
        print("  [OK] user_spo2 seeded")
    except Exception as e:
        print("  [WARN] user_spo2 seed error:", e)

    # 6. HRV
    try:
        hrv_data = [
            {"user_id": TEST_USER_ID, "date": str(today - timedelta(days=2)), "avg_hrv": 52, "min_hrv": 34, "max_hrv": 78},
            {"user_id": TEST_USER_ID, "date": str(today - timedelta(days=1)), "avg_hrv": 56, "min_hrv": 38, "max_hrv": 82},
            {"user_id": TEST_USER_ID, "date": str(today), "avg_hrv": 60, "min_hrv": 40, "max_hrv": 85},
        ]
        supabase.table("user_hrv").upsert(hrv_data).execute()
        print("  [OK] user_hrv seeded")
    except Exception as e:
        print("  [WARN] user_hrv seed error:", e)

    # 7. Blood Pressure
    try:
        supabase.table("user_bp").insert({
            "user_id": TEST_USER_ID,
            "measured_at": datetime.utcnow().isoformat(),
            "systolic": 118,
            "diastolic": 78
        }).execute()
        print("  [OK] user_bp seeded")
    except Exception as e:
        print("  [WARN] user_bp seed error:", e)

    # 8. Steps
    try:
        steps_data = [
            {"user_id": TEST_USER_ID, "date": str(today - timedelta(days=2)), "steps": 7500, "calories": 420},
            {"user_id": TEST_USER_ID, "date": str(today - timedelta(days=1)), "steps": 9200, "calories": 510},
            {"user_id": TEST_USER_ID, "date": str(today), "steps": 10400, "calories": 580},
        ]
        supabase.table("user_steps").upsert(steps_data).execute()
        print("  [OK] user_steps seeded")
    except Exception as e:
        print("  [WARN] user_steps seed error:", e)

    # 9. Temperature
    try:
        supabase.table("user_temp").insert({
            "user_id": TEST_USER_ID,
            "measured_at": datetime.utcnow().isoformat(),
            "value_c": 36.6
        }).execute()
        print("  [OK] user_temp seeded")
    except Exception as e:
        print("  [WARN] user_temp seed error:", e)

    # 10. Stress
    try:
        supabase.table("user_stress").insert({
            "user_id": TEST_USER_ID,
            "measured_at": datetime.utcnow().isoformat(),
            "stress_value": 28,
            "label": "Low"
        }).execute()
        print("  [OK] user_stress seeded")
    except Exception as e:
        print("  [WARN] user_stress seed error:", e)

    print("\nDatabase seeding completed successfully for demo user!")

if __name__ == "__main__":
    seed_database()
