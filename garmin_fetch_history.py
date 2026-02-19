"""
Garmin Connect – Multi-Day History Fetcher (with Power Data)
=============================================================
Fetches HRV, sleep, and activity data (incl. normalized power)
for TSS / hrTSS calculation over the last N days.

Requirements:
    pip install garminconnect

Usage:
    1. Set your credentials + FTP + LTHR below
    2. python garmin_fetch_history.py
"""

import os
import json
import time
from datetime import date, timedelta

try:
    import garminconnect
except ImportError:
    raise SystemExit("Run:  pip install garminconnect")


# ══════════════════════════════════════════════════════════════
#  !! CONFIGURE THESE BEFORE RUNNING !!
# ══════════════════════════════════════════════════════════════
EMAIL     = os.getenv("GARMIN_EMAIL",    "your@email.com")
PASSWORD  = os.getenv("GARMIN_PASSWORD", "yourpassword")

FTP_WATTS = 250     # Your Functional Threshold Power in watts
LTHR_BPM  = 168     # Your Lactate Threshold Heart Rate in bpm
            # If unsure about LTHR: use ~90% of your max HR
# ══════════════════════════════════════════════════════════════

DAYS_BACK   = 28
OUTPUT_FILE = "garmin_history.json"
SLEEP_SEC   = 1.0   # pause between API calls to avoid rate-limiting


# ── TSS / hrTSS formulas ──────────────────────────────────────────────────────

def calc_tss(duration_sec: float, normalized_power: float, ftp: float):
    """
    Power-based TSS:
      TSS = (t × NP × IF) / (FTP × 3600) × 100
      IF  = NP / FTP

    100 TSS ≈ one hour at exactly FTP.
    """
    if not (duration_sec and normalized_power and ftp and ftp > 0):
        return None
    intensity_factor = normalized_power / ftp
    tss = (duration_sec * normalized_power * intensity_factor) / (ftp * 3600) * 100
    return round(tss, 1)


def calc_hrtss(duration_sec: float, avg_hr: float, lthr: float):
    """
    HR-based TSS (Coggan analogy):
      hrTSS = duration_hours × (avg_hr / LTHR)² × 100

    Used when no power data is available (running, swimming, rowing, etc.)
    """
    if not (duration_sec and avg_hr and lthr and lthr > 0):
        return None
    intensity_factor_hr = avg_hr / lthr
    hrtss = (duration_sec / 3600) * (intensity_factor_hr ** 2) * 100
    return round(hrtss, 1)


def score_activity(act: dict, ftp: float, lthr: float) -> dict:
    """Pick TSS (power) or hrTSS (HR) depending on what data is available."""
    duration = act.get("duration_sec") or 0
    np       = act.get("normalized_power")
    avg_hr   = act.get("avg_hr")

    if np and np > 0 and ftp > 0:
        return {"score": calc_tss(duration, np, ftp), "method": "TSS"}
    if avg_hr and avg_hr > 0 and lthr > 0:
        return {"score": calc_hrtss(duration, avg_hr, lthr), "method": "hrTSS"}
    return {"score": None, "method": None}


# ── Garmin login ──────────────────────────────────────────────────────────────

def connect() -> garminconnect.Garmin:
    print("Connecting to Garmin Connect …")
    client = garminconnect.Garmin(EMAIL, PASSWORD)
    client.login()
    print("✓ Logged in\n")
    return client


# ── Fetch one day ─────────────────────────────────────────────────────────────

def fetch_day(client: garminconnect.Garmin, day: date) -> dict:
    d      = day.isoformat()
    result = {"date": d}

    # HRV ──────────────────────────────────────────────────────
    try:
        hrv     = client.get_hrv_data(d) or {}
        summary = hrv.get("hrvSummary", {})
        result["hrv_last_night"] = summary.get("lastNight")
        result["hrv_weekly_avg"] = summary.get("weeklyAvg")
        result["hrv_status"]     = summary.get("hrvStatusEnum")
    except Exception as e:
        result["hrv_last_night"] = result["hrv_weekly_avg"] = result["hrv_status"] = None
        print(f"    ✗ HRV: {e}")

    # Sleep + sleep-time heart rate ────────────────────────────
    try:
        sleep = client.get_sleep_data(d) or {}
        daily = sleep.get("dailySleepDTO", {})
        result["sleep_seconds"] = daily.get("sleepTimeSeconds")
        result["sleep_score"]   = daily.get("sleepScores", {}).get("overall", {}).get("value")

        sleep_start = daily.get("sleepStartTimestampGMT")
        sleep_end   = daily.get("sleepEndTimestampGMT")

        hr_data   = client.get_heart_rates(d) or {}
        hr_values = hr_data.get("heartRateValues", [])
        result["resting_hr"] = hr_data.get("restingHeartRate")

        if hr_values and sleep_start and sleep_end:
            sleep_hrs = [v[1] for v in hr_values
                         if v[1] is not None and sleep_start <= v[0] <= sleep_end]
            result["sleep_avg_hr"] = (
                round(sum(sleep_hrs) / len(sleep_hrs), 1) if sleep_hrs else result["resting_hr"]
            )
        else:
            result["sleep_avg_hr"] = result["resting_hr"]

    except Exception as e:
        result["sleep_seconds"] = result["sleep_score"] = None
        result["sleep_avg_hr"]  = result["resting_hr"]  = None
        print(f"    ✗ Sleep: {e}")

    # Activities + TSS ─────────────────────────────────────────
    try:
        raw_acts = client.get_activities_by_date(d, d) or []
        activities = []

        for a in raw_acts:
            act_id = a.get("activityId")

            # Try to get normalized power from the detailed activity endpoint
            np_watts = a.get("normPower")   # sometimes present at list level
            if not np_watts and act_id:
                try:
                    details  = client.get_activity(act_id) or {}
                    np_watts = (
                        details.get("summaryDTO", {}).get("normalizedPower")
                        or details.get("summaryDTO", {}).get("normPower")
                    )
                    time.sleep(0.4)
                except Exception:
                    pass

            act = {
                "id":               act_id,
                "name":             a.get("activityName"),
                "type":             (a.get("activityType") or {}).get("typeKey", ""),
                "duration_sec":     a.get("duration"),
                "distance_m":       a.get("distance"),
                "avg_hr":           a.get("averageHR"),
                "max_hr":           a.get("maxHR"),
                "calories":         a.get("calories"),
                "avg_power":        a.get("avgPower"),
                "normalized_power": np_watts,
            }

            scored = score_activity(act, FTP_WATTS, LTHR_BPM)
            act["stress_score"]  = scored["score"]
            act["stress_method"] = scored["method"]
            activities.append(act)

        result["activities"] = activities
        result["daily_tss"]  = round(
            sum(a["stress_score"] for a in activities if a["stress_score"]), 1
        ) if any(a["stress_score"] for a in activities) else 0
        result["tss_methods"] = list({a["stress_method"] for a in activities if a["stress_method"]})

    except Exception as e:
        result["activities"]  = []
        result["daily_tss"]   = 0
        result["tss_methods"] = []
        print(f"    ✗ Activities: {e}")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    today = date.today()

    # ── Load existing data ────────────────────────────────────────────────────
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            history = json.load(f)
        existing_dates = {r["date"] for r in history}
        print(f"✓ Loaded {len(history)} existing records from '{OUTPUT_FILE}'")
    else:
        history        = []
        existing_dates = set()
        print(f"No existing file found — fetching full {DAYS_BACK}-day history")

    # ── Determine which days to fetch ────────────────────────────────────────
    # Always re-fetch today (data may be incomplete earlier in the day)
    # and yesterday (activities sometimes sync late).
    # For older days: only fetch if missing.
    days_to_fetch = []
    for i in range(DAYS_BACK):
        day   = today - timedelta(days=i)
        d_str = day.isoformat()
        if d_str not in existing_dates or i <= 1:
            days_to_fetch.append(day)

    if not days_to_fetch:
        print("✓ All days already up to date — nothing to fetch")
        return

    print(f"Fetching {len(days_to_fetch)} day(s): "
          f"{days_to_fetch[-1].isoformat()} → {days_to_fetch[0].isoformat()}\n")

    # ── Fetch missing / recent days ───────────────────────────────────────────
    client      = connect()
    new_records = {}

    for day in days_to_fetch:
        print(f"Fetching {day.isoformat()} …", end=" ", flush=True)
        record = fetch_day(client, day)
        new_records[day.isoformat()] = record
        print(f"✓  TSS = {record.get('daily_tss', 0)}")
        time.sleep(SLEEP_SEC)

    # ── Merge: update existing records, append new ones ───────────────────────
    # Build a dict keyed by date for easy update
    history_map = {r["date"]: r for r in history}
    history_map.update(new_records)

    # Keep only last DAYS_BACK days, sorted ascending
    cutoff  = (today - timedelta(days=DAYS_BACK - 1)).isoformat()
    history = sorted(
        [r for r in history_map.values() if r["date"] >= cutoff],
        key=lambda r: r["date"]
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)

    print(f"\n✓ Saved {len(history)} days → '{OUTPUT_FILE}'")
    print(f"  ({len(new_records)} updated/added,  FTP: {FTP_WATTS} W,  LTHR: {LTHR_BPM} bpm)")


if __name__ == "__main__":
    main()
