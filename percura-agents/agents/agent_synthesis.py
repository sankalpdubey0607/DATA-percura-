"""
agent_synthesis.py - Percura Time-Aware Synthesis Agent (Phase 4 Flywheel)
===========================================================================
Reads extracted behavioral data from ALL sources (Play Store, Reddit, Social).
Calculates Historical Baseline vs Recent Events (<= 14 days).
Applies a 30% Exponential Moving Average (EMA) to recent events to dynamically 
update the simulation parameters (Patience, Trust, Effort).
Logs changes to a version history file.
"""
import os
import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
INPUT_CSVS = [
    os.path.join(DATA_DIR, "extracted_behaviors_1209.csv"), # Our benchmark
    os.path.join(DATA_DIR, "extracted_reddit.csv")
    # In the future, we will add extracted_social.csv here
]
OUTPUT_JSON = os.path.join(DATA_DIR, "cohort_parameters.json")
HISTORY_JSON = os.path.join(DATA_DIR, "cohort_history.json")

# Tuning Parameters
RECENT_WINDOW_DAYS = 14
RECENT_WEIGHT = 0.30
BASELINE_WEIGHT = 0.70

def parse_date(date_str):
    try:
        # Expected format: YYYY-MM-DD
        return datetime.strptime(date_str.split("T")[0], "%Y-%m-%d")
    except Exception:
        return datetime.now() - timedelta(days=100) # Default to historical if missing

def synthesize_parameters():
    print("=" * 60)
    print("  PERCURA DYNAMIC SYNTHESIS ENGINE (TIME-AWARE)")
    print("=" * 60)

    # Data structures for aggregation
    # We split into 'historical' and 'recent'
    cohorts = defaultdict(lambda: {
        "historical": {"count": 0, "gave_up": 0, "trust_lost": 0, "confused": 0, "effort": 0},
        "recent": {"count": 0, "gave_up": 0, "trust_lost": 0, "confused": 0, "effort": 0}
    })

    cutoff_date = datetime.now() - timedelta(days=RECENT_WINDOW_DAYS)

    # Read data from all available sources
    total_rows = 0
    for input_file in INPUT_CSVS:
        if not os.path.exists(input_file):
            continue
        
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                region = row.get("region_hint", "unknown")
                literacy = row.get("literacy_hint", "unknown")
                income = row.get("income_hint", "unknown")
                device = row.get("device_hint", "unknown")
                
                if device == "unknown" and income == "unknown":
                    continue

                cohort_key = f"{region}-{literacy}-{income}-{device}"
                row_date = parse_date(row.get("date", ""))
                
                window = "recent" if row_date >= cutoff_date else "historical"
                stats = cohorts[cohort_key][window]
                
                stats["count"] += 1
                total_rows += 1
                
                if str(row.get("gave_up", "")).lower() == "true":
                    stats["gave_up"] += 1
                    
                trust = str(row.get("trust_signal", "")).lower()
                if trust in ["lost_trust", "never_had_trust", "did_not_trust"]:
                    stats["trust_lost"] += 1
                    
                if str(row.get("emotion", "")).lower() == "confused" or "confusing_language" in str(row.get("friction_type", "")):
                    stats["confused"] += 1
                    
                if str(row.get("effort_complained", "")).lower() == "true" or "too_many_steps" in str(row.get("friction_type", "")):
                    stats["effort"] += 1

    print(f"Processed {total_rows} total rows across all data sources.")
    
    # Filter for valid cohorts (need at least 5 historical data points)
    valid_cohorts = {k: v for k, v in cohorts.items() if v["historical"]["count"] >= 5}
    print(f"Cohorts with sufficient data: {len(valid_cohorts)}")
    
    sim_parameters = {}
    
    for cohort, data in valid_cohorts.items():
        hist = data["historical"]
        rec = data["recent"]
        
        # --- CALCULATE HISTORICAL BASELINE ---
        h_total = hist["count"]
        b_gave_up = hist["gave_up"] / h_total
        b_trust = hist["trust_lost"] / h_total
        b_confused = hist["confused"] / h_total
        b_effort = hist["effort"] / h_total
        
        # --- CALCULATE RECENT TREND (if enough data) ---
        r_total = rec["count"]
        if r_total >= 3: # Need at least 3 recent complaints to shift the needle
            r_gave_up = rec["gave_up"] / r_total
            r_trust = rec["trust_lost"] / r_total
            r_confused = rec["confused"] / r_total
            r_effort = rec["effort"] / r_total
            
            # Apply EMA Blend
            final_gave_up = (b_gave_up * BASELINE_WEIGHT) + (r_gave_up * RECENT_WEIGHT)
            final_trust = (b_trust * BASELINE_WEIGHT) + (r_trust * RECENT_WEIGHT)
            final_confused = (b_confused * BASELINE_WEIGHT) + (r_confused * RECENT_WEIGHT)
            final_effort = (b_effort * BASELINE_WEIGHT) + (r_effort * RECENT_WEIGHT)
            is_trending = True
        else:
            # Fallback to pure baseline if no recent events
            final_gave_up, final_trust, final_confused, final_effort = b_gave_up, b_trust, b_confused, b_effort
            is_trending = False

        # --- MAP PERCENTAGES TO ENGINE PARAMETERS ---
        base_trust = 0.8
        calculated_trust = max(0.1, base_trust - (final_trust * 1.5))
        
        base_patience = 60
        patience_penalty = (final_gave_up * 30) + (final_effort * 20)
        calculated_patience = max(15, int(base_patience - patience_penalty))
        
        base_confusion = 0.7
        calculated_confusion = max(0.1, base_confusion - (final_confused * 2.0))
        
        base_effort_steps = 5
        calculated_effort_steps = max(1, int(base_effort_steps - (final_effort * 6)))

        sim_parameters[cohort] = {
            "metadata": {
                "historical_sample_size": h_total,
                "recent_sample_size": r_total,
                "is_trending": is_trending,
                "last_updated": datetime.now().isoformat()
            },
            "state_machine_parameters": {
                "patience_seconds": calculated_patience,
                "trust_score": round(calculated_trust, 2),
                "effort_tolerance_steps": calculated_effort_steps,
                "confusion_threshold": round(calculated_confusion, 2),
                "noise_vulnerability": 0.2
            }
        }
    
    # Save the live constraints
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(sim_parameters, f, indent=4)
        
    # --- VERSION HISTORY (Audit Trail) ---
    history = {}
    if os.path.exists(HISTORY_JSON):
        with open(HISTORY_JSON, "r", encoding="utf-8") as f:
            history = json.load(f)
            
    # Append current state
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    history[run_timestamp] = sim_parameters
    
    with open(HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
        
    print(f"\nSuccessfully synthesized and blended parameters for {len(sim_parameters)} archetypes.")
    print(f"Live Params saved to: {OUTPUT_JSON}")
    print(f"Version History saved to: {HISTORY_JSON}")
    
    print("\n--- NOTABLE RECENT TRENDS ---")
    trends_found = False
    for cohort, data in sim_parameters.items():
        if data["metadata"]["is_trending"]:
            trends_found = True
            print(f"[{cohort}] Affected by {data['metadata']['recent_sample_size']} recent events in last {RECENT_WINDOW_DAYS} days.")
            print(f"   -> Current Trust: {data['state_machine_parameters']['trust_score']}, Patience: {data['state_machine_parameters']['patience_seconds']}s")
            
    if not trends_found:
        print("No significant trends in the last 14 days. Operating on historical baseline.")

if __name__ == "__main__":
    synthesize_parameters()
