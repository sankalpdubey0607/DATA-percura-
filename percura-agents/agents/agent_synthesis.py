"""
agent_synthesis.py - Percura Synthesis Agent (Phase 2)
======================================================
Reads extracted behavioral data from app reviews and mathematically 
synthesizes the starting parameters (Patience, Trust, Effort) for the 
simulation state machine.

It groups extracted behaviors into demographic cohorts based on:
Region, Literacy, Income, and Device.
"""
import os
import csv
import json
from collections import defaultdict

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
INPUT_CSV = os.path.join(DATA_DIR, "extracted_behaviors_1209.csv")
OUTPUT_JSON = os.path.join(DATA_DIR, "cohort_parameters.json")

def synthesize_parameters():
    print("=" * 60)
    print("  PERCURA SYNTHESIS AGENT (PARAMETER CALIBRATION)")
    print("=" * 60)

    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: Cannot find {INPUT_CSV}")
        return

    # Data structures for aggregation
    cohorts = defaultdict(lambda: {
        "count": 0,
        "gave_up_count": 0,
        "trust_lost_count": 0,
        "confused_count": 0,
        "effort_complained_count": 0
    })

    # Read data
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Build cohort key (demographic bucket)
            region = row.get("region_hint", "unknown")
            literacy = row.get("literacy_hint", "unknown")
            income = row.get("income_hint", "unknown")
            device = row.get("device_hint", "unknown")
            
            # Skip if fully unknown (though our new prompt fixes this)
            if device == "unknown" and income == "unknown":
                continue

            cohort_key = f"{region}-{literacy}-{income}-{device}"
            
            # Update metrics
            stats = cohorts[cohort_key]
            stats["count"] += 1
            
            # Gave up?
            if str(row.get("gave_up", "")).lower() == "true":
                stats["gave_up_count"] += 1
                
            # Trust issues?
            trust = str(row.get("trust_signal", "")).lower()
            if trust in ["lost_trust", "never_had_trust", "did_not_trust"]:
                stats["trust_lost_count"] += 1
                
            # Confusion?
            if str(row.get("emotion", "")).lower() == "confused" or "confusing_language" in str(row.get("friction_type", "")):
                stats["confused_count"] += 1
                
            # Effort?
            if str(row.get("effort_complained", "")).lower() == "true" or "too_many_steps" in str(row.get("friction_type", "")):
                stats["effort_complained_count"] += 1

    # Filter for cohorts with statistical significance (e.g. > 10 complaints)
    valid_cohorts = {k: v for k, v in cohorts.items() if v["count"] >= 10}
    
    print(f"Total unique cohorts found: {len(cohorts)}")
    print(f"Cohorts with significant data (>10 rows): {len(valid_cohorts)}")
    
    # Calculate Simulation Parameters mathematically
    sim_parameters = {}
    
    for cohort, stats in valid_cohorts.items():
        total = stats["count"]
        
        pct_gave_up = stats["gave_up_count"] / total
        pct_trust_issues = stats["trust_lost_count"] / total
        pct_confused = stats["confused_count"] / total
        pct_effort = stats["effort_complained_count"] / total
        
        # --- PARAMETER HEURISTICS ---
        # 1. Trust Score (0.0 to 1.0)
        # Base trust is 0.8. Drops based on how many people lost trust.
        base_trust = 0.8
        calculated_trust = max(0.1, base_trust - (pct_trust_issues * 1.5))
        
        # 2. Patience Level (Seconds: 15s to 90s)
        # Base patience is 60s. High dropoff/effort complaint lowers it drastically.
        base_patience = 60
        patience_penalty = (pct_gave_up * 30) + (pct_effort * 20)
        calculated_patience = max(15, int(base_patience - patience_penalty))
        
        # 3. Confusion Threshold (0.0 to 1.0)
        # How easily they get confused. 0.0 means gets confused instantly. 
        # Base is 0.7. Drops if history of confusion.
        base_confusion = 0.7
        calculated_confusion = max(0.1, base_confusion - (pct_confused * 2.0))
        
        # 4. Effort Tolerance (Steps: 1 to 7)
        # Base is 5 steps. Drops if they complained about effort.
        base_effort_steps = 5
        calculated_effort_steps = max(1, int(base_effort_steps - (pct_effort * 6)))

        sim_parameters[cohort] = {
            "metadata": {
                "sample_size": total,
                "pct_gave_up": round(pct_gave_up, 2),
                "pct_trust_issues": round(pct_trust_issues, 2)
            },
            "state_machine_parameters": {
                "patience_seconds": calculated_patience,
                "trust_score": round(calculated_trust, 2),
                "effort_tolerance_steps": calculated_effort_steps,
                "confusion_threshold": round(calculated_confusion, 2),
                "noise_vulnerability": 0.2 # Default for now
            }
        }
    
    # Save to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(sim_parameters, f, indent=4)
        
    print(f"\nSuccessfully synthesized parameters for {len(sim_parameters)} archetypes.")
    print(f"Saved to: {OUTPUT_JSON}")
    
    # Print a few examples
    print("\n--- SAMPLE COHORTS ---")
    for i, (cohort, data) in enumerate(list(sim_parameters.items())[:3]):
        params = data["state_machine_parameters"]
        print(f"\nCohort: {cohort} (N={data['metadata']['sample_size']})")
        print(f"  Patience:    {params['patience_seconds']}s")
        print(f"  Trust Score: {params['trust_score']}")
        print(f"  Effort Tol:  {params['effort_tolerance_steps']} steps")

if __name__ == "__main__":
    synthesize_parameters()
