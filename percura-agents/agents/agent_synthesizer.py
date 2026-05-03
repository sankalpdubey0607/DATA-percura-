"""
agent_synthesizer.py - Percura Agent 0: Parameter Synthesizer
=============================================================
Reads extracted_behaviors.csv (real data from LLM extraction).
Computes statistical probabilities for friction types, drop-off stages, and trust.
Dynamically updates config/archetype_params.json with DATA-BACKED parameters
instead of assumptions.
"""
import os
import sys
import json
import pandas as pd
from collections import Counter
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AGENTS_DIR = os.path.join(ROOT_DIR, "percura-agents")
CONFIG_DIR = os.path.join(AGENTS_DIR, "config")
DATA_PROCESSED = os.path.join(ROOT_DIR, "data", "processed")

# Support reading either the final master file or just the reviews
MASTER_FILE = os.path.join(DATA_PROCESSED, "all_behaviors_master.csv")
REVIEWS_FILE = os.path.join(DATA_PROCESSED, "extracted_behaviors.csv")
REDDIT_FILE = os.path.join(DATA_PROCESSED, "extracted_reddit.csv")
PARAMS_FILE = os.path.join(CONFIG_DIR, "archetype_params.json")

def load_data():
    df_list = []
    if os.path.exists(MASTER_FILE):
        df_list.append(pd.read_csv(MASTER_FILE, encoding="utf-8"))
    else:
        if os.path.exists(REVIEWS_FILE):
            df_list.append(pd.read_csv(REVIEWS_FILE, encoding="utf-8"))
        if os.path.exists(REDDIT_FILE):
            df_list.append(pd.read_csv(REDDIT_FILE, encoding="utf-8"))
    
    if not df_list:
        print("ERROR: No extracted data found. Run the extraction pipeline first.")
        sys.exit(1)
        
    return pd.concat(df_list, ignore_index=True)

def map_archetype_to_data_hints(match_rules):
    """Convert Sorter demographic rules into Extractor hint filters."""
    literacy = match_rules.get("literacy", [])
    occupation = match_rules.get("occupation", [])
    device = match_rules.get("device", [])
    
    # Literacy mapping
    lit_hints = []
    if any(l in ["illiterate", "primary"] for l in literacy): lit_hints.append("low")
    if "secondary" in literacy: lit_hints.append("medium")
    if "graduate" in literacy: lit_hints.append("high")
    if not lit_hints: lit_hints = ["low", "medium", "high"]

    # Income/Occupation mapping
    inc_hints = []
    if any(o in ["no_income", "informal"] for o in occupation): inc_hints.append("low")
    if any(o in ["blue_collar", "white_collar"] for o in occupation): inc_hints.append("medium")
    if "professional" in occupation: inc_hints.append("high")
    if not inc_hints: inc_hints = ["low", "medium", "high"]

    # Device mapping
    dev_hints = []
    if "basic_android" in device: dev_hints.append("basic")
    if "mid_android" in device: dev_hints.append("mid")
    if any(d in ["high_android", "iphone"] for d in device): dev_hints.append("high")
    if not dev_hints: dev_hints = ["basic", "mid", "high"]
    
    return lit_hints, inc_hints, dev_hints

def calculate_trust_prior(df_subset, l, o):
    # Map literacy and occupation to a base trust score
    # Wider base range to ensure full 0.1-0.85 spread
    lit_scores = {"illiterate": 0.15, "primary": 0.35, "secondary": 0.55, "graduate": 0.80}
    occ_scores = {"no_income": 0.15, "informal": 0.25, "blue_collar": 0.45, "white_collar": 0.70, "professional": 0.90}
    
    # Weight literacy 40%, occupation 60% (occupation is stronger signal)
    base = (lit_scores.get(l, 0.5) * 0.4) + (occ_scores.get(o, 0.5) * 0.6)
    
    if len(df_subset) == 0: 
        return round(max(0.1, min(0.85, base)), 2)
        
    trust_signals = df_subset["trust_signal"].astype(str).str.lower()
    positive = trust_signals.isin(["trusted", "high_trust", "none"]).sum()
    negative = trust_signals.isin(["lost_trust", "never_had_trust", "scam_fear"]).sum()
    
    total = len(df_subset)
    if total > 0:
        adjustment = ((positive - negative) / total) * 0.15  # Max swing of 0.15
    else:
        adjustment = 0
        
    return round(max(0.1, min(0.85, base + adjustment)), 2)

def calculate_effort_tolerance(df_subset):
    if len(df_subset) == 0: return 5
    gave_up = df_subset["gave_up"].astype(str).str.lower() == "true"
    gave_up_rate = gave_up.mean()
    # High gave_up rate means low effort tolerance
    # Scale: 0% gave up -> 10 tolerance, 100% gave up -> 1 tolerance
    tolerance = 10 - (gave_up_rate * 9)
    return max(1, min(10, int(round(tolerance))))

def get_top_frictions(df_subset, n=4):
    if len(df_subset) == 0: return ["unknown"]
    frictions = []
    for f_str in df_subset["friction_type"].dropna():
        for f in str(f_str).split("|"):
            f = f.strip()
            if f and f not in ["nan", "unknown"]:
                frictions.append(f)
    if not frictions: return ["unknown"]
    return [f[0] for f in Counter(frictions).most_common(n)]

def get_top_drop_off(df_subset):
    """Compute primary drop-off stage per app category using the 'category' column."""
    defaults = {"fintech": "payment", "edtech": "feature_use", "ecommerce": "checkout"}
    
    # Use the 'category' column directly (more reliable than app_name matching)
    has_category = "category" in df_subset.columns if len(df_subset) > 0 else False
    
    res = {}
    for cat in ["fintech", "edtech", "ecommerce"]:
        if len(df_subset) > 0 and has_category:
            cat_subset = df_subset[df_subset["category"].astype(str).str.lower() == cat]
            stages = cat_subset["drop_off_stage"].dropna().astype(str).str.lower()
            stages = stages[~stages.isin(["nan", "unknown", ""])]
        else:
            stages = pd.Series(dtype=str)
            
        if len(stages) > 0:
            res[cat] = stages.mode()[0]
        else:
            res[cat] = defaults[cat]
            
    # Default fallback overall
    if len(df_subset) > 0:
        all_stages = df_subset["drop_off_stage"].dropna().astype(str).str.lower()
        all_stages = all_stages[~all_stages.isin(["nan", "unknown", ""])]
        res["default"] = all_stages.mode()[0] if len(all_stages) > 0 else "feature_use"
    else:
        res["default"] = "feature_use"
        
    return res

def synthesize_parameters():
    print("=" * 60)
    print("  PERCURA AGENT 0: PARAMETER SYNTHESIZER")
    print("  Linking Extracted Data -> Simulation Archetypes")
    print("=" * 60)

    df = load_data()
    print(f"  Loaded {len(df)} extracted behavioral records.")

    # Clean hints
    df["literacy_hint"] = df["literacy_hint"].astype(str).str.lower().str.strip()
    df["income_hint"] = df["income_hint"].astype(str).str.lower().str.strip()
    df["device_hint"] = df["device_hint"].astype(str).str.lower().str.strip()

    print("\n  Synthesizing data-backed parameters for all 80 archetypes...\n")

    literacies = ["illiterate", "primary", "secondary", "graduate"]
    occupations = ["no_income", "informal", "blue_collar", "white_collar", "professional"]
    devices = ["basic_android", "mid_android", "high_android", "iphone"]

    new_patterns = []

    for l in literacies:
        for o in occupations:
            for d in devices:
                match_rules = {"literacy": [l], "occupation": [o], "device": [d]}
                lit_hints, inc_hints, dev_hints = map_archetype_to_data_hints(match_rules)
                
                # Progressive fallback filtering
                mask = df["literacy_hint"].isin(lit_hints) & df["income_hint"].isin(inc_hints) & df["device_hint"].isin(dev_hints)
                subset = df[mask]
                
                if len(subset) < 5:
                    subset = df[df["income_hint"].isin(inc_hints) & df["device_hint"].isin(dev_hints)]
                if len(subset) < 5:
                    subset = df[df["income_hint"].isin(inc_hints)]
                if len(subset) < 5:
                    subset = df

                new_trust = calculate_trust_prior(subset, l, o)
                new_effort = calculate_effort_tolerance(subset)
                top_drop = get_top_drop_off(subset)
                top_frictions = get_top_frictions(subset)
                new_attention = new_effort * 10
                
                # Base tweaks based on exact demographic to prevent total homogeneity
                if l == "illiterate":
                    new_attention = max(15, new_attention - 10)
                    new_effort = max(1, new_effort - 1)
                elif l == "primary":
                    new_attention = max(20, new_attention)
                if o == "professional":
                    new_attention = min(100, new_attention + 15)
                elif o == "white_collar":
                    new_attention = min(90, new_attention + 10)
                
                # Floor at 15 so rural override (0.7x) stays above auditor threshold of 10
                new_attention = max(15, new_attention)
                
                pid = f"{l}_{o}_{d}"
                new_patterns.append({
                    "id": pid,
                    "match_rules": match_rules,
                    "params": {
                        "attention_budget": new_attention,
                        "trust_prior": new_trust,
                        "effort_tolerance": new_effort,
                        "cognitive_load_limit": max(1, min(10, int(new_effort * 0.8))),
                        "price_sensitivity": round(max(0.1, 1.0 - (new_trust * 0.5)), 2),
                        "social_proof_need": round(max(0.1, 1.0 - new_trust), 2),
                        "verification_patience": round(new_effort / 10, 2),
                        "noise_level": round(0.3 - (new_trust * 0.2), 2),
                        "primary_drop_off_stage": top_drop,
                        "top_friction_triggers": top_frictions
                    }
                })

    # Read existing config to keep overrides/default
    with open(PARAMS_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["patterns"] = new_patterns
    config["description"] = "Behavioral parameter table for Percura archetype assignment. NOTE: These parameters have been SYNTHESIZED from live behavioral data extractions covering all 80 baseline combinations."
    config["last_updated"] = datetime.now().strftime("%Y-%m-%d")

    with open(PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print(f"  SUCCESS: {PARAMS_FILE} rewritten with 80 data-backed patterns.")
    print("=" * 60)

if __name__ == "__main__":
    synthesize_parameters()
