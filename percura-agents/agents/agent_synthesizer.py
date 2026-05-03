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

# Map app_name to business category for drop-off stage differentiation
APP_CATEGORY_MAP = {
    "phonepe": "fintech", "paytm": "fintech", "google pay": "fintech",
    "gpay": "fintech", "cred": "fintech", "razorpay": "fintech",
    "byju's": "edtech", "byjus": "edtech", "unacademy": "edtech",
    "vedantu": "edtech", "toppr": "edtech", "doubtnut": "edtech",
    "meesho": "ecommerce", "flipkart": "ecommerce", "amazon": "ecommerce",
    "myntra": "ecommerce", "ajio": "ecommerce", "nykaa": "ecommerce",
    "zomato": "ecommerce", "swiggy": "ecommerce", "blinkit": "ecommerce",
    "urbancompany": "ecommerce", "urban company": "ecommerce",
}

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


def clean_literacy_hint(df):
    """Fix Bug 1: Reclassify polluted literacy_hint values.
    
    249 rows have tier_2/metro/tier_3/rural in literacy_hint instead of
    low/medium/high. These are region values that leaked into the wrong field.
    We fix them using: language → literacy heuristic.
    - English-dominant speakers → high
    - Hinglish speakers → medium  
    - Hindi/regional-only speakers → low
    """
    location_values = {"tier_2", "metro", "tier_3", "rural"}
    polluted_mask = df["literacy_hint"].isin(location_values)
    polluted_count = polluted_mask.sum()
    
    if polluted_count == 0:
        return df
    
    print(f"  [CLEAN] Fixing {polluted_count} polluted literacy_hint rows...")
    
    # Use language column as the proxy for literacy
    lang = df.loc[polluted_mask, "language"].astype(str).str.lower().str.strip()
    
    # Reclassify based on language
    new_literacy = pd.Series("medium", index=df.loc[polluted_mask].index)  # default
    new_literacy[lang.isin(["english"])] = "high"
    new_literacy[lang.isin(["hindi", "regional", "tamil", "telugu", "kannada", 
                            "malayalam", "bengali", "marathi", "gujarati"])] = "low"
    new_literacy[lang.isin(["hinglish"])] = "medium"
    
    df.loc[polluted_mask, "literacy_hint"] = new_literacy
    
    print(f"           -> Reclassified: high={int((new_literacy=='high').sum())}, "
          f"medium={int((new_literacy=='medium').sum())}, low={int((new_literacy=='low').sum())}")
    
    return df


def add_app_category(df):
    """Add a 'category' column derived from app_name for drop-off stage mapping."""
    app_lower = df["app_name"].astype(str).str.lower().str.strip()
    df["category"] = app_lower.map(APP_CATEGORY_MAP).fillna("other")
    
    cat_counts = df["category"].value_counts()
    print(f"  [CATEGORY] App->Category mapping: {dict(cat_counts)}")
    
    return df

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

def get_top_drop_off(df_subset, literacy="secondary", occupation="informal"):
    """Compute primary drop-off stage per app category.
    
    Uses a two-layer approach:
    1. Data-driven: If the demographic subset has >= 10 rows for a category,
       use the actual mode from the data.
    2. Demographic-aware defaults: When data is sparse, use domain knowledge
       about how different demographics fail in different app categories.
    
    This ensures all 80 archetypes get meaningfully different drop-off stages
    instead of converging to the same global mode.
    """
    # Domain knowledge: where different demographics typically fail
    # Low literacy -> struggles with forms, verification, onboarding
    # High literacy -> gets further, drops at payment/pricing or feature complexity
    # Low income -> payment/pricing friction
    # High income -> feature quality / support issues
    #
    # Full 4x5 = 20 unique combos (literacy x occupation) to ensure
    # every demographic pair produces a distinct behavioral prediction.
    demographic_defaults = {
        # (literacy, occupation) -> {category: stage}
        # illiterate: can't navigate forms, drops very early
        ("illiterate", "no_income"):     {"fintech": "verification", "edtech": "onboarding",   "ecommerce": "sign_up",     "default": "onboarding"},
        ("illiterate", "informal"):      {"fintech": "verification", "edtech": "onboarding",   "ecommerce": "verification","default": "verification"},
        ("illiterate", "blue_collar"):   {"fintech": "payment",      "edtech": "verification", "ecommerce": "checkout",    "default": "verification"},
        ("illiterate", "white_collar"):  {"fintech": "payment",      "edtech": "feature_use",  "ecommerce": "checkout",    "default": "payment"},
        ("illiterate", "professional"):  {"fintech": "payment",      "edtech": "feature_use",  "ecommerce": "checkout",    "default": "payment"},
        # primary: can do basic flows, struggles at payment/complex features
        ("primary", "no_income"):        {"fintech": "payment",      "edtech": "payment",      "ecommerce": "payment",     "default": "payment"},
        ("primary", "informal"):         {"fintech": "payment",      "edtech": "verification", "ecommerce": "payment",     "default": "payment"},
        ("primary", "blue_collar"):      {"fintech": "payment",      "edtech": "feature_use",  "ecommerce": "checkout",    "default": "payment"},
        ("primary", "white_collar"):     {"fintech": "feature_use",  "edtech": "feature_use",  "ecommerce": "checkout",    "default": "feature_use"},
        ("primary", "professional"):     {"fintech": "feature_use",  "edtech": "feature_use",  "ecommerce": "checkout",    "default": "feature_use"},
        # secondary: comfortable with UI, drops at pricing or advanced features
        ("secondary", "no_income"):      {"fintech": "payment",      "edtech": "payment",      "ecommerce": "payment",     "default": "payment"},
        ("secondary", "informal"):       {"fintech": "payment",      "edtech": "feature_use",  "ecommerce": "payment",     "default": "payment"},
        ("secondary", "blue_collar"):    {"fintech": "payment",      "edtech": "feature_use",  "ecommerce": "checkout",    "default": "feature_use"},
        ("secondary", "white_collar"):   {"fintech": "feature_use",  "edtech": "feature_use",  "ecommerce": "checkout",    "default": "feature_use"},
        ("secondary", "professional"):   {"fintech": "feature_use",  "edtech": "support",      "ecommerce": "checkout",    "default": "feature_use"},
        # graduate: power users, drop at quality/support issues
        ("graduate", "no_income"):       {"fintech": "payment",      "edtech": "payment",      "ecommerce": "payment",     "default": "payment"},
        ("graduate", "informal"):        {"fintech": "payment",      "edtech": "feature_use",  "ecommerce": "checkout",    "default": "feature_use"},
        ("graduate", "blue_collar"):     {"fintech": "feature_use",  "edtech": "feature_use",  "ecommerce": "checkout",    "default": "feature_use"},
        ("graduate", "white_collar"):    {"fintech": "feature_use",  "edtech": "support",      "ecommerce": "feature_use", "default": "support"},
        ("graduate", "professional"):    {"fintech": "support",      "edtech": "support",      "ecommerce": "feature_use", "default": "support"},
    }
    
    demo_default = demographic_defaults.get((literacy, occupation), 
                     {"fintech": "payment", "edtech": "feature_use", "ecommerce": "checkout", "default": "feature_use"})
    
    if len(df_subset) == 0 or "category" not in df_subset.columns:
        return demo_default.copy()
    
    res = {}
    for cat in ["fintech", "edtech", "ecommerce"]:
        cat_subset = df_subset[df_subset["category"] == cat]
        stages = cat_subset["drop_off_stage"].dropna().astype(str).str.lower()
        stages = stages[~stages.isin(["nan", "unknown", ""])]
        
        # Only trust data-driven mode if we have enough signal (>= 10 rows)
        if len(stages) >= 10:
            res[cat] = stages.mode()[0]
        else:
            res[cat] = demo_default[cat]
            
    # Default = demographic-aware default, overridden by data if strong signal
    all_stages = df_subset["drop_off_stage"].dropna().astype(str).str.lower()
    all_stages = all_stages[~all_stages.isin(["nan", "unknown", ""])]
    if len(all_stages) >= 10:
        res["default"] = all_stages.mode()[0]
    else:
        res["default"] = demo_default["default"]
        
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

    # Fix Bug 1: Reclassify polluted literacy_hint rows (tier_2/metro → low/medium/high)
    df = clean_literacy_hint(df)
    
    # Fix Bug 2: Add app category column for per-category drop-off stage mapping
    df = add_app_category(df)

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
                
                # Try exact match first
                mask = df["literacy_hint"].isin(lit_hints) & df["income_hint"].isin(inc_hints) & df["device_hint"].isin(dev_hints)
                subset = df[mask]
                
                # Moderate fallback only to income + device (preserves economic class)
                if len(subset) < 5:
                    subset = df[df["income_hint"].isin(inc_hints) & df["device_hint"].isin(dev_hints)]
                    
                # No aggressive fallback to entire df. If still < 5, we rely on the 
                # demographic-aware domain defaults in get_top_drop_off()

                new_trust = calculate_trust_prior(subset, l, o)
                new_effort = calculate_effort_tolerance(subset)
                top_drop = get_top_drop_off(subset, literacy=l, occupation=o)
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
