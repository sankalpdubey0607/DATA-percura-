"""
agent_sorter.py - Percura Agent 1: Sorting Agent
=================================================
Reads the Nvidia Nemotron-Personas-India dataset (en_IN split).
Assigns every persona to exactly one archetype based on:
  age, education_level, occupation, district/state
Outputs: nvidia_personas_sorted.csv + sorter_report.txt

Performance: Chunked processing (50k rows), tqdm progress bars,
             checkpointing every 500k rows, <4GB RAM.
"""
import os
import sys
import json
import time
import re
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from collections import Counter

# ── Configuration ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
DATA_REPORTS = os.path.join(BASE_DIR, "data", "reports")

PARAMS_FILE = os.path.join(CONFIG_DIR, "archetype_params.json")
OUTPUT_FILE = os.path.join(DATA_PROCESSED, "nvidia_personas_sorted.csv")
REPORT_FILE = os.path.join(DATA_REPORTS, "sorter_report.txt")
CHECKPOINT_FILE = os.path.join(DATA_PROCESSED, "sorter_checkpoint.json")

CHUNK_SIZE = 50_000
CHECKPOINT_INTERVAL = 500_000
DATASET_NAME = "nvidia/Nemotron-Personas-India"
DATASET_SPLIT = "en_IN"

# ── Metro / Tier-2 district lists ────────────────────────────
# Districts that contain or correspond to major metro cities
METRO_DISTRICTS = [
    "mumbai", "delhi", "bangalore", "bengaluru", "hyderabad",
    "chennai", "kolkata", "pune", "ahmedabad",
    "new delhi", "north delhi", "south delhi", "east delhi",
    "west delhi", "central delhi", "north west delhi",
    "south west delhi", "north east delhi", "south east delhi",
    "shahdara", "thane", "mumbai suburban",
    "bangalore urban", "hyderabad", "rangareddy",
    "chennai", "kolkata", "north 24 parganas",
    "south 24 parganas", "pune", "ahmedabad",
]

TIER_2_DISTRICTS = [
    "jaipur", "lucknow", "kanpur", "nagpur", "indore",
    "bhopal", "patna", "vadodara", "surat", "agra",
    "visakhapatnam", "coimbatore", "kochi", "ernakulam",
    "thiruvananthapuram", "guwahati", "kamrup metropolitan",
    "ludhiana", "chandigarh", "dehradun", "ranchi",
    "bhubaneswar", "khordha", "mysore", "mysuru",
    "jodhpur", "gwalior", "varanasi", "allahabad",
    "prayagraj", "meerut", "nashik", "rajkot",
    "madurai", "tiruchirappalli", "vijayawada",
    "krishna", "guntur", "warangal", "aurangabad",
    "chhatrapati sambhajinagar", "amritsar", "jalandhar",
    "kozhikode", "thrissur", "raipur", "gurgaon",
    "gurugram", "noida", "gautam buddha nagar",
    "faridabad", "ghaziabad",
]

# ── Education level mapping ──────────────────────────────────
# The Nvidia dataset has an education_level field with values like:
# "Illiterate", "Literate", "Below Primary", "Primary",
# "Middle", "Matric/Secondary", "Higher Secondary",
# "Non-Technical Diploma", "Technical Diploma",
# "Graduate", "Post Graduate"

EDUCATION_MAP = {
    # illiterate
    "illiterate": "illiterate",

    # primary
    "literate": "primary",
    "below primary": "primary",
    "primary": "primary",

    # secondary
    "middle": "secondary",
    "matric/secondary": "secondary",
    "matric": "secondary",
    "secondary": "secondary",
    "higher secondary": "secondary",
    "non-technical diploma": "secondary",
    "technical diploma": "secondary",

    # graduate
    "graduate": "graduate",
    "post graduate": "graduate",
}

# ── Occupation mapping ───────────────────────────────────────
# The Nvidia dataset has ~3000 occupation titles from NCO-2004.
# We map them to 5 categories using keyword matching.

NO_INCOME_KEYWORDS = [
    "student", "unemployed", "homemaker", "housewife",
    "retired", "no occupation", "non-worker", "house wife",
    "home maker", "dependent", "pensioner",
]

INFORMAL_KEYWORDS = [
    "daily wage", "labourer", "laborer", "street vendor",
    "hawker", "rickshaw", "domestic worker", "maid",
    "sweeper", "rag picker", "waste collector", "coolie",
    "porter", "helper", "peon", "attendant", "watchman",
    "small farmer", "marginal farmer", "agricultural labourer",
    "agricultural laborer", "farm labourer", "farm laborer",
    "toddy tapper", "fisherman", "fisherwoman",
    "bidi worker", "bidi maker", "construction worker",
    "stone cutter", "wood cutter", "charcoal maker",
    "washer", "washerman", "washerwoman", "dhobi",
    "cobbler", "barber", "nai", "potter", "blacksmith",
    "weaver", "tailor", "seamstress", "basket maker",
    "rope maker", "mat maker", "broom maker",
    "tea stall", "chai", "dhaba", "roadside",
    "hand cart", "cycle rickshaw", "auto rickshaw",
]

BLUE_COLLAR_KEYWORDS = [
    "driver", "mechanic", "electrician", "plumber",
    "carpenter", "mason", "painter", "welder",
    "fitter", "turner", "machinist", "operator",
    "factory worker", "assembly", "manufacturing",
    "technician", "lineman", "wireman",
    "security guard", "guard", "chowkidar",
    "delivery", "courier", "postman",
    "conductor", "cleaner", "sweeper",
    "gardener", "mali", "cook", "chef",
    "baker", "butcher", "halwai",
    "goldsmith", "silversmith", "jeweller",
    "printer", "compositor", "bookbinder",
    "shoe maker", "leather worker",
    "tractor driver", "crane operator",
    "forklift", "mining", "quarry",
]

PROFESSIONAL_KEYWORDS = [
    "doctor", "physician", "surgeon", "dentist",
    "lawyer", "advocate", "judge", "magistrate",
    "engineer", "software", "developer", "programmer",
    "architect", "chartered accountant", "ca ",
    "company secretary", "cs ", "cost accountant",
    "professor", "scientist", "researcher",
    "pilot", "naval officer", "army officer",
    "ias ", "ips ", "ifs ", "civil servant",
    "consultant", "analyst", "manager", "director",
    "ceo", "cfo", "cto", "vice president",
    "chief", "head of", "sr. manager", "senior manager",
    "general manager", "executive director",
    "investment banker", "financial analyst",
    "data scientist", "machine learning",
    "artificial intelligence", "blockchain",
    "product manager", "project manager",
    "marketing manager", "sales manager",
    "human resource manager", "hr manager",
]

WHITE_COLLAR_KEYWORDS = [
    "teacher", "lecturer", "instructor", "tutor",
    "nurse", "pharmacist", "lab technician",
    "clerk", "typist", "stenographer", "data entry",
    "accountant", "bookkeeper", "cashier",
    "bank", "insurance", "government",
    "office", "administrative", "secretary",
    "receptionist", "telephone operator",
    "sales", "marketing", "advertising",
    "journalist", "reporter", "editor",
    "librarian", "archivist", "curator",
    "social worker", "counsellor", "counselor",
    "police", "constable", "inspector",
    "fireman", "firefighter",
    "railway", "postal",
    "supervisor", "foreman", "overseer",
    "shop", "store", "retail", "merchant",
    "trader", "dealer", "broker", "agent",
    "contractor", "sub-contractor",
    "photographer", "videographer", "cameraperson",
]


# ══════════════════════════════════════════════════════════════
# MAPPING FUNCTIONS
# ══════════════════════════════════════════════════════════════

def map_education(edu_level):
    """Map education_level string to one of: illiterate, primary, secondary, graduate"""
    if not edu_level or pd.isna(edu_level):
        return "secondary"  # safe default
    edu_lower = str(edu_level).strip().lower()
    if edu_lower in EDUCATION_MAP:
        return EDUCATION_MAP[edu_lower]
    # Fuzzy fallback
    if any(k in edu_lower for k in ["illiterate", "anpadh", "no school"]):
        return "illiterate"
    if any(k in edu_lower for k in ["primary", "elementary", "class 1", "class 2", "class 3", "class 4", "class 5"]):
        return "primary"
    if any(k in edu_lower for k in ["graduate", "bachelor", "master", "phd", "degree", "engineer", "mba", "college"]):
        return "graduate"
    return "secondary"


def map_occupation(occupation_str):
    """Map occupation string to one of: no_income, informal, blue_collar, white_collar, professional"""
    if not occupation_str or pd.isna(occupation_str):
        return "no_income"
    occ_lower = str(occupation_str).strip().lower()

    # Check in priority order (most specific first)
    for kw in NO_INCOME_KEYWORDS:
        if kw in occ_lower:
            return "no_income"
    for kw in PROFESSIONAL_KEYWORDS:
        if kw in occ_lower:
            return "professional"
    for kw in INFORMAL_KEYWORDS:
        if kw in occ_lower:
            return "informal"
    for kw in BLUE_COLLAR_KEYWORDS:
        if kw in occ_lower:
            return "blue_collar"
    for kw in WHITE_COLLAR_KEYWORDS:
        if kw in occ_lower:
            return "white_collar"

    # Fallback: use education to guess
    return "blue_collar"


def map_device(occupation_mapped):
    """Infer device type from occupation category (no device field in dataset)."""
    device_map = {
        "no_income": "basic_android",
        "informal": "basic_android",
        "blue_collar": "mid_android",
        "white_collar": "high_android",
        "professional": "iphone",
    }
    return device_map.get(occupation_mapped, "mid_android")


def map_region(district, state):
    """Map district+state to one of: metro, tier_2, tier_3, rural"""
    if not district or pd.isna(district):
        if not state or pd.isna(state):
            return "tier_3"
        district = ""
    dist_lower = str(district).strip().lower()
    state_lower = str(state).strip().lower() if state and not pd.isna(state) else ""

    # Check metro
    for metro in METRO_DISTRICTS:
        if metro in dist_lower or dist_lower in metro:
            return "metro"

    # Check tier_2
    for t2 in TIER_2_DISTRICTS:
        if t2 in dist_lower or dist_lower in t2:
            return "tier_2"

    # Heuristic: if district name suggests rural area
    rural_hints = ["rural", "gram", "panchayat", "taluk", "tehsil", "block"]
    if any(hint in dist_lower for hint in rural_hints):
        return "rural"

    # Default: tier_3 for unrecognized districts
    return "tier_3"


def map_age_band(age):
    """Map integer age to age band."""
    try:
        age = int(age)
    except (ValueError, TypeError):
        return "young"
    if age <= 25:
        return "youth"
    elif age <= 35:
        return "young"
    elif age <= 50:
        return "middle"
    else:
        return "senior"


def extract_name_from_persona(persona_text):
    """Extract the first name from the persona text field."""
    if not persona_text or pd.isna(persona_text):
        return ""
    text = str(persona_text).strip()
    # Common patterns: "Ramesh is a...", "Meet Ramesh, a...", "Ramesh, a 42-year-old..."
    match = re.match(r'^(?:Meet\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', text)
    if match:
        return match.group(1).strip()
    # Try first word if capitalized
    words = text.split()
    if words and words[0][0].isupper():
        return words[0].strip(",").strip(".")
    return ""


# ══════════════════════════════════════════════════════════════
# BEHAVIORAL PARAMETER ASSIGNMENT
# ══════════════════════════════════════════════════════════════

def load_archetype_params():
    """Load behavioral parameters from config JSON."""
    try:
        with open(PARAMS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  ERROR: {PARAMS_FILE} not found!")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"  ERROR: Invalid JSON in {PARAMS_FILE}: {e}")
        sys.exit(1)


def find_matching_pattern(literacy, occupation, device, config):
    """Find the first matching pattern from config, or return default."""
    for pattern in config["patterns"]:
        rules = pattern["match_rules"]
        if (literacy in rules.get("literacy", []) and
            occupation in rules.get("occupation", []) and
            device in rules.get("device", [])):
            return pattern["params"], pattern["id"], False
    return config["default"], "default", True


def apply_overrides(params, region, config):
    """Apply override rules (e.g., rural penalty)."""
    params = dict(params)  # copy
    for override in config.get("overrides", []):
        cond = override.get("condition", {})
        if cond.get("region") == region:
            apply = override["apply"]
            if "attention_budget_multiplier" in apply:
                params["attention_budget"] = int(
                    params["attention_budget"] * apply["attention_budget_multiplier"]
                )
            if "trust_prior_offset" in apply:
                params["trust_prior"] = round(
                    max(0, params["trust_prior"] + apply["trust_prior_offset"]), 2
                )
            if "add_friction_trigger" in apply:
                triggers = list(params.get("top_friction_triggers", []))
                new_trigger = apply["add_friction_trigger"]
                if new_trigger not in triggers:
                    triggers.append(new_trigger)
                params["top_friction_triggers"] = triggers
    return params


# ══════════════════════════════════════════════════════════════
# CHECKPOINT SYSTEM
# ══════════════════════════════════════════════════════════════

def load_checkpoint():
    """Load checkpoint if exists."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_checkpoint(rows_processed, chunk_idx):
    """Save progress checkpoint."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "rows_processed": rows_processed,
            "chunk_idx": chunk_idx,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)


# ══════════════════════════════════════════════════════════════
# MAIN PROCESSING
# ══════════════════════════════════════════════════════════════

def process_chunk(df, config):
    """Process a single chunk of data, returning enriched DataFrame."""
    # Map fields
    df["age_band"] = df["age"].apply(map_age_band)
    df["literacy_mapped"] = df["education_level"].apply(map_education)
    df["occupation_mapped"] = df["occupation"].apply(map_occupation)
    df["device_mapped"] = df["occupation_mapped"].apply(map_device)
    df["region_mapped"] = df.apply(
        lambda r: map_region(r.get("district", ""), r.get("state", "")), axis=1
    )
    df["extracted_name"] = df["persona"].apply(extract_name_from_persona)

    # Build archetype string
    df["archetype"] = (
        df["age_band"] + "-" +
        df["region_mapped"] + "-" +
        df["literacy_mapped"] + "-" +
        df["occupation_mapped"] + "-" +
        df["device_mapped"]
    )

    # Assign behavioral parameters
    param_columns = [
        "attention_budget", "trust_prior", "effort_tolerance",
        "cognitive_load_limit", "price_sensitivity", "social_proof_need",
        "verification_patience", "noise_level",
        "primary_drop_off_stage", "top_friction_triggers",
    ]
    used_default_flags = []

    rows_data = []
    for _, row in df.iterrows():
        params, pattern_id, is_default = find_matching_pattern(
            row["literacy_mapped"], row["occupation_mapped"],
            row["device_mapped"], config
        )
        # Apply region overrides
        params = apply_overrides(params, row["region_mapped"], config)
        used_default_flags.append(is_default)

        row_params = {}
        for col in param_columns:
            val = params.get(col, "")
            if isinstance(val, list):
                val = "|".join(val)
            row_params[col] = val
        row_params["used_default"] = is_default
        rows_data.append(row_params)

    params_df = pd.DataFrame(rows_data, index=df.index)
    df = pd.concat([df, params_df], axis=1)

    # Add metadata columns
    df["parameter_version"] = config.get("version", "1.0")
    df["date_sorted"] = datetime.now().strftime("%Y-%m-%d")

    return df


def run_sorter():
    """Main entry point for Agent 1."""
    t0 = time.time()

    # Create output dirs
    for d in [DATA_PROCESSED, DATA_REPORTS]:
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("  PERCURA AGENT 1: SORTING AGENT")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load config
    print("\n  [1/5] Loading archetype parameters...")
    config = load_archetype_params()
    print(f"         Loaded {len(config['patterns'])} patterns + "
          f"{len(config.get('overrides', []))} overrides")

    # Load dataset
    print(f"\n  [2/5] Loading Nvidia dataset ({DATASET_SPLIT})...")
    print(f"         This downloads ~5GB on first run. Please be patient.")

    try:
        from datasets import load_dataset
        ds = load_dataset(DATASET_NAME, "default", split=DATASET_SPLIT)
        total_rows = len(ds)
        print(f"         Loaded {total_rows:,} personas")
    except Exception as e:
        print(f"  ERROR loading dataset: {e}")
        print(f"  Make sure 'datasets' and 'pyarrow' are installed.")
        sys.exit(1)

    # Print column names for verification
    print(f"\n  [3/5] Dataset columns ({len(ds.column_names)}):")
    for col in ds.column_names:
        print(f"         - {col}")

    # Check for resume
    checkpoint = load_checkpoint()
    start_chunk = 0
    write_mode = "w"
    write_header = True
    if checkpoint and os.path.exists(OUTPUT_FILE):
        start_chunk = checkpoint.get("chunk_idx", 0) + 1
        start_row = start_chunk * CHUNK_SIZE
        if start_row < total_rows:
            print(f"\n  [RESUME] From chunk {start_chunk} "
                  f"(row {start_row:,})")
            write_mode = "a"
            write_header = False
        else:
            print(f"\n  [RESUME] All chunks already processed!")
            start_chunk = 0  # reprocess

    # Process in chunks
    print(f"\n  [4/5] Processing {total_rows:,} personas in chunks of "
          f"{CHUNK_SIZE:,}...")
    n_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
    stats = Counter()
    archetype_counter = Counter()
    default_count = 0
    total_processed = start_chunk * CHUNK_SIZE

    first_write = (write_mode == "w")

    for chunk_idx in tqdm(range(start_chunk, n_chunks), desc="  Processing",
                          unit="chunk"):
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, total_rows)

        try:
            # Get chunk from dataset
            chunk_ds = ds.select(range(start, end))
            df = chunk_ds.to_pandas()

            # Process
            df = process_chunk(df, config)

            # Count stats
            archetype_counter.update(df["archetype"].value_counts().to_dict())
            stats["literacy_illiterate"] += (df["literacy_mapped"] == "illiterate").sum()
            stats["literacy_primary"] += (df["literacy_mapped"] == "primary").sum()
            stats["literacy_secondary"] += (df["literacy_mapped"] == "secondary").sum()
            stats["literacy_graduate"] += (df["literacy_mapped"] == "graduate").sum()
            stats["occ_no_income"] += (df["occupation_mapped"] == "no_income").sum()
            stats["occ_informal"] += (df["occupation_mapped"] == "informal").sum()
            stats["occ_blue_collar"] += (df["occupation_mapped"] == "blue_collar").sum()
            stats["occ_white_collar"] += (df["occupation_mapped"] == "white_collar").sum()
            stats["occ_professional"] += (df["occupation_mapped"] == "professional").sum()
            stats["region_metro"] += (df["region_mapped"] == "metro").sum()
            stats["region_tier_2"] += (df["region_mapped"] == "tier_2").sum()
            stats["region_tier_3"] += (df["region_mapped"] == "tier_3").sum()
            stats["region_rural"] += (df["region_mapped"] == "rural").sum()
            default_count += df["used_default"].sum()
            total_processed += len(df)

            # Write to CSV
            df.to_csv(OUTPUT_FILE, mode="a" if not first_write else "w",
                       header=first_write, index=False, encoding="utf-8")
            first_write = False

            # Checkpoint
            if total_processed % CHECKPOINT_INTERVAL < CHUNK_SIZE:
                save_checkpoint(total_processed, chunk_idx)

        except Exception as e:
            print(f"\n  ERROR in chunk {chunk_idx}: {e}")
            save_checkpoint(total_processed, chunk_idx - 1)
            raise

    # Final checkpoint
    save_checkpoint(total_processed, n_chunks - 1)

    elapsed = time.time() - t0

    # ── Step 6: Generate report ──────────────────────────────
    print(f"\n  [5/5] Generating report...")

    n_archetypes = len(archetype_counter)
    default_pct = default_count / total_processed * 100 if total_processed else 0

    report_lines = [
        "=" * 60,
        "  PERCURA AGENT 1: SORTER REPORT",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Processing time: {elapsed:.1f}s ({elapsed/60:.1f} min)",
        "=" * 60,
        "",
        f"  Total personas processed:   {total_processed:,}",
        f"  Total unique archetypes:    {n_archetypes}",
        f"  Used DEFAULT parameters:    {int(default_count):,} ({default_pct:.1f}%)",
        "",
        "--- REGION DISTRIBUTION ---",
        f"  metro:    {stats['region_metro']:>8,}",
        f"  tier_2:   {stats['region_tier_2']:>8,}",
        f"  tier_3:   {stats['region_tier_3']:>8,}",
        f"  rural:    {stats['region_rural']:>8,}",
        "",
        "--- LITERACY DISTRIBUTION ---",
        f"  illiterate:  {stats['literacy_illiterate']:>8,}",
        f"  primary:     {stats['literacy_primary']:>8,}",
        f"  secondary:   {stats['literacy_secondary']:>8,}",
        f"  graduate:    {stats['literacy_graduate']:>8,}",
        "",
        "--- OCCUPATION DISTRIBUTION ---",
        f"  no_income:     {stats['occ_no_income']:>8,}",
        f"  informal:      {stats['occ_informal']:>8,}",
        f"  blue_collar:   {stats['occ_blue_collar']:>8,}",
        f"  white_collar:  {stats['occ_white_collar']:>8,}",
        f"  professional:  {stats['occ_professional']:>8,}",
        "",
        "--- TOP 20 ARCHETYPES (by count) ---",
    ]
    for archetype, count in archetype_counter.most_common(20):
        pct = count / total_processed * 100
        report_lines.append(f"  {archetype:55s} {count:>8,}  ({pct:.1f}%)")

    report_lines += [
        "",
        "--- BOTTOM 10 ARCHETYPES (smallest) ---",
    ]
    for archetype, count in archetype_counter.most_common()[:-11:-1]:
        report_lines.append(f"  {archetype:55s} {count:>8,}")

    if default_pct > 5:
        report_lines += [
            "",
            "--- WARNING ---",
            f"  {default_pct:.1f}% of personas used DEFAULT parameters!",
            "  This exceeds the 5% threshold. Review occupation mapping.",
        ]

    report_lines += [
        "",
        "=" * 60,
        f"  Output: {OUTPUT_FILE}",
        f"  Report: {REPORT_FILE}",
        "=" * 60,
    ]

    report_text = "\n".join(report_lines)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\n  Done in {elapsed:.1f}s")

    return {
        "total_processed": total_processed,
        "n_archetypes": n_archetypes,
        "default_pct": default_pct,
        "elapsed": elapsed,
    }


if __name__ == "__main__":
    run_sorter()
