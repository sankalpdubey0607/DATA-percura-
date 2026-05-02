"""
agent_auditor.py - Percura Agent 2: Audit, Verification & Removal Agent
========================================================================
Reads nvidia_personas_sorted.csv (output of Agent 1).
Runs 6 audit checks on every row.
Outputs: nvidia_personas_clean.csv, nvidia_personas_removed.csv,
         nvidia_personas_flagged.csv, audit_report.txt

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
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
DATA_REPORTS = os.path.join(BASE_DIR, "data", "reports")

INPUT_FILE = os.path.join(DATA_PROCESSED, "nvidia_personas_sorted.csv")
CLEAN_FILE = os.path.join(DATA_PROCESSED, "nvidia_personas_clean.csv")
REMOVED_FILE = os.path.join(DATA_PROCESSED, "nvidia_personas_removed.csv")
FLAGGED_FILE = os.path.join(DATA_PROCESSED, "nvidia_personas_flagged.csv")
REPORT_FILE = os.path.join(DATA_REPORTS, "audit_report.txt")
CHECKPOINT_FILE = os.path.join(DATA_PROCESSED, "auditor_checkpoint.json")

CHUNK_SIZE = 50_000
CHECKPOINT_INTERVAL = 500_000

# Valid archetype format: band-region-literacy-occupation-device
ARCHETYPE_PATTERN = re.compile(
    r"^(youth|young|middle|senior)-"
    r"(metro|tier_2|tier_3|rural)-"
    r"(illiterate|primary|secondary|graduate)-"
    r"(no_income|informal|blue_collar|white_collar|professional)-"
    r"(basic_android|mid_android|high_android|iphone)$"
)


# ══════════════════════════════════════════════════════════════
# AUDIT CHECK FUNCTIONS
# ══════════════════════════════════════════════════════════════

def check_1_completeness(row):
    """
    AUDIT CHECK 1: Completeness Check
    Returns (passed: bool, reason: str or None)
    """
    # Check key fields
    for field in ["education_level", "occupation", "district", "age"]:
        val = row.get(field)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return False, "incomplete_key_fields"
        if isinstance(val, str) and val.strip() == "":
            return False, "incomplete_key_fields"

    # Age range check
    try:
        age = int(row.get("age", 0))
        if age < 18 or age > 90:
            return False, "age_out_of_range"
    except (ValueError, TypeError):
        return False, "invalid_age"

    # Name check (extracted_name from sorter)
    name = row.get("extracted_name", "")
    if isinstance(name, float) and pd.isna(name):
        name = ""
    name = str(name).strip().lower()
    if name in ["", "unknown", "test", "null", "none", "nan"]:
        # Check persona field as fallback
        persona = str(row.get("persona", ""))
        if len(persona.strip()) < 10:
            return False, "missing_name"

    # Location too vague
    district = str(row.get("district", "")).strip().lower()
    state = str(row.get("state", "")).strip().lower()
    if district in ["india", "", "unknown"] and state in ["", "unknown", "india"]:
        return False, "location_too_vague"

    return True, None


def check_2_duplicate(df):
    """
    AUDIT CHECK 2: Duplicate Check
    Returns DataFrame with added columns: is_exact_duplicate, is_near_duplicate
    Must operate on full chunk at once.
    """
    # Build dedup key: extracted_name + age + district + occupation
    df["_dedup_key"] = (
        df["extracted_name"].fillna("").astype(str).str.lower().str.strip() + "|" +
        df["age"].astype(str) + "|" +
        df["district"].fillna("").astype(str).str.lower().str.strip() + "|" +
        df["occupation"].fillna("").astype(str).str.lower().str.strip()
    )

    # Near-duplicate key: name + age + district (occupation may differ)
    df["_near_key"] = (
        df["extracted_name"].fillna("").astype(str).str.lower().str.strip() + "|" +
        df["age"].astype(str) + "|" +
        df["district"].fillna("").astype(str).str.lower().str.strip()
    )

    # Mark exact duplicates (keep first)
    df["is_exact_duplicate"] = df.duplicated(subset=["_dedup_key"], keep="first")

    # Mark near duplicates (same name+age+district but different occupation)
    near_dup_counts = df.groupby("_near_key")["occupation"].transform("nunique")
    df["is_near_duplicate"] = (near_dup_counts > 1) & (~df["is_exact_duplicate"])

    # Clean up temp columns
    df.drop(columns=["_dedup_key", "_near_key"], inplace=True)

    return df


def check_3_behavioral_usefulness(row):
    """
    AUDIT CHECK 3: Behavioral Usefulness Check
    Returns (passed: bool, reason: str or None)
    """
    try:
        trust = float(row.get("trust_prior", 0.5))
        attention = float(row.get("attention_budget", 45))
        effort = float(row.get("effort_tolerance", 4))
        noise = float(row.get("noise_level", 0.15))
    except (ValueError, TypeError):
        return True, None  # Can't check, let it pass

    if trust == 0:
        return False, "trust_prior_zero"
    if trust == 1:
        return False, "trust_prior_one"
    if attention < 10:
        return False, "attention_too_low"
    if attention > 120:
        return False, "attention_too_high"
    if effort < 1:
        return False, "effort_too_low"
    if noise > 0.5:
        return False, "noise_too_high"

    return True, None


def check_5_consistency(row):
    """
    AUDIT CHECK 5: Data Consistency Check
    Returns (flag: str or None, reason: str or None)
    Rows are NOT removed, only flagged.
    """
    try:
        age = int(row.get("age", 30))
    except (ValueError, TypeError):
        age = 30

    occupation = str(row.get("occupation", "")).lower()
    occupation_mapped = str(row.get("occupation_mapped", "")).lower()
    literacy = str(row.get("literacy_mapped", "")).lower()
    region = str(row.get("region_mapped", "")).lower()
    device = str(row.get("device_mapped", "")).lower()

    # Young person but retired
    if age <= 22 and "retired" in occupation:
        return "review_needed", "age_occupation_mismatch"

    # Graduate but listed as illiterate occupation match
    if literacy == "graduate" and occupation_mapped == "no_income":
        # Possible (students), but check age
        if age > 30:
            return "review_needed", "literacy_occupation_mismatch"

    # Metro + graduate + professional + basic_android (unlikely)
    if (region == "metro" and literacy == "graduate" and
            occupation_mapped == "professional" and device == "basic_android"):
        return "review_needed", "unlikely_combination"

    # Very old + student
    if age > 60 and "student" in occupation:
        return "review_needed", "age_occupation_mismatch"

    return None, None


def check_6_simulation_viability(row):
    """
    AUDIT CHECK 6: Simulation Viability Check
    Returns (passed: bool, reason: str or None)
    """
    # Check friction triggers
    triggers = str(row.get("top_friction_triggers", ""))
    if not triggers or triggers.strip() in ["", "nan", "[]", "none"]:
        return False, "empty_friction_triggers"

    # Check drop-off stage
    stage = str(row.get("primary_drop_off_stage", ""))
    if not stage or stage.strip().lower() in ["", "unknown", "nan", "none"]:
        return False, "unknown_drop_off_stage"

    # Check archetype format
    archetype = str(row.get("archetype", ""))
    if not ARCHETYPE_PATTERN.match(archetype):
        return False, "malformed_archetype"

    # Check parameter version
    pv = str(row.get("parameter_version", ""))
    if not pv or pv.strip() in ["", "nan", "none"]:
        return False, "missing_parameter_version"

    return True, None


# ══════════════════════════════════════════════════════════════
# CHECKPOINT SYSTEM
# ══════════════════════════════════════════════════════════════

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_checkpoint(rows_processed, chunk_idx):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "rows_processed": rows_processed,
            "chunk_idx": chunk_idx,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)


# ══════════════════════════════════════════════════════════════
# MAIN PROCESSING
# ══════════════════════════════════════════════════════════════

def run_auditor():
    """Main entry point for Agent 2."""
    t0 = time.time()

    for d in [DATA_PROCESSED, DATA_REPORTS]:
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("  PERCURA AGENT 2: AUDIT & VERIFICATION AGENT")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"\n  ERROR: {INPUT_FILE} not found!")
        print(f"  Run Agent 1 (agent_sorter.py) first.")
        sys.exit(1)

    # Count total rows
    print("\n  [1/4] Counting rows...")
    total_rows = sum(1 for _ in open(INPUT_FILE, encoding="utf-8")) - 1
    print(f"         Input: {total_rows:,} rows")

    n_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE

    # Check for resume
    checkpoint = load_checkpoint()
    start_chunk = 0
    if checkpoint:
        start_chunk = checkpoint.get("chunk_idx", 0) + 1
        if start_chunk * CHUNK_SIZE < total_rows:
            print(f"  [RESUME] From chunk {start_chunk}")

    # Initialize counters
    removal_reasons = Counter()
    flag_reasons = Counter()
    archetype_counter = Counter()
    total_clean = 0
    total_removed = 0
    total_flagged = 0
    total_processed = 0

    # Initialize output files
    clean_first = True
    removed_first = True
    flagged_first = True

    if start_chunk > 0 and os.path.exists(CLEAN_FILE):
        clean_first = False
        removed_first = False
        flagged_first = False

    print(f"\n  [2/4] Running 6 audit checks on {total_rows:,} rows...")

    reader = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, encoding="utf-8",
                         low_memory=False)

    for chunk_idx, df in tqdm(enumerate(reader), total=n_chunks,
                               desc="  Auditing", unit="chunk"):
        if chunk_idx < start_chunk:
            continue

        try:
            # ── Check 1: Completeness ──
            check1_results = df.apply(
                lambda r: check_1_completeness(r), axis=1, result_type="expand"
            )
            df["_check1_pass"] = check1_results[0]
            df["_check1_reason"] = check1_results[1]

            # ── Check 2: Duplicates ──
            df = check_2_duplicate(df)

            # ── Check 3: Behavioral usefulness ──
            check3_results = df.apply(
                lambda r: check_3_behavioral_usefulness(r), axis=1,
                result_type="expand"
            )
            df["_check3_pass"] = check3_results[0]
            df["_check3_reason"] = check3_results[1]

            # ── Check 5: Consistency (flag only) ──
            check5_results = df.apply(
                lambda r: check_5_consistency(r), axis=1, result_type="expand"
            )
            df["consistency_flag"] = check5_results[0]
            df["consistency_reason"] = check5_results[1]

            # ── Check 6: Simulation viability ──
            check6_results = df.apply(
                lambda r: check_6_simulation_viability(r), axis=1,
                result_type="expand"
            )
            df["_check6_pass"] = check6_results[0]
            df["_check6_reason"] = check6_results[1]

            # ── Determine final status ──
            df["removal_reason"] = None

            # Check 1 failures
            mask_c1 = ~df["_check1_pass"]
            df.loc[mask_c1, "removal_reason"] = df.loc[mask_c1, "_check1_reason"]

            # Check 2 failures (exact duplicates only — near dupes are flagged)
            mask_c2 = df["is_exact_duplicate"]
            df.loc[mask_c2 & df["removal_reason"].isna(), "removal_reason"] = "exact_duplicate"

            # Check 3 failures
            mask_c3 = ~df["_check3_pass"]
            df.loc[mask_c3 & df["removal_reason"].isna(), "removal_reason"] = df.loc[
                mask_c3 & df["removal_reason"].isna(), "_check3_reason"
            ]

            # Check 6 failures
            mask_c6 = ~df["_check6_pass"]
            df.loc[mask_c6 & df["removal_reason"].isna(), "removal_reason"] = df.loc[
                mask_c6 & df["removal_reason"].isna(), "_check6_reason"
            ]

            # Split into clean vs removed
            removed_mask = df["removal_reason"].notna()
            df_removed = df[removed_mask].copy()
            df_clean = df[~removed_mask].copy()

            # Add audit columns
            df_clean["audit_passed"] = True
            df_removed["audit_passed"] = False
            df_removed["removal_date"] = datetime.now().strftime("%Y-%m-%d")

            # Flagged subset (clean rows with consistency flags)
            df_flagged = df_clean[df_clean["consistency_flag"].notna()].copy()

            # Near-duplicate flag on clean rows
            near_dup_mask = df_clean["is_near_duplicate"]
            if near_dup_mask.any():
                df_clean.loc[near_dup_mask, "consistency_flag"] = "review_needed"
                df_clean.loc[near_dup_mask, "consistency_reason"] = (
                    df_clean.loc[near_dup_mask, "consistency_reason"].fillna("") + " near_duplicate"
                ).str.strip()
                near_flagged = df_clean[near_dup_mask & ~df_clean.index.isin(df_flagged.index)]
                df_flagged = pd.concat([df_flagged, near_flagged])

            # Drop internal columns before writing
            drop_cols = ["_check1_pass", "_check1_reason", "_check3_pass",
                         "_check3_reason", "_check6_pass", "_check6_reason",
                         "is_exact_duplicate", "is_near_duplicate"]
            for dfc in [df_clean, df_removed, df_flagged]:
                dfc.drop(columns=[c for c in drop_cols if c in dfc.columns],
                         inplace=True, errors="ignore")

            # Write outputs
            df_clean.to_csv(CLEAN_FILE, mode="a" if not clean_first else "w",
                            header=clean_first, index=False, encoding="utf-8")
            clean_first = False

            if len(df_removed) > 0:
                df_removed.to_csv(REMOVED_FILE, mode="a" if not removed_first else "w",
                                  header=removed_first, index=False, encoding="utf-8")
                removed_first = False

            if len(df_flagged) > 0:
                df_flagged.to_csv(FLAGGED_FILE, mode="a" if not flagged_first else "w",
                                  header=flagged_first, index=False, encoding="utf-8")
                flagged_first = False

            # Update counters
            total_clean += len(df_clean)
            total_removed += len(df_removed)
            total_flagged += len(df_flagged)
            total_processed += len(df)

            removal_reasons.update(
                df_removed["removal_reason"].value_counts().to_dict()
            )
            flag_reasons.update(
                df_flagged["consistency_reason"].dropna().value_counts().to_dict()
            )
            archetype_counter.update(
                df_clean["archetype"].value_counts().to_dict()
            )

            # Checkpoint
            if total_processed % CHECKPOINT_INTERVAL < CHUNK_SIZE:
                save_checkpoint(total_processed, chunk_idx)

        except Exception as e:
            print(f"\n  ERROR in chunk {chunk_idx}: {e}")
            save_checkpoint(total_processed, chunk_idx - 1)
            raise

    save_checkpoint(total_processed, n_chunks - 1)
    elapsed = time.time() - t0

    # ── Check 4: Archetype balance ──────────────────────────
    print(f"\n  [3/4] Checking archetype balance...")

    small_archetypes = {k: v for k, v in archetype_counter.items() if v < 100}
    large_archetypes = {k: v for k, v in archetype_counter.items() if v > 500_000}

    # ── Generate report ──────────────────────────────────────
    print(f"\n  [4/4] Generating audit report...")

    quality_score = total_clean / total_rows * 100 if total_rows else 0

    report_lines = [
        "=" * 60,
        "  PERCURA AGENT 2: AUDIT REPORT",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Processing time: {elapsed:.1f}s ({elapsed/60:.1f} min)",
        "=" * 60,
        "",
        "--- SUMMARY ---",
        f"  Total rows input:     {total_rows:>10,}",
        f"  Total rows PASSED:    {total_clean:>10,}",
        f"  Total rows REMOVED:   {total_removed:>10,}",
        f"  Total rows FLAGGED:   {total_flagged:>10,}",
        "",
        f"  DATA QUALITY SCORE:   {quality_score:.1f}%",
        f"  Target:               85.0%",
        f"  Status:               {'PASS' if quality_score >= 85 else 'NEEDS REVIEW'}",
        "",
        "--- REMOVAL REASONS ---",
    ]
    for reason, count in removal_reasons.most_common():
        pct = count / total_rows * 100
        report_lines.append(f"  {reason:40s} {count:>8,}  ({pct:.1f}%)")

    report_lines += ["", "--- FLAG REASONS ---"]
    for reason, count in flag_reasons.most_common():
        report_lines.append(f"  {reason:40s} {count:>8,}")

    report_lines += [
        "",
        "--- ARCHETYPE BALANCE ---",
        f"  Total unique archetypes:    {len(archetype_counter)}",
        f"  Archetypes < 100 personas:  {len(small_archetypes)}",
        f"  Archetypes > 500k personas: {len(large_archetypes)}",
    ]

    if small_archetypes:
        report_lines += ["", "  Small archetypes (< 100, not simulation-ready):"]
        for arch, count in sorted(small_archetypes.items(), key=lambda x: x[1]):
            report_lines.append(f"    {arch:55s} {count:>6,}")

    if large_archetypes:
        report_lines += ["", "  Oversized archetypes (> 500k, needs downsampling):"]
        for arch, count in sorted(large_archetypes.items(), key=lambda x: -x[1]):
            report_lines.append(f"    {arch:55s} {count:>8,}")

    report_lines += ["", "  Top 10 largest archetypes:"]
    for arch, count in archetype_counter.most_common(10):
        pct = count / total_clean * 100 if total_clean else 0
        report_lines.append(f"    {arch:55s} {count:>8,}  ({pct:.1f}%)")

    report_lines += ["", "  Bottom 10 smallest archetypes:"]
    for arch, count in archetype_counter.most_common()[:-11:-1]:
        report_lines.append(f"    {arch:55s} {count:>8,}")

    # Recommendations
    safe_archetypes = [k for k, v in archetype_counter.items() if v >= 100]
    needs_data = [k for k, v in archetype_counter.items() if v < 100]

    report_lines += [
        "",
        "--- RECOMMENDATIONS ---",
        "",
        f"  Archetypes SAFE to simulate now:    {len(safe_archetypes)}",
        f"  Archetypes needing more data:       {len(needs_data)}",
        "",
        "  Safe archetypes (top 10 by size):",
    ]
    for arch in safe_archetypes[:10]:
        report_lines.append(f"    - {arch} ({archetype_counter[arch]:,} personas)")

    if needs_data:
        report_lines += ["", "  Archetypes needing more data:"]
        for arch in needs_data[:10]:
            report_lines.append(f"    - {arch} ({archetype_counter[arch]:,} personas)")

    report_lines += [
        "",
        "=" * 60,
        "  OUTPUT FILES:",
        f"  Clean:   {CLEAN_FILE}  ({total_clean:,} rows)",
        f"  Removed: {REMOVED_FILE}  ({total_removed:,} rows)",
        f"  Flagged: {FLAGGED_FILE}  ({total_flagged:,} rows)",
        f"  Report:  {REPORT_FILE}",
        "=" * 60,
    ]

    report_text = "\n".join(report_lines)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\n  Done in {elapsed:.1f}s")

    return {
        "total_input": total_rows,
        "total_clean": total_clean,
        "total_removed": total_removed,
        "total_flagged": total_flagged,
        "quality_score": quality_score,
        "n_archetypes": len(archetype_counter),
        "safe_archetypes": len(safe_archetypes),
        "elapsed": elapsed,
    }


if __name__ == "__main__":
    run_auditor()
