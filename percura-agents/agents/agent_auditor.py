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

import numpy as np

# ══════════════════════════════════════════════════════════════
# VECTORIZED AUDIT CHECKS
# ══════════════════════════════════════════════════════════════

def apply_check_1_completeness(df):
    """AUDIT CHECK 1: Completeness Check"""
    df["_check1_pass"] = True
    df["_check1_reason"] = None
    
    # Missing key fields
    for field in ["education_level", "occupation", "district", "age"]:
        missing = df[field].isna() | (df[field].astype(str).str.strip() == "")
        df.loc[missing & df["_check1_pass"], "_check1_reason"] = "incomplete_key_fields"
        df.loc[missing, "_check1_pass"] = False

    # Age check
    age_num = pd.to_numeric(df["age"], errors="coerce")
    invalid_age = age_num.isna() | (age_num < 18) | (age_num > 90)
    df.loc[invalid_age & df["_check1_pass"], "_check1_reason"] = "invalid_age_or_out_of_range"
    df.loc[invalid_age, "_check1_pass"] = False

    # Name check
    name_str = df["extracted_name"].fillna("").astype(str).str.strip().str.lower()
    persona_len = df["persona"].fillna("").astype(str).str.len()
    bad_name = name_str.isin(["", "unknown", "test", "null", "none", "nan"])
    missing_name = bad_name & (persona_len < 10)
    df.loc[missing_name & df["_check1_pass"], "_check1_reason"] = "missing_name"
    df.loc[missing_name, "_check1_pass"] = False

    # Location vague
    dist_str = df["district"].fillna("").astype(str).str.strip().str.lower()
    state_str = df["state"].fillna("").astype(str).str.strip().str.lower()
    vague_loc = dist_str.isin(["india", "", "unknown"]) & state_str.isin(["", "unknown", "india"])
    df.loc[vague_loc & df["_check1_pass"], "_check1_reason"] = "location_too_vague"
    df.loc[vague_loc, "_check1_pass"] = False

    return df

def check_2_duplicate(df):
    """AUDIT CHECK 2: Duplicate Check"""
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

def apply_check_3_behavioral(df):
    """AUDIT CHECK 3: Behavioral Usefulness Check"""
    df["_check3_pass"] = True
    df["_check3_reason"] = None

    trust = pd.to_numeric(df["trust_prior"], errors="coerce").fillna(0.5)
    attention = pd.to_numeric(df["attention_budget"], errors="coerce").fillna(45)
    effort = pd.to_numeric(df["effort_tolerance"], errors="coerce").fillna(4)
    noise = pd.to_numeric(df["noise_level"], errors="coerce").fillna(0.15)

    mask_trust0 = trust == 0
    df.loc[mask_trust0 & df["_check3_pass"], "_check3_reason"] = "trust_prior_zero"
    df.loc[mask_trust0, "_check3_pass"] = False

    mask_trust1 = trust == 1
    df.loc[mask_trust1 & df["_check3_pass"], "_check3_reason"] = "trust_prior_one"
    df.loc[mask_trust1, "_check3_pass"] = False

    mask_att_low = attention < 10
    df.loc[mask_att_low & df["_check3_pass"], "_check3_reason"] = "attention_too_low"
    df.loc[mask_att_low, "_check3_pass"] = False

    mask_att_high = attention > 120
    df.loc[mask_att_high & df["_check3_pass"], "_check3_reason"] = "attention_too_high"
    df.loc[mask_att_high, "_check3_pass"] = False

    mask_eff_low = effort < 1
    df.loc[mask_eff_low & df["_check3_pass"], "_check3_reason"] = "effort_too_low"
    df.loc[mask_eff_low, "_check3_pass"] = False

    mask_noise = noise > 0.5
    df.loc[mask_noise & df["_check3_pass"], "_check3_reason"] = "noise_too_high"
    df.loc[mask_noise, "_check3_pass"] = False

    return df

def apply_check_5_consistency(df):
    """AUDIT CHECK 5: Data Consistency Check (Flag Only)"""
    df["consistency_flag"] = None
    df["consistency_reason"] = None

    age = pd.to_numeric(df["age"], errors="coerce").fillna(30)
    occ = df["occupation"].fillna("").astype(str).str.lower()
    occ_map = df["occupation_mapped"].fillna("").astype(str).str.lower()
    lit = df["literacy_mapped"].fillna("").astype(str).str.lower()
    reg = df["region_mapped"].fillna("").astype(str).str.lower()
    dev = df["device_mapped"].fillna("").astype(str).str.lower()

    # Retired but young
    mask1 = (age <= 22) & occ.str.contains("retired")
    df.loc[mask1, "consistency_flag"] = "review_needed"
    df.loc[mask1, "consistency_reason"] = "age_occupation_mismatch"

    # Graduate but no income and older
    mask2 = (lit == "graduate") & (occ_map == "no_income") & (age > 30)
    df.loc[mask2 & df["consistency_flag"].isna(), "consistency_flag"] = "review_needed"
    df.loc[mask2 & df["consistency_reason"].isna(), "consistency_reason"] = "literacy_occupation_mismatch"

    # Unlikely rich combo
    mask3 = (reg == "metro") & (lit == "graduate") & (occ_map == "professional") & (dev == "basic_android")
    df.loc[mask3 & df["consistency_flag"].isna(), "consistency_flag"] = "review_needed"
    df.loc[mask3 & df["consistency_reason"].isna(), "consistency_reason"] = "unlikely_combination"

    # Very old student
    mask4 = (age > 60) & occ.str.contains("student")
    df.loc[mask4 & df["consistency_flag"].isna(), "consistency_flag"] = "review_needed"
    df.loc[mask4 & df["consistency_reason"].isna(), "consistency_reason"] = "age_occupation_mismatch"

    return df

def apply_check_6_viability(df):
    """AUDIT CHECK 6: Simulation Viability Check"""
    df["_check6_pass"] = True
    df["_check6_reason"] = None

    # Friction triggers
    trig = df["top_friction_triggers"].fillna("").astype(str).str.strip().str.lower()
    mask_trig = trig.isin(["", "nan", "[]", "none"])
    df.loc[mask_trig & df["_check6_pass"], "_check6_reason"] = "empty_friction_triggers"
    df.loc[mask_trig, "_check6_pass"] = False

    # Drop-off stage
    stage = df["primary_drop_off_stage"].fillna("").astype(str).str.strip().str.lower()
    mask_stage = stage.isin(["", "unknown", "nan", "none"])
    df.loc[mask_stage & df["_check6_pass"], "_check6_reason"] = "unknown_drop_off_stage"
    df.loc[mask_stage, "_check6_pass"] = False

    # Archetype format
    arch = df["archetype"].fillna("").astype(str)
    # Vectorized regex match using str.match
    mask_arch = ~arch.str.match(
        r"^(youth|young|middle|senior)-"
        r"(metro|tier_2|tier_3|rural)-"
        r"(illiterate|primary|secondary|graduate)-"
        r"(no_income|informal|blue_collar|white_collar|professional)-"
        r"(basic_android|mid_android|high_android|iphone)$"
    )
    df.loc[mask_arch & df["_check6_pass"], "_check6_reason"] = "malformed_archetype"
    df.loc[mask_arch, "_check6_pass"] = False

    # Parameter version
    pv = df["parameter_version"].fillna("").astype(str).str.strip().str.lower()
    mask_pv = pv.isin(["", "nan", "none"])
    df.loc[mask_pv & df["_check6_pass"], "_check6_reason"] = "missing_parameter_version"
    df.loc[mask_pv, "_check6_pass"] = False

    return df


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
            df = apply_check_1_completeness(df)

            # ── Check 2: Duplicates ──
            df = check_2_duplicate(df)

            # ── Check 3: Behavioral usefulness ──
            df = apply_check_3_behavioral(df)

            # ── Check 5: Consistency (flag only) ──
            df = apply_check_5_consistency(df)

            # ── Check 6: Simulation viability ──
            df = apply_check_6_viability(df)

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
