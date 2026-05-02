"""
pipeline_merge_report.py — Fix 6: Merge all extracted data + generate report
Creates: data/processed/all_behaviors_master.csv
         data/reports/extraction_report.txt
"""
import csv, os, sys
from collections import Counter
from datetime import datetime

OUT_DIR = os.path.join("data", "processed")
REPORT_DIR = os.path.join("data", "reports")
REVIEWS_EXTRACTED = os.path.join(OUT_DIR, "extracted_behaviors.csv")
REDDIT_EXTRACTED = os.path.join(OUT_DIR, "extracted_reddit.csv")
MASTER_FILE = os.path.join(OUT_DIR, "all_behaviors_master.csv")
FILTERED_OUT_FILE = os.path.join(OUT_DIR, "reviews_filtered_out.csv")
REPORT_FILE = os.path.join(REPORT_DIR, "extraction_report.txt")

MASTER_FIELDS = [
    "source", "extraction_version", "app_name", "app_id", "review_id",
    "rating", "date", "review_text",
    "drop_off_stage", "friction_type", "emotion", "gave_up", "trust_signal",
    "effort_complained", "language", "literacy_hint", "device_hint",
    "income_hint", "region_hint", "key_quote", "confidence",
    "product_mentioned", "issue_category", "api_used",
]


def run_merge_and_report():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    master_rows = []

    # ── Load review extractions ──
    review_count = 0
    if os.path.exists(REVIEWS_EXTRACTED):
        with open(REVIEWS_EXTRACTED, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                master_rows.append({
                    "source": "playstore",
                    "extraction_version": "v2",
                    "app_name": row.get("app_name", ""),
                    "app_id": row.get("app_id", ""),
                    "review_id": row.get("review_id", ""),
                    "rating": row.get("rating", ""),
                    "date": row.get("date", ""),
                    "review_text": row.get("review_text", ""),
                    "drop_off_stage": row.get("drop_off_stage", ""),
                    "friction_type": row.get("friction_type", ""),
                    "emotion": row.get("emotion", ""),
                    "gave_up": row.get("gave_up", ""),
                    "trust_signal": row.get("trust_signal", ""),
                    "effort_complained": row.get("effort_complained", ""),
                    "language": row.get("language", ""),
                    "literacy_hint": row.get("literacy_hint", ""),
                    "device_hint": row.get("device_hint", ""),
                    "income_hint": row.get("income_hint", ""),
                    "region_hint": row.get("region_hint", ""),
                    "key_quote": row.get("key_quote", ""),
                    "confidence": row.get("confidence", ""),
                    "product_mentioned": row.get("app_name", ""),
                    "issue_category": "",
                    "api_used": row.get("api_used", "groq"),
                })
                review_count += 1
        print(f"  Loaded {review_count} extracted reviews")
    else:
        print(f"  WARNING: {REVIEWS_EXTRACTED} not found")

    # ── Load Reddit extractions ──
    reddit_count = 0
    if os.path.exists(REDDIT_EXTRACTED):
        with open(REDDIT_EXTRACTED, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                master_rows.append({
                    "source": "reddit",
                    "extraction_version": "v2",
                    "app_name": row.get("product_mentioned", ""),
                    "app_id": "",
                    "review_id": row.get("post_id", ""),
                    "rating": "",
                    "date": row.get("date", ""),
                    "review_text": row.get("title", ""),
                    "drop_off_stage": row.get("drop_off_stage", ""),
                    "friction_type": row.get("friction_type", ""),
                    "emotion": row.get("emotion", ""),
                    "gave_up": row.get("gave_up", ""),
                    "trust_signal": row.get("trust_signal", ""),
                    "effort_complained": row.get("effort_complained", ""),
                    "language": row.get("language", ""),
                    "literacy_hint": row.get("literacy_hint", ""),
                    "device_hint": row.get("device_hint", ""),
                    "income_hint": row.get("income_hint", ""),
                    "region_hint": row.get("region_hint", ""),
                    "key_quote": row.get("key_quote", ""),
                    "confidence": row.get("confidence", ""),
                    "product_mentioned": row.get("product_mentioned", ""),
                    "issue_category": row.get("issue_category", ""),
                    "api_used": row.get("api_used", "groq"),
                })
                reddit_count += 1
        print(f"  Loaded {reddit_count} extracted Reddit posts")
    else:
        print(f"  WARNING: {REDDIT_EXTRACTED} not found")

    # ── Write master CSV ──
    with open(MASTER_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_FIELDS)
        w.writeheader()
        for row in master_rows:
            w.writerow(row)

    total_master = len(master_rows)
    print(f"\n  Master file: {MASTER_FILE} ({total_master} rows)")

    # ── Load filter stats ──
    filter_reasons = Counter()
    filtered_total = 0
    if os.path.exists(FILTERED_OUT_FILE):
        with open(FILTERED_OUT_FILE, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                filter_reasons[row.get("filter_reason", "unknown")] += 1
                filtered_total += 1

    # ── Generate report ──
    # Compute distributions from master data
    confidence_dist = Counter(r["confidence"] for r in master_rows)
    stage_dist = Counter(r["drop_off_stage"] for r in master_rows)
    emotion_dist = Counter(r["emotion"] for r in master_rows)
    language_dist = Counter(r["language"] for r in master_rows)
    trust_dist = Counter(r["trust_signal"] for r in master_rows)
    gave_up_dist = Counter(str(r["gave_up"]).lower() for r in master_rows)
    source_dist = Counter(r["source"] for r in master_rows)

    # Friction types (pipe-separated)
    friction_counter = Counter()
    for r in master_rows:
        ft = r.get("friction_type", "")
        if ft:
            for f in ft.split("|"):
                f = f.strip()
                if f:
                    friction_counter[f] += 1

    # App distribution (playstore only)
    app_dist = Counter(r["app_name"] for r in master_rows if r["source"] == "playstore")

    # Per-app stats
    app_confidence = {}
    for r in master_rows:
        if r["source"] == "playstore":
            app = r["app_name"]
            if app not in app_confidence:
                app_confidence[app] = Counter()
            app_confidence[app][r["confidence"]] += 1

    raw_total = 2934  # from raw_reviews.csv
    raw_reddit = 133

    report_lines = [
        "=" * 60,
        "  PERCURA DATA EXTRACTION REPORT v2",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "--- DATA COLLECTION ---",
        f"  Total reviews collected (Play Store):  {raw_total}",
        f"  Total posts collected (Reddit):        {raw_reddit}",
        f"  Apps scraped:                          6 (PhonePe, Paytm, Meesho, BYJU's, UrbanCompany, Zomato)",
        "",
        "--- PRE-FILTER RESULTS ---",
        f"  Reviews filtered out:                  {filtered_total}",
        f"  Reviews sent to API:                   {raw_total - filtered_total}",
        "  Filter breakdown:",
    ]
    for reason, count in filter_reasons.most_common():
        report_lines.append(f"    {reason:35s} {count:5d}")

    report_lines += [
        "",
        "--- EXTRACTION RESULTS ---",
        f"  Reviews successfully extracted:         {review_count}",
        f"  Reddit posts extracted:                 {reddit_count}",
        f"  Total in master file:                   {total_master}",
        f"  Extraction success rate (reviews):      {review_count/(raw_total-filtered_total)*100:.1f}%" if (raw_total-filtered_total) > 0 else "  N/A",
        "",
        "--- SOURCE DISTRIBUTION ---",
    ]
    for src, count in source_dist.most_common():
        report_lines.append(f"    {src:20s} {count:5d}")

    report_lines += ["", "--- CONFIDENCE DISTRIBUTION ---"]
    for conf, count in confidence_dist.most_common():
        pct = count / total_master * 100 if total_master else 0
        report_lines.append(f"    {conf:20s} {count:5d}  ({pct:.1f}%)")

    report_lines += ["", "--- DROP-OFF STAGE DISTRIBUTION ---"]
    for stage, count in stage_dist.most_common():
        pct = count / total_master * 100 if total_master else 0
        report_lines.append(f"    {stage:25s} {count:5d}  ({pct:.1f}%)")

    report_lines += ["", "--- TOP FRICTION TYPES ---"]
    for ft, count in friction_counter.most_common(15):
        report_lines.append(f"    {ft:25s} {count:5d}")

    report_lines += ["", "--- EMOTION DISTRIBUTION ---"]
    for emo, count in emotion_dist.most_common():
        pct = count / total_master * 100 if total_master else 0
        report_lines.append(f"    {emo:20s} {count:5d}  ({pct:.1f}%)")

    report_lines += ["", "--- TRUST SIGNAL DISTRIBUTION ---"]
    for ts, count in trust_dist.most_common():
        report_lines.append(f"    {ts:25s} {count:5d}")

    report_lines += ["", "--- GAVE UP DISTRIBUTION ---"]
    for gu, count in gave_up_dist.most_common():
        report_lines.append(f"    {gu:20s} {count:5d}")

    report_lines += ["", "--- LANGUAGE BREAKDOWN ---"]
    for lang, count in language_dist.most_common():
        pct = count / total_master * 100 if total_master else 0
        report_lines.append(f"    {lang:20s} {count:5d}  ({pct:.1f}%)")

    report_lines += ["", "--- PER-APP EXTRACTION QUALITY ---"]
    for app in sorted(app_confidence.keys()):
        counts = app_confidence[app]
        total_app = sum(counts.values())
        high_pct = counts.get("high", 0) / total_app * 100 if total_app else 0
        med_pct = counts.get("medium", 0) / total_app * 100 if total_app else 0
        low_pct = counts.get("low", 0) / total_app * 100 if total_app else 0
        report_lines.append(f"    {app:15s}  total={total_app:4d}  high={high_pct:.0f}%  med={med_pct:.0f}%  low={low_pct:.0f}%")

    report_lines += [
        "",
        "--- RECOMMENDATIONS FOR AGENT 1 (SORTER) ---",
        "",
        "  1. CONFIDENCE: Review the 'low' confidence extractions manually.",
        "     These are short or ambiguous reviews where the AI was guessing.",
        "     Consider weighting high-confidence rows more in archetype creation.",
        "",
        "  2. FRICTION TYPES: The most common friction types should map directly",
        "     to archetype behavioral parameters. Use frequency to set priors.",
        "",
        "  3. TRUST SIGNALS: 'lost_trust' and 'never_had_trust' segments are",
        "     critical for the 'Skeptic' and 'Betrayed Loyalist' archetypes.",
        "",
        "  4. LANGUAGE MIX: Hinglish and Hindi reviews provide the strongest",
        "     signals for tier-2/tier-3 user archetypes. Weight these higher",
        "     for regional simulation accuracy.",
        "",
        "  5. GAVE UP = TRUE: These users represent the hardest drop-off cases.",
        "     Map their friction_type + drop_off_stage combinations to create",
        "     the 'rage quit' behavioral paths in simulation.",
        "",
        "  6. REDDIT DATA: Reddit posts have higher literacy and detail.",
        "     Use these for the 'tech-savvy urban' archetype parameters.",
        "",
        "  7. DATA GAPS: UrbanCompany and BYJU's have the richest signal.",
        "     PhonePe has lowest signal density (too many generic reviews).",
        "     Consider collecting more negative PhonePe reviews.",
        "",
        "=" * 60,
        "  END OF REPORT",
        "=" * 60,
    ]

    report_text = "\n".join(report_lines)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n  Report: {REPORT_FILE}")
    print(f"\n{'='*55}")
    print(f"  MERGE + REPORT COMPLETE")
    print(f"  Master file: {total_master} rows")
    print(f"  Report: {REPORT_FILE}")
    print(f"{'='*55}")

    return total_master


if __name__ == "__main__":
    run_merge_and_report()
