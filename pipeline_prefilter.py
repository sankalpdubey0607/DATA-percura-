"""
pipeline_prefilter.py — Fix 1: Pre-filter reviews before extraction
Outputs: data/processed/reviews_filtered_out.csv, data/processed/reviews_to_extract.csv
"""
import csv, re, os, sys

RAW_FILE = "raw_reviews.csv"
OUT_DIR = os.path.join("data", "processed")
FILTERED_OUT_FILE = os.path.join(OUT_DIR, "reviews_filtered_out.csv")
TO_EXTRACT_FILE = os.path.join(OUT_DIR, "reviews_to_extract.csv")

EMOJI_PATTERN = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF\U0000FE00-\U0000FE0F\U0000200D\U00002764"
    "\U0000200B-\U0000200F\U0000FE0E-\U0000FE0F]+", flags=re.UNICODE)

NEGATIVE_WORDS = {
    "not","bad","problem","issue","fail","wrong","slow","crash","error",
    "cheat","fraud","waste","worst","terrible","horrible","pathetic",
    "useless","fix","bug","refund","scam","bakwaas","bekar","kharab",
    "ganda","faltu"
}

def run_prefilter():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    seen_texts = set()
    passed, filtered = [], []
    counts = {"too_short": 0, "emoji_only": 0, "duplicate": 0, "positive_5star_no_signal": 0}

    for row in rows:
        text = (row.get("review_text") or "").strip()
        rating = row.get("rating", "")

        # RULE 1: Too short (Relaxed to boost volume)
        if len(text) < 2:
            counts["too_short"] += 1
            filtered.append({**row, "filter_reason": "too_short"})
            continue

        # RULE 2: Emoji only (Disabled to allow extracting signals from emojis)
        stripped = EMOJI_PATTERN.sub("", text).strip()
        if False: # disabled
            counts["emoji_only"] += 1
            filtered.append({**row, "filter_reason": "emoji_only"})
            continue

        # RULE 3: Duplicate
        key = text.strip().lower()
        if key in seen_texts:
            counts["duplicate"] += 1
            filtered.append({**row, "filter_reason": "duplicate"})
            continue
        seen_texts.add(key)

        # RULE 4: Rating 5 with no negative words
        try:
            r = int(rating)
        except (ValueError, TypeError):
            r = 0
        if r == 5 and not any(w in text.lower() for w in NEGATIVE_WORDS):
            counts["positive_5star_no_signal"] += 1
            filtered.append({**row, "filter_reason": "positive_5star_no_signal"})
            continue

        passed.append(row)

    # Write filtered out
    with open(FILTERED_OUT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames + ["filter_reason"])
        w.writeheader()
        for row in filtered:
            w.writerow(row)

    # Write passed
    with open(TO_EXTRACT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in passed:
            w.writerow(row)

    total = len(rows)
    print(f"\n{'='*55}")
    print(f"  PRE-FILTER COMPLETE")
    print(f"{'='*55}")
    print(f"  Total reviews:    {total}")
    print(f"  Passed:           {len(passed)}")
    print(f"  Filtered out:     {len(filtered)}")
    print(f"\n  Breakdown:")
    for reason, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {reason:35s} {c:5d}  ({c/total*100:.1f}%)")
    print(f"\n  Saved: {TO_EXTRACT_FILE}")
    print(f"  Saved: {FILTERED_OUT_FILE}")
    print(f"{'='*55}")
    return len(passed)

if __name__ == "__main__":
    run_prefilter()
