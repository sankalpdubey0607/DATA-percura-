"""
Pre-filter analysis script — dry run to show what gets kept vs rejected.
Does NOT modify any files. Just prints counts.
"""
import csv
import re
import sys

RAW_REVIEWS_FILE = r"d:\DATA PERCURA\raw_reviews.csv"
RAW_REDDIT_FILE = r"d:\DATA PERCURA\raw_reddit.csv"

# ── Emoji regex ──
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"
    "\U00002764"  # ❤
    "\U0000200B-\U0000200F"  # zero-width chars
    "\U0000FE0E-\U0000FE0F"  # variation selectors
    "]+",
    flags=re.UNICODE,
)

NEGATIVE_WORDS = {
    "not", "bad", "problem", "issue", "fail", "wrong", "slow",
    "crash", "error", "cheat", "fraud", "waste", "worst", "terrible",
    "horrible", "pathetic", "useless", "fix", "bug", "refund", "scam",
    "bakwaas", "bekar", "kharab", "ganda", "faltu"
}

def strip_emojis(text):
    return EMOJI_PATTERN.sub("", text)

def has_negative_words(text):
    text_lower = text.lower()
    for word in NEGATIVE_WORDS:
        if word in text_lower:
            return True
    return False

def apply_filters(rows):
    """Apply all 4 filter rules. Returns (passed, filtered_out_with_reasons)"""
    seen_texts = set()
    passed = []
    filtered = []
    
    filter_counts = {
        "too_short": 0,
        "emoji_only": 0,
        "duplicate": 0,
        "positive_5star_no_signal": 0,
    }
    
    for row in rows:
        review_text = (row.get("review_text") or "").strip()
        rating = row.get("rating", "")
        
        # RULE 1: Too short (< 15 chars after stripping whitespace)
        if len(review_text.strip()) < 15:
            filter_counts["too_short"] += 1
            filtered.append({**row, "filter_reason": "too_short"})
            continue
        
        # RULE 2: Emoji only (after removing emojis, fewer than 5 real chars)
        stripped = strip_emojis(review_text).strip()
        if len(stripped) < 5:
            filter_counts["emoji_only"] += 1
            filtered.append({**row, "filter_reason": "emoji_only"})
            continue
        
        # RULE 3: Duplicate text
        text_key = review_text.strip().lower()
        if text_key in seen_texts:
            filter_counts["duplicate"] += 1
            filtered.append({**row, "filter_reason": "duplicate"})
            continue
        seen_texts.add(text_key)
        
        # RULE 4: Rating 5 with no negative words
        try:
            r = int(rating)
        except (ValueError, TypeError):
            r = 0
        
        if r == 5 and not has_negative_words(review_text):
            filter_counts["positive_5star_no_signal"] += 1
            filtered.append({**row, "filter_reason": "positive_5star_no_signal"})
            continue
        
        passed.append(row)
    
    return passed, filtered, filter_counts

# ── Load data ──
print("=" * 60)
print("  PERCURA PRE-FILTER ANALYSIS (DRY RUN)")
print("=" * 60)

# Load reviews
rows = []
with open(RAW_REVIEWS_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

total = len(rows)
print(f"\nTotal raw reviews: {total}")

# App breakdown
from collections import Counter
app_counts = Counter(r["app_name"] for r in rows)
print(f"\nApps ({len(app_counts)}):")
for app, count in app_counts.most_common():
    print(f"  {app:20s} {count:5d} reviews")

# Rating breakdown
rating_counts = Counter(r["rating"] for r in rows)
print(f"\nRating distribution:")
for rating in sorted(rating_counts.keys()):
    print(f"  {rating}-star: {rating_counts[rating]:5d}")

# ── Apply filters ──
passed, filtered, filter_counts = apply_filters(rows)

print(f"\n{'=' * 60}")
print(f"  PRE-FILTER RESULTS")
print(f"{'=' * 60}")
print(f"\n  Total reviews:           {total}")
print(f"  PASSED filter (-> API):   {len(passed)}")
print(f"  FILTERED OUT:            {len(filtered)}")
print(f"\n  Breakdown by reason:")
for reason, count in sorted(filter_counts.items(), key=lambda x: -x[1]):
    pct = count / total * 100
    print(f"    {reason:35s} {count:5d}  ({pct:.1f}%)")

# Passed by app
passed_apps = Counter(r["app_name"] for r in passed)
print(f"\n  Reviews PASSING filter by app:")
for app, count in passed_apps.most_common():
    orig = app_counts[app]
    pct = count / orig * 100 if orig else 0
    print(f"    {app:20s} {count:4d} / {orig:4d}  ({pct:.1f}% kept)")

# Passed by rating
passed_ratings = Counter(r["rating"] for r in passed)
print(f"\n  Reviews PASSING filter by rating:")
for rating in sorted(passed_ratings.keys()):
    orig = rating_counts.get(rating, 0)
    count = passed_ratings[rating]
    print(f"    {rating}-star: {count:4d} / {orig:4d}")

# ── Cost estimate ──
print(f"\n{'=' * 60}")
print(f"  API COST ESTIMATE")
print(f"{'=' * 60}")

reviews_to_api = len(passed)
# Reddit posts
reddit_rows = []
try:
    with open(RAW_REDDIT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reddit_rows.append(row)
    reddit_count = len(reddit_rows)
except:
    reddit_count = 0

total_api_calls = reviews_to_api + reddit_count

# Groq pricing: llama-3.3-70b-versatile
# Input: ~$0.59/M tokens, Output: ~$0.79/M tokens (Groq pricing)
# Average review prompt: ~300 tokens input, ~200 tokens output
# Average reddit prompt: ~800 tokens input, ~200 tokens output

avg_review_input_tokens = 350
avg_review_output_tokens = 200
avg_reddit_input_tokens = 800
avg_reddit_output_tokens = 250

review_input_cost = reviews_to_api * avg_review_input_tokens / 1_000_000 * 0.59
review_output_cost = reviews_to_api * avg_review_output_tokens / 1_000_000 * 0.79
reddit_input_cost = reddit_count * avg_reddit_input_tokens / 1_000_000 * 0.59
reddit_output_cost = reddit_count * avg_reddit_output_tokens / 1_000_000 * 0.79

total_cost = review_input_cost + review_output_cost + reddit_input_cost + reddit_output_cost

print(f"\n  Reviews passing filter:   {reviews_to_api}")
print(f"  Reddit posts:             {reddit_count}")
print(f"  Total API calls:          {total_api_calls}")
print(f"\n  Groq (llama-3.3-70b-versatile) pricing:")
print(f"    Review input tokens:    ~{reviews_to_api * avg_review_input_tokens:,} ({review_input_cost:.4f} USD)")
print(f"    Review output tokens:   ~{reviews_to_api * avg_review_output_tokens:,} ({review_output_cost:.4f} USD)")
print(f"    Reddit input tokens:    ~{reddit_count * avg_reddit_input_tokens:,} ({reddit_input_cost:.4f} USD)")
print(f"    Reddit output tokens:   ~{reddit_count * avg_reddit_output_tokens:,} ({reddit_output_cost:.4f} USD)")
print(f"\n  * ESTIMATED TOTAL COST:   ${total_cost:.4f} USD")
print(f"    (That's ~₹{total_cost * 85:.2f} INR)")

# Time estimate at 15 RPM (4s delay)
time_seconds = total_api_calls * 4
time_minutes = time_seconds / 60
time_hours = time_minutes / 60
print(f"\n  ★ ESTIMATED TIME:         {time_minutes:.0f} minutes ({time_hours:.1f} hours)")
print(f"    (at 15 RPM / 4s per call)")

# ── Sample filtered reviews ──
print(f"\n{'=' * 60}")
print(f"  SAMPLE: Reviews that WOULD be filtered out")
print(f"{'=' * 60}")
for reason in ["too_short", "emoji_only", "duplicate", "positive_5star_no_signal"]:
    samples = [r for r in filtered if r["filter_reason"] == reason][:3]
    if samples:
        print(f"\n  [{reason}] (showing {len(samples)} of {filter_counts[reason]}):")
        for s in samples:
            text = (s.get("review_text") or "")[:60]
            print(f"    ★{s['rating']}  {s['app_name']:12s}  \"{text}\"")

# ── Sample passed reviews ──
print(f"\n{'=' * 60}")
print(f"  SAMPLE: Reviews that WOULD be sent to API")
print(f"{'=' * 60}")
for rating in ["1", "2", "3"]:
    samples = [r for r in passed if r["rating"] == rating][:3]
    if samples:
        print(f"\n  [Rating {rating}] (showing {len(samples)}):")
        for s in samples:
            text = (s.get("review_text") or "")[:80]
            print(f"    {s['app_name']:12s}  \"{text}\"")

print(f"\n{'=' * 60}")
print(f"  WAITING FOR YOUR GO-AHEAD BEFORE BUILDING ANYTHING")
print(f"{'=' * 60}")
