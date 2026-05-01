"""
extractor.py
------------
Reads raw_reviews.csv and raw_reddit.csv, runs AI behavioral extraction
via Groq API, and writes to extracted_behaviors.csv.

Improvements over v1:
  - Uses llama-3.3-70b-versatile (much better JSON extraction than 8b)
  - Quality filter: skips emoji-only / single-word / ultra-short reviews
  - Resume capability: detects already-processed IDs, appends new rows
  - Reddit extraction: processes raw_reddit.csv with adapted prompt
  - Source + category columns for Person B's archetype pipeline

Groq free tier for llama-3.3-70b-versatile:
  - 30 RPM limit
  - Strategy: 4s delay (= 15 RPM, safely under 30 RPM)
  - Exponential backoff on 429 errors
"""

import csv
import json
import os
import re
import time
import random

from groq import Groq, RateLimitError as GroqRateLimitError
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
RAW_REVIEWS_FILE = "raw_reviews.csv"
RAW_REDDIT_FILE  = "raw_reddit.csv"
OUTPUT_FILE      = "extracted_behaviors.csv"
ERRORS_FILE      = "errors.csv"

# Model config — change to "llama-3.1-8b-instant" if you hit persistent rate limits
GROQ_MODEL = "llama-3.3-70b-versatile"

# Rate limiting
BASE_DELAY   = 4.0      # seconds between requests (15 RPM)
MAX_RETRIES  = 6
BACKOFF_BASE = 20       # seconds for first 429 retry (doubles each attempt)

# Quality filter thresholds
MIN_CHAR_LENGTH = 15    # minimum characters after stripping emojis
MIN_WORD_COUNT  = 3     # minimum words after stripping emojis

# ── Category mapping ─────────────────────────────────────────────────────────
CATEGORY_MAP = {
    "PhonePe":       "fintech",
    "Paytm":         "fintech",
    "Meesho":        "ecommerce",
    "BYJU's":        "edtech",
    "Zomato":        "food_delivery",
    "UrbanCompany":  "services",
    "Jupiter":       "fintech",
    "Fi Money":      "fintech",
    "Slice":         "fintech",
    "Vedantu":       "edtech",
    "Unacademy":     "edtech",
    "PhysicsWallah": "edtech",
    "Flipkart":      "ecommerce",
    "Glowroad":      "ecommerce",
    "Swiggy":        "food_delivery",
    "Practo":        "health",
    "mfine":         "health",
    "1mg":           "health",
    "Apna":          "job_hunting",
    "WorkIndia":     "job_hunting",
}

# ── Output schema ─────────────────────────────────────────────────────────────
OUTPUT_FIELDS = [
    "source", "app_name", "app_id", "category", "review_id", "rating", "date",
    "review_text", "drop_off_stage", "friction_type", "emotion", "gave_up",
    "trust_signal", "effort_complained", "language",
    "literacy_hint", "device_hint", "income_hint", "region_hint",
    "key_quote", "confidence", "api_used",
]
ERROR_FIELDS = ["row_number", "source", "app_name", "review_id", "error", "raw_response"]


# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a behavioral data extraction specialist for an Indian user research platform. Your job is to read real user reviews and complaints about apps, and extract structured behavioral signals from them.

You must ALWAYS output valid JSON and nothing else. No explanation, no preamble, no markdown code blocks.

If you cannot determine a value from the review text, use "unknown". Never guess or invent information that is not in the review.

Be conservative but thorough. Look for:
- Specific UX friction points (what exactly went wrong)
- Emotional tone and intensity
- Trust signals (did they lose faith in the app?)
- Evidence of giving up vs continuing
- Demographic clues from language, spelling, vocabulary
- Device/income/region clues from context"""


def build_playstore_prompt(app_name: str, review_text: str) -> str:
    return f"""Extract behavioral signals from this Google Play Store review.
App: {app_name}
Review: {review_text}

Return ONLY this JSON:
{{
  "drop_off_stage": "",
  "friction_type": [],
  "emotion": "",
  "gave_up": null,
  "trust_signal": "",
  "effort_complained": null,
  "language": "",
  "archetype_signals": {{
    "literacy_hint": "",
    "device_hint": "",
    "income_hint": "",
    "region_hint": ""
  }},
  "key_quote": "",
  "confidence": ""
}}

drop_off_stage values: sign_up | login | onboarding | form_fill | payment | verification | home_screen | feature_use | checkout | support | update | unknown

friction_type values (pick all that apply): too_many_steps | confusing_language | trust_issue | slow_loading | form_too_long | verification_failed | price_shock | hidden_charges | forced_signup | no_hindi_support | technical_error | poor_support | data_privacy_fear | ads_spam | battery_drain | app_crash | unknown

emotion values: frustrated | confused | betrayed | bored | scared | disappointed | angry | hopeful | neutral | happy | unknown

gave_up: true if user deleted/stopped/uninstalled. false if still using. null if unclear.

trust_signal: trusted | did_not_trust | lost_trust | never_had_trust | none | unknown

effort_complained: true if user mentioned too many steps/too much work. false if not. null if unclear.

language: english | hindi | hinglish | gujarati | tamil | telugu | marathi | kannada | bengali | other | unknown

archetype hints (infer from vocabulary, grammar, spelling, content clues):
  literacy_hint: high (good grammar, technical terms) | medium (decent but casual) | low (broken grammar, phonetic spelling) | unknown
  device_hint: iphone | high-end-android | mid-android | basic-android | unknown
  income_hint: high | middle | low | unknown
  region_hint: metro | tier-2 | tier-3 | rural | unknown

key_quote: copy the single most revealing phrase about their experience, under 15 words. Empty string if none.
confidence: high (clear signals) | medium (some inference needed) | low (mostly guessing)"""


def build_reddit_prompt(title: str, body: str, comments: str) -> str:
    return f"""Extract behavioral signals from this Reddit post about an Indian app or product.

Post Title: {title}
Post Body: {body}
Top Comments: {comments}

Return ONLY this JSON:
{{
  "app_mentioned": "",
  "drop_off_stage": "",
  "friction_type": [],
  "emotion": "",
  "gave_up": null,
  "trust_signal": "",
  "effort_complained": null,
  "language": "",
  "archetype_signals": {{
    "literacy_hint": "",
    "device_hint": "",
    "income_hint": "",
    "region_hint": ""
  }},
  "key_quote": "",
  "confidence": ""
}}

app_mentioned: name of the app or product being discussed. Use "unknown" if not clear.

drop_off_stage values: sign_up | login | onboarding | form_fill | payment | verification | home_screen | feature_use | checkout | support | update | unknown

friction_type values (pick all that apply): too_many_steps | confusing_language | trust_issue | slow_loading | form_too_long | verification_failed | price_shock | hidden_charges | forced_signup | no_hindi_support | technical_error | poor_support | data_privacy_fear | ads_spam | battery_drain | app_crash | unknown

emotion values: frustrated | confused | betrayed | bored | scared | disappointed | angry | hopeful | neutral | happy | unknown

gave_up: true if user deleted/stopped/uninstalled. false if still using. null if unclear.

trust_signal: trusted | did_not_trust | lost_trust | never_had_trust | none | unknown

effort_complained: true if user mentioned too many steps/too much work. false if not. null if unclear.

language: english | hindi | hinglish | gujarati | tamil | telugu | marathi | kannada | bengali | other | unknown

archetype hints:
  literacy_hint: high | medium | low | unknown
  device_hint: iphone | high-end-android | mid-android | basic-android | unknown
  income_hint: high | middle | low | unknown
  region_hint: metro | tier-2 | tier-3 | rural | unknown

key_quote: copy the single most revealing phrase, under 15 words. Empty string if none.
confidence: high | medium | low"""


# ── Helpers ───────────────────────────────────────────────────────────────────
# Regex to strip emojis and special unicode symbols
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # misc
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended-A
    "\U00002600-\U000026FF"  # misc symbols
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # zero width joiner
    "]+",
    flags=re.UNICODE,
)


def _is_low_quality(text: str) -> bool:
    """Returns True if review text is too short/meaningless to extract signals from."""
    stripped = EMOJI_PATTERN.sub("", text).strip()

    # Too short after removing emojis
    if len(stripped) < MIN_CHAR_LENGTH:
        return True

    # Too few real words
    words = [w for w in stripped.split() if len(w) > 1]
    if len(words) < MIN_WORD_COUNT:
        return True

    return False


def _clean_json(raw: str) -> str:
    """Strip markdown code fences if the model wraps JSON in them."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def _call_groq_with_backoff(client, system_prompt: str, user_prompt: str) -> str:
    """Call Groq API with exponential backoff on rate limit errors."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            chat = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=500,
                temperature=0,
            )
            return chat.choices[0].message.content

        except GroqRateLimitError as e:
            last_error = e
            wait = BACKOFF_BASE * (2 ** attempt) + random.uniform(1, 5)
            print(f"    [RATE LIMIT] 429 on attempt {attempt+1}/{MAX_RETRIES}. Sleeping {wait:.0f}s...")
            time.sleep(wait)

        except Exception as e:
            raise e  # Non-rate-limit errors bubble up immediately

    raise Exception(f"Groq rate limit persisted after {MAX_RETRIES} retries: {last_error}")


def _load_existing_ids(output_file: str) -> tuple[set, bool]:
    """
    Load review IDs already processed from existing output file.
    Returns (set_of_ids, schema_matches).
    If the file exists but has a different schema, returns (empty_set, False).
    """
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        return set(), False

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_headers = set(reader.fieldnames or [])
            expected_headers = set(OUTPUT_FIELDS)

            # Check if schemas match (allow existing to be a subset)
            if not expected_headers.issubset(existing_headers) and existing_headers != expected_headers:
                print(f"  [WARN] Schema mismatch in {output_file}. Will start fresh.")
                return set(), False

            ids = {row.get("review_id", "") for row in reader}
            return ids, True
    except Exception:
        return set(), False


def _write_extracted_row(writer, source, app_name, app_id, category, review_id,
                          rating, date, review_text, extracted):
    """Write a single extracted row to the output CSV."""
    arch = extracted.get("archetype_signals", {})
    friction = extracted.get("friction_type", [])
    if isinstance(friction, list):
        friction = "|".join(friction)

    writer.writerow({
        "source":            source,
        "app_name":          app_name,
        "app_id":            app_id,
        "category":          category,
        "review_id":         review_id,
        "rating":            rating,
        "date":              date,
        "review_text":       review_text,
        "drop_off_stage":    extracted.get("drop_off_stage",    "unknown"),
        "friction_type":     friction,
        "emotion":           extracted.get("emotion",           "unknown"),
        "gave_up":           extracted.get("gave_up"),
        "trust_signal":      extracted.get("trust_signal",      "unknown"),
        "effort_complained": extracted.get("effort_complained"),
        "language":          extracted.get("language",          "unknown"),
        "literacy_hint":     arch.get("literacy_hint",          "unknown"),
        "device_hint":       arch.get("device_hint",            "unknown"),
        "income_hint":       arch.get("income_hint",            "unknown"),
        "region_hint":       arch.get("region_hint",            "unknown"),
        "key_quote":         extracted.get("key_quote",         ""),
        "confidence":        extracted.get("confidence",        "low"),
        "api_used":          "groq",
    })


# ── Main extractor ────────────────────────────────────────────────────────────
def extract_behaviors(
    raw_reviews_file: str = RAW_REVIEWS_FILE,
    raw_reddit_file: str  = RAW_REDDIT_FILE,
    output_file: str      = OUTPUT_FILE,
    errors_file: str      = ERRORS_FILE,
) -> dict:

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # ── Resume support ────────────────────────────────────────────────────────
    existing_ids, schema_ok = _load_existing_ids(output_file)
    resume_mode = len(existing_ids) > 0 and schema_ok

    if resume_mode:
        print(f"\n[Extractor] RESUME MODE: {len(existing_ids)} reviews already processed. Appending new rows.")
        write_mode = "a"
        write_header = False
    else:
        print(f"\n[Extractor] FRESH START: writing to {output_file}")
        write_mode = "w"
        write_header = True
        existing_ids = set()

    # ── Count total work ──────────────────────────────────────────────────────
    playstore_total = 0
    if os.path.exists(raw_reviews_file):
        with open(raw_reviews_file, "r", encoding="utf-8") as f:
            playstore_total = sum(1 for _ in f) - 1

    reddit_total = 0
    if os.path.exists(raw_reddit_file):
        with open(raw_reddit_file, "r", encoding="utf-8") as f:
            reddit_total = sum(1 for _ in f) - 1

    print(f"[Extractor] Sources: {playstore_total} Play Store reviews + {reddit_total} Reddit posts")
    print(f"  Model: {GROQ_MODEL}")
    print(f"  Rate: {BASE_DELAY}s/request = {60/BASE_DELAY:.0f} RPM")
    print(f"  Quality filter: skip reviews < {MIN_CHAR_LENGTH} chars or < {MIN_WORD_COUNT} words")

    stats = {
        "processed": 0, "success": 0, "errors": 0,
        "skipped_existing": 0, "skipped_low_quality": 0,
    }

    with (
        open(output_file, write_mode, newline="", encoding="utf-8") as outfile,
        open(errors_file, "w", newline="", encoding="utf-8") as errfile,
    ):
        writer     = csv.DictWriter(outfile, fieldnames=OUTPUT_FIELDS)
        err_writer = csv.DictWriter(errfile, fieldnames=ERROR_FIELDS)
        if write_header:
            writer.writeheader()
        err_writer.writeheader()

        # ── PHASE 1: Play Store Reviews ───────────────────────────────────────
        if os.path.exists(raw_reviews_file) and playstore_total > 0:
            print(f"\n{'─' * 50}")
            print(f"  PHASE 1: Play Store Reviews ({playstore_total} total)")
            print(f"{'─' * 50}")

            with open(raw_reviews_file, "r", encoding="utf-8") as infile:
                reader = csv.DictReader(infile)

                for idx, row in enumerate(reader, start=1):
                    app_name    = row.get("app_name", "")
                    review_text = row.get("review_text", "").strip()
                    review_id   = row.get("review_id", "")
                    rating      = row.get("rating", "")
                    date        = row.get("date", "")
                    app_id      = row.get("app_id", "")
                    # Support both old (no category) and new raw files
                    category    = row.get("category", CATEGORY_MAP.get(app_name, "unknown"))

                    # Skip if already processed (resume mode)
                    if review_id in existing_ids:
                        stats["skipped_existing"] += 1
                        continue

                    # Skip empty reviews
                    if not review_text:
                        stats["skipped_low_quality"] += 1
                        continue

                    # Quality filter
                    if _is_low_quality(review_text):
                        stats["skipped_low_quality"] += 1
                        if idx % 200 == 0:
                            print(f"  [{idx}/{playstore_total}] skipped (low quality)")
                        continue

                    print(f"  [{idx}/{playstore_total}] {app_name} | {review_id[:12]}...")

                    raw_response = ""
                    try:
                        user_prompt = build_playstore_prompt(app_name, review_text)
                        raw_response = _call_groq_with_backoff(groq_client, SYSTEM_PROMPT, user_prompt)
                        extracted = json.loads(_clean_json(raw_response))

                        _write_extracted_row(
                            writer, "playstore", app_name, app_id, category,
                            review_id, rating, date, review_text, extracted
                        )
                        stats["success"] += 1

                    except json.JSONDecodeError as e:
                        print(f"    [ERROR] JSON parse failed: {e}")
                        stats["errors"] += 1
                        err_writer.writerow({
                            "row_number": idx, "source": "playstore",
                            "app_name": app_name, "review_id": review_id,
                            "error": f"json_parse: {e}",
                            "raw_response": raw_response[:500],
                        })

                    except Exception as e:
                        short = str(e)[:120]
                        print(f"    [ERROR] API failed: {short}")
                        stats["errors"] += 1
                        err_writer.writerow({
                            "row_number": idx, "source": "playstore",
                            "app_name": app_name, "review_id": review_id,
                            "error": short,
                            "raw_response": raw_response[:300],
                        })

                    stats["processed"] += 1

                    # Progress checkpoint every 50 successful extractions
                    if stats["processed"] % 50 == 0:
                        pct = idx / playstore_total * 100
                        print(f"\n  *** CHECKPOINT: row {idx}/{playstore_total} ({pct:.1f}%) "
                              f"| Extracted={stats['success']} | Errors={stats['errors']} "
                              f"| Skipped={stats['skipped_low_quality']} ***\n")

                    outfile.flush()
                    errfile.flush()
                    time.sleep(BASE_DELAY)

        # ── PHASE 2: Reddit Posts ─────────────────────────────────────────────
        if os.path.exists(raw_reddit_file) and reddit_total > 0:
            print(f"\n{'─' * 50}")
            print(f"  PHASE 2: Reddit Posts ({reddit_total} total)")
            print(f"{'─' * 50}")

            with open(raw_reddit_file, "r", encoding="utf-8") as infile:
                reader = csv.DictReader(infile)

                for idx, row in enumerate(reader, start=1):
                    post_id   = row.get("post_id", "")
                    title     = row.get("title", "").strip()
                    body      = row.get("body", "").strip()
                    comment_1 = row.get("comment_1", "").strip()
                    comment_2 = row.get("comment_2", "").strip()
                    comment_3 = row.get("comment_3", "").strip()
                    date      = row.get("date", "")
                    subreddit = row.get("subreddit", "")

                    # Skip if already processed
                    if post_id in existing_ids:
                        stats["skipped_existing"] += 1
                        continue

                    # Combine all text for extraction
                    combined_text = f"{title}. {body}"
                    comments_text = " | ".join(c for c in [comment_1, comment_2, comment_3] if c)

                    # Quality filter (more lenient for Reddit — titles alone can be meaningful)
                    if len(title.strip()) < 10 and len(body.strip()) < 10:
                        stats["skipped_low_quality"] += 1
                        continue

                    print(f"  [Reddit {idx}/{reddit_total}] r/{subreddit} | {post_id}")

                    raw_response = ""
                    try:
                        user_prompt = build_reddit_prompt(title, body, comments_text)
                        raw_response = _call_groq_with_backoff(groq_client, SYSTEM_PROMPT, user_prompt)
                        extracted = json.loads(_clean_json(raw_response))

                        # Reddit-specific: get app name from AI extraction
                        app_mentioned = extracted.get("app_mentioned", "unknown")
                        category = CATEGORY_MAP.get(app_mentioned, "unknown")

                        _write_extracted_row(
                            writer, "reddit", app_mentioned, "", category,
                            post_id, "", date, combined_text[:500], extracted
                        )
                        stats["success"] += 1

                    except json.JSONDecodeError as e:
                        print(f"    [ERROR] JSON parse failed: {e}")
                        stats["errors"] += 1
                        err_writer.writerow({
                            "row_number": idx, "source": "reddit",
                            "app_name": "", "review_id": post_id,
                            "error": f"json_parse: {e}",
                            "raw_response": raw_response[:500],
                        })

                    except Exception as e:
                        short = str(e)[:120]
                        print(f"    [ERROR] API failed: {short}")
                        stats["errors"] += 1
                        err_writer.writerow({
                            "row_number": idx, "source": "reddit",
                            "app_name": "", "review_id": post_id,
                            "error": short,
                            "raw_response": raw_response[:300],
                        })

                    stats["processed"] += 1
                    outfile.flush()
                    errfile.flush()
                    time.sleep(BASE_DELAY)

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 57}")
    print(f"  [Extractor] DONE")
    print(f"  Extracted:         {stats['success']}")
    print(f"  Errors:            {stats['errors']}")
    print(f"  Skipped (resume):  {stats['skipped_existing']}")
    print(f"  Skipped (quality): {stats['skipped_low_quality']}")
    print(f"{'=' * 57}")
    return stats


if __name__ == "__main__":
    extract_behaviors()
