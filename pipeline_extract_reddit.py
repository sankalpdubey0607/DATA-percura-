"""
pipeline_extract_reddit.py — Fix 5: Reddit extraction
Input:  raw_reddit.csv
Output: data/processed/extracted_reddit.csv
"""
import csv, json, os, re, time, random, sys
from groq import Groq, RateLimitError as GroqRateLimitError
from dotenv import load_dotenv

load_dotenv()

# Force unbuffered stdout on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

OUT_DIR = os.path.join("data", "processed")
INPUT_FILE = "raw_reddit.csv"
OUTPUT_FILE = os.path.join(OUT_DIR, "extracted_reddit.csv")
ERRORS_FILE = os.path.join(OUT_DIR, "reddit_errors.csv")

GROQ_MODEL = "llama-3.1-8b-instant"
BASE_DELAY = 3.0
MAX_RETRIES = 6
BACKOFF_BASE = 8

SYSTEM_PROMPT = """You are a behavioral data extraction specialist for an Indian startup called Percura. Your job is to extract behavioral signals from real Indian app user reviews.

You must output ONLY valid JSON with no explanation, no markdown, no preamble. Just the raw JSON object.

Be aggressive in extraction. Even short reviews have signals. A review saying "payment fail" clearly indicates drop_off_stage = payment and friction_type = technical_error. Do not default to unknown when there is any signal at all."""


def build_reddit_prompt(subreddit, title, body, comment_1):
    body_trunc = (body or "")[:500]
    comment_trunc = (comment_1 or "")[:300]
    return f"""Extract behavioral signals from this Reddit discussion about an Indian app or digital service experience.

Subreddit: {subreddit}
Post title: {title}
Post content: {body_trunc}
Top comment: {comment_trunc}

This is a user describing a real experience with a digital product or service in India. Extract the same JSON structure as app reviews.

Additionally extract:
"product_mentioned": the app or service being discussed. (If the post is a political rant, news, social commentary, or entirely unrelated to a digital product, set this to "irrelevant").
"issue_category": "privacy" | "payment" | "trust" | "ux" | "scam" | "support" | "technical" | "other"

STRICT ENUM COMPLIANCE: For drop_off_stage, friction_type, emotion, trust_signal, confidence, you MUST ONLY use the EXACT allowed values. NEVER copy raw text into these fields. If unsure, use "unknown". If product_mentioned is "irrelevant", output "unknown" or empty for all fields.

Return ONLY this JSON, no other text:
{{
  "drop_off_stage": "",
  "friction_type": [],
  "emotion": "",
  "gave_up": null,
  "trust_signal": "",
  "effort_complained": null,
  "language": "",
  "literacy_hint": "",
  "device_hint": "",
  "income_hint": "",
  "region_hint": "",
  "key_quote": "",
  "confidence": "",
  "product_mentioned": "",
  "issue_category": ""
}}

ALLOWED VALUES:
drop_off_stage: "sign_up" | "login" | "onboarding" | "form_fill" | "payment" | "verification" | "home_screen" | "feature_use" | "checkout" | "support" | "unknown"
friction_type (array): "too_many_steps" | "confusing_language" | "trust_issue" | "slow_loading" | "form_too_long" | "verification_failed" | "price_shock" | "hidden_charges" | "forced_signup" | "no_hindi_support" | "technical_error" | "poor_support" | "data_privacy_fear"
emotion: "frustrated" | "confused" | "betrayed" | "bored" | "scared" | "disappointed" | "angry" | "neutral" | "happy"
gave_up: true | false | null
trust_signal: "trusted" | "did_not_trust" | "lost_trust" | "never_had_trust" | "none" | "unknown"
effort_complained: true | false | null
language: "english" | "hindi" | "hinglish" | "other"
literacy_hint: "high" | "medium" | "low" | "unknown"
device_hint: "iphone" | "high_android" | "mid_android" | "basic_android" | "unknown"
income_hint: "high" | "middle" | "low" | "unknown"
region_hint: "metro" | "tier_2" | "tier_3" | "rural" | "unknown"
key_quote: most revealing phrase, under 15 words
confidence: "high" | "medium" | "low" """


OUTPUT_FIELDS = [
    "source", "subreddit", "post_id", "title", "date", "url",
    "product_mentioned", "issue_category",
    "drop_off_stage", "friction_type", "emotion", "gave_up", "trust_signal",
    "effort_complained", "language", "literacy_hint", "device_hint",
    "income_hint", "region_hint", "key_quote", "confidence", "api_used",
]
ERROR_FIELDS = ["index", "post_id", "error", "raw_response"]


def clean_json(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def call_groq(client, system, user):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            chat = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                max_tokens=500, temperature=0,
            )
            return chat.choices[0].message.content
        except GroqRateLimitError as e:
            last_err = e
            wait = BACKOFF_BASE * (2 ** attempt) + random.uniform(1, 5)
            print(f"    [429] attempt {attempt+1}/{MAX_RETRIES}, sleeping {wait:.0f}s...")
            time.sleep(wait)
        except Exception as e:
            raise e
    raise Exception(f"Rate limit after {MAX_RETRIES} retries: {last_err}")


def run_reddit_extraction():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found.")
        sys.exit(1)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    rows = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    total = len(rows)

    # Check for already extracted
    already_done = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                already_done.add(row.get("post_id", ""))

    if already_done:
        print(f"\n[RESUME] {len(already_done)} Reddit posts already done")
        wmode, wheader = "a", False
    else:
        wmode, wheader = "w", True

    print(f"\n{'='*55}")
    print(f"  REDDIT EXTRACTION: {total} posts")
    print(f"  Already done: {len(already_done)}")
    print(f"  Remaining: {total - len(already_done)}")
    print(f"{'='*55}\n")

    stats = {"success": 0, "errors": 0, "skipped": 0}

    with (
        open(OUTPUT_FILE, wmode, newline="", encoding="utf-8") as outf,
        open(ERRORS_FILE, "w", newline="", encoding="utf-8") as errf,
    ):
        writer = csv.DictWriter(outf, fieldnames=OUTPUT_FIELDS)
        err_writer = csv.DictWriter(errf, fieldnames=ERROR_FIELDS)
        if wheader:
            writer.writeheader()
        err_writer.writeheader()

        for idx, row in enumerate(rows):
            post_id = row.get("post_id", "")
            if post_id in already_done:
                stats["skipped"] += 1
                continue

            title = row.get("title", "").strip()
            body = row.get("body", "").strip()
            comment_1 = row.get("comment_1", "").strip()
            subreddit = row.get("subreddit", "")
            date = row.get("date", "")
            url = row.get("url", "")

            # Skip posts with no content
            if len(title) < 10 and len(body) < 10:
                stats["skipped"] += 1
                continue

            print(f"  [{idx+1}/{total}] r/{subreddit} | {title[:50]}...")

            raw_resp = ""
            try:
                prompt = build_reddit_prompt(subreddit, title, body, comment_1)
                raw_resp = call_groq(client, SYSTEM_PROMPT, prompt)
                ex = json.loads(clean_json(raw_resp))

                friction = ex.get("friction_type", [])
                if isinstance(friction, list):
                    friction = "|".join(friction)

                product_mentioned = ex.get("product_mentioned", "unknown")
                if isinstance(product_mentioned, str) and product_mentioned.lower() == "irrelevant":
                    stats["skipped"] += 1
                    continue

                writer.writerow({
                    "source": "reddit", "subreddit": subreddit,
                    "post_id": post_id, "title": title,
                    "date": date, "url": url,
                    "product_mentioned": ex.get("product_mentioned", "unknown"),
                    "issue_category": ex.get("issue_category", "other"),
                    "drop_off_stage": ex.get("drop_off_stage", "unknown"),
                    "friction_type": friction,
                    "emotion": ex.get("emotion", "unknown"),
                    "gave_up": ex.get("gave_up"),
                    "trust_signal": ex.get("trust_signal", "unknown"),
                    "effort_complained": ex.get("effort_complained"),
                    "language": ex.get("language", "unknown"),
                    "literacy_hint": ex.get("literacy_hint", "unknown"),
                    "device_hint": ex.get("device_hint", "unknown"),
                    "income_hint": ex.get("income_hint", "unknown"),
                    "region_hint": ex.get("region_hint", "unknown"),
                    "key_quote": ex.get("key_quote", ""),
                    "confidence": ex.get("confidence", "low"),
                    "api_used": "groq",
                })
                stats["success"] += 1

            except json.JSONDecodeError as e:
                stats["errors"] += 1
                err_writer.writerow({
                    "index": idx, "post_id": post_id,
                    "error": f"json_parse: {e}",
                    "raw_response": raw_resp[:500],
                })
            except Exception as e:
                stats["errors"] += 1
                err_writer.writerow({
                    "index": idx, "post_id": post_id,
                    "error": str(e)[:200],
                    "raw_response": raw_resp[:300],
                })

            outf.flush()
            time.sleep(BASE_DELAY)

    print(f"\n{'='*55}")
    print(f"  REDDIT EXTRACTION COMPLETE")
    print(f"  Success:  {stats['success']}")
    print(f"  Errors:   {stats['errors']}")
    print(f"  Skipped:  {stats['skipped']}")
    print(f"  Output:   {OUTPUT_FILE}")
    print(f"{'='*55}")
    return stats


if __name__ == "__main__":
    run_reddit_extraction()
