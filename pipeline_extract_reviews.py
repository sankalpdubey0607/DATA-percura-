"""
pipeline_extract_reviews.py — Fix 3+4: Improved prompt + all 6 apps + checkpointing
Input:  data/processed/reviews_to_extract.csv
Output: data/processed/extracted_behaviors.csv
        data/processed/extraction_checkpoint.json
"""
import csv, json, os, re, time, random, sys
from datetime import datetime
from groq import Groq, RateLimitError as GroqRateLimitError
from dotenv import load_dotenv

load_dotenv()

# Force unbuffered stdout on Windows so prints show up in real-time
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join("data", "processed")
INPUT_FILE = os.path.join(OUT_DIR, "reviews_to_extract.csv")
OUTPUT_FILE = os.path.join(OUT_DIR, "extracted_behaviors.csv")
CHECKPOINT_FILE = os.path.join(OUT_DIR, "extraction_checkpoint.json")
ERRORS_FILE = os.path.join(OUT_DIR, "extraction_errors.csv")

GROQ_MODEL = "llama-3.1-8b-instant"
BASE_DELAY = 3.0
MAX_RETRIES = 6
BACKOFF_BASE = 8
CHECKPOINT_INTERVAL = 100

# ── Improved system prompt (Fix 3) ──────────────────────────────────────────
SYSTEM_PROMPT = """You are a behavioral data extraction specialist for an Indian startup called Percura. Your job is to extract behavioral signals from real Indian app user reviews.

You must output ONLY valid JSON with no explanation, no markdown, no preamble. Just the raw JSON object.

Be aggressive in extraction. Even short reviews have signals. A review saying "payment fail" clearly indicates drop_off_stage = payment and friction_type = technical_error. Do not default to unknown when there is any signal at all."""


def build_review_prompt(app_name, review_text, rating):
    return f"""Extract behavioral signals from this Indian app review.

App name: {app_name}
Review text: {review_text}
Star rating: {rating} out of 5

IMPORTANT EXTRACTION RULES:
- Rating 1-2 = almost always gave_up = true
- Rating 3 = gave_up = false, but frustrated
- Rating 4-5 with complaint words = gave_up = false, trust_signal = "lost_trust" or "did_not_trust"
- Hinglish/Hindi text: translate meaning then extract
- If rating is 1 or 2 and no other info: emotion = "frustrated", gave_up = true, trust_signal = "lost_trust"
- Words like "delete", "uninstall", "hataya", "hata diya" = gave_up = true
- Words like "payment fail", "payment nahi hua" = drop_off_stage = "payment", friction_type includes "technical_error"
- Words like "OTP", "verify", "verification" = drop_off_stage = "verification"
- Words like "slow", "hang", "crash", "lagging" = friction_type includes "slow_loading"
- Words like "bakwaas", "bekar", "worst", "pathetic" = emotion = "angry"
- Words like "fraud", "scam", "cheat", "thagi" = trust_signal = "never_had_trust", emotion = "betrayed"
- EMOJI OR SHORT REVIEWS: Infer emotion from emojis (😡 = angry, 😞 = disappointed, 👍 = happy). Infer gave_up from rating (1-2 = true, 4-5 = false). 
- STRICT ENUM COMPLIANCE: For drop_off_stage, friction_type, emotion, trust_signal, confidence, you MUST ONLY use the EXACT allowed values. NEVER copy raw text into these fields. If unsure, use "unknown".

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
  "confidence": ""
}}

ALLOWED VALUES:

drop_off_stage: "sign_up" | "login" | "onboarding" | "form_fill" | "payment" | "verification" | "home_screen" | "feature_use" | "checkout" | "support" | "unknown"
DEFAULT TO "feature_use" if something broke but no specific stage mentioned

friction_type (array, pick all that apply): "too_many_steps" | "confusing_language" | "trust_issue" | "slow_loading" | "form_too_long" | "verification_failed" | "price_shock" | "hidden_charges" | "forced_signup" | "no_hindi_support" | "technical_error" | "poor_support" | "data_privacy_fear"
DEFAULT TO ["technical_error"] if something failed but type is unclear

emotion: "frustrated" | "confused" | "betrayed" | "bored" | "scared" | "disappointed" | "angry" | "neutral" | "happy"
Use rating to guide: 1-2 = frustrated/angry, 3 = disappointed, 4-5 = neutral/happy

gave_up: true | false | null
true if rating 1 and complaint present. false if rating 4-5 even with complaint. null only if truly impossible to determine

trust_signal: "trusted" | "did_not_trust" | "lost_trust" | "never_had_trust" | "none" | "unknown"

effort_complained: true | false | null

language: "english" | "hindi" | "hinglish" | "gujarati" | "tamil" | "telugu" | "marathi" | "kannada" | "other"

literacy_hint: "high" | "medium" | "low" | "unknown"
device_hint: "iphone" | "high_android" | "mid_android" | "basic_android" | "unknown"
income_hint: "high" | "middle" | "low" | "unknown"
region_hint: "metro" | "tier_2" | "tier_3" | "rural" | "unknown"

key_quote: The single most behaviorally revealing phrase from the review text, under 15 words. If review is too short, use the full text. Never leave empty if there is any text.

confidence: "high" | "medium" | "low" """


OUTPUT_FIELDS = [
    "app_name", "app_id", "review_id", "rating", "date", "review_text",
    "drop_off_stage", "friction_type", "emotion", "gave_up", "trust_signal",
    "effort_complained", "language", "literacy_hint", "device_hint",
    "income_hint", "region_hint", "key_quote", "confidence", "api_used",
]
ERROR_FIELDS = ["index", "app_name", "review_id", "error", "raw_response"]


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
                max_tokens=500,
                temperature=0,
            )
            return chat.choices[0].message.content
        except GroqRateLimitError as e:
            last_err = e
            wait = BACKOFF_BASE * (2 ** attempt) + random.uniform(1, 5)
            print(f"    [429] attempt {attempt+1}/{MAX_RETRIES}, sleeping {wait:.0f}s...", flush=True)
            time.sleep(wait)
        except Exception as e:
            raise e
    raise Exception(f"Rate limit after {MAX_RETRIES} retries: {last_err}")


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return None


def save_checkpoint(index, app_name):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "last_processed_index": index,
            "app": app_name,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)


def progress_bar(current, total, width=20):
    filled = int(width * current / total) if total > 0 else 0
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total}"


def run_extraction():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found. Run pipeline_prefilter.py first.")
        sys.exit(1)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Load all rows
    rows = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    total = len(rows)

    # Check for checkpoint / resume
    checkpoint = load_checkpoint()
    start_idx = 0
    already_extracted = set()

    if os.path.exists(OUTPUT_FILE) and checkpoint:
        start_idx = checkpoint["last_processed_index"] + 1
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                already_extracted.add(row.get("review_id", ""))
        print(f"\n[RESUME] From index {start_idx} ({len(already_extracted)} already done)")
        write_mode, write_header = "a", False
    else:
        write_mode, write_header = "w", True
        start_idx = 0

    # Group by app for progress display
    from collections import Counter
    app_counts = Counter(r["app_name"] for r in rows)
    app_done = Counter()
    if already_extracted:
        for r in rows[:start_idx]:
            if r["review_id"] in already_extracted:
                app_done[r["app_name"]] += 1

    print(f"\n{'='*55}")
    print(f"  EXTRACTION: {total} reviews across {len(app_counts)} apps")
    print(f"  Model: {GROQ_MODEL}")
    print(f"  Rate: {BASE_DELAY}s/call = {60/BASE_DELAY:.0f} RPM")
    if start_idx > 0:
        print(f"  Resuming from index {start_idx}")
    print(f"{'='*55}\n")

    stats = {"success": 0, "errors": 0, "skipped": len(already_extracted)}
    current_app = ""

    with (
        open(OUTPUT_FILE, write_mode, newline="", encoding="utf-8") as outf,
        open(ERRORS_FILE, "w", newline="", encoding="utf-8") as errf,
    ):
        writer = csv.DictWriter(outf, fieldnames=OUTPUT_FIELDS)
        err_writer = csv.DictWriter(errf, fieldnames=ERROR_FIELDS)
        if write_header:
            writer.writeheader()
        err_writer.writeheader()

        for idx in range(start_idx, total):
            row = rows[idx]
            app_name = row.get("app_name", "")
            review_text = row.get("review_text", "").strip()
            review_id = row.get("review_id", "")
            rating = row.get("rating", "")
            date = row.get("date", "")
            app_id = row.get("app_id", "")

            if review_id in already_extracted:
                stats["skipped"] += 1
                continue

            # Print app progress header when app changes
            if app_name != current_app:
                current_app = app_name
                done = app_done.get(app_name, 0)
                app_total = app_counts[app_name]
                print(f"\n  [{app_name}] {progress_bar(done, app_total)}")

            # Call API
            raw_resp = ""
            try:
                prompt = build_review_prompt(app_name, review_text, rating)
                raw_resp = call_groq(client, SYSTEM_PROMPT, prompt)
                ex = json.loads(clean_json(raw_resp))

                friction = ex.get("friction_type", [])
                if isinstance(friction, list):
                    friction = "|".join(friction)

                writer.writerow({
                    "app_name": app_name, "app_id": app_id,
                    "review_id": review_id, "rating": rating,
                    "date": date, "review_text": review_text,
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
                app_done[app_name] = app_done.get(app_name, 0) + 1

            except json.JSONDecodeError as e:
                stats["errors"] += 1
                err_writer.writerow({
                    "index": idx, "app_name": app_name,
                    "review_id": review_id,
                    "error": f"json_parse: {e}",
                    "raw_response": raw_resp[:500],
                })
            except Exception as e:
                stats["errors"] += 1
                err_writer.writerow({
                    "index": idx, "app_name": app_name,
                    "review_id": review_id,
                    "error": str(e)[:200],
                    "raw_response": raw_resp[:300],
                })

            outf.flush()
            processed = stats["success"] + stats["errors"]

            # Progress display
            if processed % 10 == 0:
                done = app_done.get(app_name, 0)
                app_total = app_counts[app_name]
                pct = idx / total * 100
                print(f"  [{app_name:15s}] {progress_bar(done, app_total)} | overall {pct:.0f}%")

            # Checkpoint every N rows
            if processed % CHECKPOINT_INTERVAL == 0 and processed > 0:
                save_checkpoint(idx, app_name)
                print(f"\n  >>> CHECKPOINT saved at index {idx} ({processed} processed) <<<\n")

            time.sleep(BASE_DELAY)

    # Final checkpoint
    save_checkpoint(total - 1, current_app)

    print(f"\n{'='*55}")
    print(f"  REVIEW EXTRACTION COMPLETE")
    print(f"{'='*55}")
    print(f"  Success:  {stats['success']}")
    print(f"  Errors:   {stats['errors']}")
    print(f"  Skipped:  {stats['skipped']}")

    # Print per-app summary
    print(f"\n  Per-app results:")
    for app in app_counts:
        done = app_done.get(app, 0)
        app_total = app_counts[app]
        status = "done" if done >= app_total else "partial"
        print(f"    [{app:15s}] {progress_bar(done, app_total)} {status}")

    print(f"\n  Output: {OUTPUT_FILE}")
    print(f"{'='*55}")
    return stats


if __name__ == "__main__":
    run_extraction()
