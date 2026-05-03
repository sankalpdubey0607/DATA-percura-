"""
pipeline_extract_fast.py — Multi-key parallel extraction
Uses 4 Groq API keys with round-robin to 4x throughput.
Resumes from existing extracted_behaviors.csv automatically.
"""
import csv, json, os, re, time, random, sys, threading
from datetime import datetime
from collections import Counter
from groq import Groq, RateLimitError as GroqRateLimitError
from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join("data", "processed")
INPUT_FILE = os.path.join(OUT_DIR, "reviews_to_extract.csv")
OUTPUT_FILE = os.path.join(OUT_DIR, "extracted_behaviors.csv")
ERRORS_FILE = os.path.join(OUT_DIR, "extraction_errors.csv")

GROQ_MODEL = "llama-3.1-8b-instant"
MAX_RETRIES = 4
BACKOFF_BASE = 10

# ── All API keys from .env ───────────────────────────────────────────────────
API_KEYS = [
    os.getenv("GROQ_API_KEY"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4"),
]
API_KEYS = [k for k in API_KEYS if k]  # filter None

# ── Prompts (same improved prompt) ───────────────────────────────────────────
SYSTEM_PROMPT = """You are a behavioral data extraction specialist for an Indian startup called Percura. Your job is to extract behavioral signals from real Indian app user reviews.

You must output ONLY valid JSON with no explanation, no markdown, no preamble. Just the raw JSON object.

Be aggressive in extraction. Even short reviews have signals. A review saying "payment fail" clearly indicates drop_off_stage = payment and friction_type = technical_error. Do not default to unknown when there is any signal at all."""


def build_review_prompt(app_name, review_text, rating):
    return f"""Extract behavioral signals from this Indian app review.

App name: {app_name}
Review text: {review_text}
Star rating: {rating} out of 5

CORE EXTRACTION RULES:
- Rating 1-2 = almost always gave_up = true
- Rating 3 = gave_up = false, but frustrated or disappointed
- Rating 4-5 with complaint words = gave_up = false, trust_signal = "lost_trust" or "did_not_trust"
- Hinglish/Hindi text: translate meaning then extract
- If rating is 1 or 2 and no other info: emotion = "frustrated", gave_up = true, trust_signal = "lost_trust"
- Words like "delete", "uninstall", "hataya", "hata diya" = gave_up = true
- Words like "payment fail", "payment nahi hua" = drop_off_stage = "payment", friction_type includes "technical_error"
- Words like "OTP", "verify", "verification" = drop_off_stage = "verification"
- Words like "slow", "hang", "crash", "lagging" = friction_type includes "slow_loading"
- Words like "bakwaas", "bekar", "worst", "pathetic" = emotion = "angry"
- Words like "fraud", "scam", "cheat", "thagi" = trust_signal = "never_had_trust", emotion = "betrayed"
- EMOJI OR SHORT REVIEWS: Infer emotion from emojis. Infer gave_up from rating (1-2 = true, 4-5 = false).
- STRICT ENUM COMPLIANCE: Only use exact allowed values. NEVER copy raw text into enum fields.

DEVICE / INCOME / REGION INFERENCE RULES (CRITICAL):
You MUST infer device_hint, income_hint, and region_hint even when NOT explicitly stated.

DEVICE_HINT inference:
- Mentions iPhone/iOS/Apple = "iphone"
- Mentions Samsung Galaxy S/Note/Fold, OnePlus, Pixel = "high_android"
- Mentions Redmi/Realme/Poco/Samsung M/A series/Vivo/Oppo = "mid_android"
- Mentions Jio phone, Nokia basic = "basic_android"
- Complaints about RAM, storage, phone heating, lagging = "mid_android" or "basic_android"
- App crash/slow on their device = "mid_android" (most common in India)
- If no device clue, use app-based inference:
  * PhonePe/Paytm/Zomato user = "mid_android" (mass market apps)
  * BYJU's user = "mid_android" (student/parent demographic)
  * Meesho user = "basic_android" or "mid_android" (budget shoppers)
  * UrbanCompany user = "mid_android" or "high_android" (service-oriented)
- NEVER leave as "unknown"

INCOME_HINT inference:
- Mentions "expensive", "too costly", price complaints = "low" or "middle"
- Mentions "refund", "money wasted" = "low" or "middle"
- App-based: Meesho = "low"/"middle", PhonePe/Paytm = "middle", BYJU's = "middle", Zomato = "middle", UrbanCompany = "middle"/"high"
- Hinglish/broken English + budget app = "low"
- Good English + premium service = "middle" or "high"
- NEVER leave as "unknown"

REGION_HINT inference:
- Clean English, tech-savvy = "metro"
- Hinglish, casual slang = "metro" or "tier_2"
- Hindi-dominant = "tier_2"
- Broken grammar, phonetic spelling = "tier_2" or "tier_3"
- App-based: Meesho = "tier_2"/"tier_3", PhonePe/Paytm = "tier_2", Zomato = "metro"/"tier_2", UrbanCompany = "metro", BYJU's = "tier_2"
- NEVER leave as "unknown"

Return ONLY this JSON:
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
friction_type (array): "too_many_steps" | "confusing_language" | "trust_issue" | "slow_loading" | "form_too_long" | "verification_failed" | "price_shock" | "hidden_charges" | "forced_signup" | "no_hindi_support" | "technical_error" | "poor_support" | "data_privacy_fear"
emotion: "frustrated" | "confused" | "betrayed" | "bored" | "scared" | "disappointed" | "angry" | "neutral" | "happy"
gave_up: true | false | null
trust_signal: "trusted" | "did_not_trust" | "lost_trust" | "never_had_trust" | "none" | "unknown"
effort_complained: true | false | null
language: "english" | "hindi" | "hinglish" | "gujarati" | "tamil" | "telugu" | "marathi" | "kannada" | "other"
literacy_hint: "high" | "medium" | "low" | "unknown"
device_hint: "iphone" | "high_android" | "mid_android" | "basic_android" -- NEVER "unknown"
income_hint: "high" | "middle" | "low" -- NEVER "unknown"
region_hint: "metro" | "tier_2" | "tier_3" | "rural" -- NEVER "unknown"
key_quote: most revealing phrase, under 15 words
confidence: "high" | "medium" | "low" """


OUTPUT_FIELDS = [
    "app_name", "app_id", "review_id", "rating", "date", "review_text",
    "drop_off_stage", "friction_type", "emotion", "gave_up", "trust_signal",
    "effort_complained", "language", "literacy_hint", "device_hint",
    "income_hint", "region_hint", "key_quote", "confidence", "api_used",
]


def clean_json(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def call_groq(client, system, user, key_idx):
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
            print(f"    [KEY-{key_idx} 429] attempt {attempt+1}/{MAX_RETRIES}, sleeping {wait:.0f}s...", flush=True)
            time.sleep(wait)
        except Exception as e:
            raise e
    raise Exception(f"Rate limit after {MAX_RETRIES} retries on key-{key_idx}: {last_err}")


# ── Worker function for each key ─────────────────────────────────────────────
write_lock = threading.Lock()
stats = {"success": 0, "errors": 0}
stats_lock = threading.Lock()


def worker(key_idx, api_key, work_items, writer, err_writer, outf, total_work):
    """Process assigned reviews using one API key."""
    client = Groq(api_key=api_key)
    local_success = 0
    local_errors = 0

    for item in work_items:
        idx, row = item
        app_name = row.get("app_name", "")
        review_text = row.get("review_text", "").strip()
        review_id = row.get("review_id", "")
        rating = row.get("rating", "")
        date = row.get("date", "")
        app_id = row.get("app_id", "")

        raw_resp = ""
        try:
            prompt = build_review_prompt(app_name, review_text, rating)
            raw_resp = call_groq(client, SYSTEM_PROMPT, prompt, key_idx)
            ex = json.loads(clean_json(raw_resp))

            friction = ex.get("friction_type", [])
            if isinstance(friction, list):
                friction = "|".join(friction)

            out_row = {
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
                "api_used": f"groq-key{key_idx}",
            }

            with write_lock:
                writer.writerow(out_row)
                outf.flush()
            local_success += 1

        except json.JSONDecodeError as e:
            local_errors += 1
            with write_lock:
                err_writer.writerow({
                    "index": idx, "app_name": app_name,
                    "review_id": review_id,
                    "error": f"json_parse: {e}",
                    "raw_response": raw_resp[:500],
                })
        except Exception as e:
            local_errors += 1
            with write_lock:
                err_writer.writerow({
                    "index": idx, "app_name": app_name,
                    "review_id": review_id,
                    "error": str(e)[:200],
                    "raw_response": raw_resp[:300],
                })

        with stats_lock:
            stats["success"] += (1 if local_success > (stats["success"] + stats["errors"] - local_errors) else 0)
            done = stats["success"] + stats["errors"]

        # Print progress every 5 items per worker
        if local_success % 5 == 0 and local_success > 0:
            with stats_lock:
                total_done = sum(1 for _ in [])  # just for sync
            print(f"  [KEY-{key_idx}] {local_success} done | {app_name}", flush=True)

        # Stagger requests to stay under 30 RPM per key
        time.sleep(2.5)

    with stats_lock:
        stats["success"] += local_success
        stats["errors"] += local_errors

    print(f"  [KEY-{key_idx}] FINISHED: {local_success} success, {local_errors} errors", flush=True)
    return local_success, local_errors


def run_fast_extraction():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found.")
        sys.exit(1)

    # Load all input rows
    rows = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    total = len(rows)

    # Load already-extracted IDs
    already_done = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                already_done.add(row.get("review_id", ""))

    # Build work queue (skip already done)
    work = []
    for idx, row in enumerate(rows):
        if row.get("review_id", "") not in already_done:
            work.append((idx, row))

    print(f"\n{'='*60}")
    print(f"  FAST PARALLEL EXTRACTION")
    print(f"  Total reviews: {total}")
    print(f"  Already done:  {len(already_done)}")
    print(f"  Remaining:     {len(work)}")
    print(f"  API keys:      {len(API_KEYS)}")
    print(f"  Model:         {GROQ_MODEL}")
    print(f"  Effective RPM: ~{len(API_KEYS) * 20} (4 keys x 20 RPM)")
    est_min = len(work) / (len(API_KEYS) * 20)
    print(f"  ETA:           ~{est_min:.0f} min")
    print(f"{'='*60}\n")

    if len(work) == 0:
        print("  Nothing to do! All reviews already extracted.")
        return

    # Split work across keys (round-robin)
    key_work = [[] for _ in API_KEYS]
    for i, item in enumerate(work):
        key_work[i % len(API_KEYS)].append(item)

    for i, kw in enumerate(key_work):
        print(f"  KEY-{i}: {len(kw)} reviews assigned")

    # Open output files
    write_mode = "a" if already_done else "w"
    write_header = not already_done

    ERROR_FIELDS = ["index", "app_name", "review_id", "error", "raw_response"]

    with (
        open(OUTPUT_FILE, write_mode, newline="", encoding="utf-8") as outf,
        open(ERRORS_FILE, "w", newline="", encoding="utf-8") as errf,
    ):
        writer = csv.DictWriter(outf, fieldnames=OUTPUT_FIELDS)
        err_writer = csv.DictWriter(errf, fieldnames=ERROR_FIELDS)
        if write_header:
            writer.writeheader()
        err_writer.writeheader()

        # Launch threads
        threads = []
        t_start = time.time()

        for i, api_key in enumerate(API_KEYS):
            t = threading.Thread(
                target=worker,
                args=(i, api_key, key_work[i], writer, err_writer, outf, len(work)),
                daemon=True,
            )
            threads.append(t)
            t.start()
            time.sleep(0.5)  # stagger starts

        # Monitor progress
        while any(t.is_alive() for t in threads):
            time.sleep(15)
            done_now = 0
            if os.path.exists(OUTPUT_FILE):
                with open(OUTPUT_FILE, "r", encoding="utf-8") as check:
                    done_now = sum(1 for _ in check) - 1  # minus header
            elapsed = time.time() - t_start
            rpm = done_now / (elapsed / 60) if elapsed > 0 else 0
            remaining = total - done_now
            eta = remaining / rpm if rpm > 0 else 999
            print(f"\n  >> PROGRESS: {done_now}/{total} ({done_now/total*100:.0f}%) | "
                  f"{rpm:.0f} RPM | ETA: {eta:.0f} min\n", flush=True)

        # Wait for all threads
        for t in threads:
            t.join()

    elapsed = time.time() - t_start

    # Final count
    final_count = 0
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        final_count = sum(1 for _ in f) - 1

    print(f"\n{'='*60}")
    print(f"  EXTRACTION COMPLETE")
    print(f"  Total extracted: {final_count}/{total}")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"  Avg RPM: {(final_count - len(already_done)) / (elapsed/60):.0f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_fast_extraction()
