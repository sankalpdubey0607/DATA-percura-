"""Finish remaining 251 reviews using 3 working keys (skip exhausted key-0)."""
import csv, json, os, re, time, random, sys, threading
from groq import Groq, RateLimitError as GroqRateLimitError
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')

OUT_DIR = os.path.join("data", "processed")
INPUT_FILE = os.path.join(OUT_DIR, "reviews_to_extract.csv")
OUTPUT_FILE = os.path.join(OUT_DIR, "extracted_behaviors.csv")
GROQ_MODEL = "llama-3.1-8b-instant"

KEYS = [
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4"),
]

SYSTEM_PROMPT = """You are a behavioral data extraction specialist for an Indian startup called Percura. Your job is to extract behavioral signals from real Indian app user reviews.
You must output ONLY valid JSON with no explanation, no markdown, no preamble. Just the raw JSON object.
Be aggressive in extraction. Even short reviews have signals."""

def build_prompt(app_name, review_text, rating):
    return f"""Extract behavioral signals from this Indian app review.
App name: {app_name}
Review text: {review_text}
Star rating: {rating} out of 5

CORE RULES:
- Rating 1-2 = gave_up=true, emotion=frustrated/angry
- Rating 3 = gave_up=false, disappointed
- Words: delete/uninstall=gave_up=true, payment fail=payment+technical_error, OTP/verify=verification, slow/crash=slow_loading, fraud/scam=betrayed+never_had_trust

DEVICE/INCOME/REGION - ALWAYS INFER, NEVER "unknown":
- device_hint: iPhone mentions=iphone, Samsung S/OnePlus/Pixel=high_android, Redmi/Realme/Poco/Vivo=mid_android, Jio phone=basic_android. No clue? PhonePe/Paytm/Zomato=mid_android, Meesho=basic_android/mid_android, UrbanCompany=mid_android/high_android, BYJU's=mid_android
- income_hint: Meesho=low/middle, PhonePe/Paytm=middle, BYJU's=middle, Zomato=middle, UrbanCompany=middle/high. Hinglish+budget app=low
- region_hint: Clean English=metro, Hinglish=metro/tier_2, Hindi-dominant=tier_2, Broken grammar=tier_2/tier_3. Meesho=tier_2/tier_3, UrbanCompany=metro, Zomato=metro/tier_2

Return ONLY JSON:
{{"drop_off_stage":"","friction_type":[],"emotion":"","gave_up":null,"trust_signal":"","effort_complained":null,"language":"","literacy_hint":"","device_hint":"","income_hint":"","region_hint":"","key_quote":"","confidence":""}}

drop_off_stage: sign_up|login|onboarding|form_fill|payment|verification|home_screen|feature_use|checkout|support|unknown
friction_type: too_many_steps|confusing_language|trust_issue|slow_loading|form_too_long|verification_failed|price_shock|hidden_charges|forced_signup|no_hindi_support|technical_error|poor_support|data_privacy_fear
emotion: frustrated|confused|betrayed|bored|scared|disappointed|angry|neutral|happy
device_hint: iphone|high_android|mid_android|basic_android (NEVER unknown)
income_hint: high|middle|low (NEVER unknown)
region_hint: metro|tier_2|tier_3|rural (NEVER unknown)
confidence: high|medium|low"""

OUTPUT_FIELDS = [
    "app_name","app_id","review_id","rating","date","review_text",
    "drop_off_stage","friction_type","emotion","gave_up","trust_signal",
    "effort_complained","language","literacy_hint","device_hint",
    "income_hint","region_hint","key_quote","confidence","api_used",
]

def clean_json(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"): raw = raw[4:]
    return raw.strip()

def call_groq(client, prompt, key_idx):
    for attempt in range(4):
        try:
            chat = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}],
                response_format={"type":"json_object"}, max_tokens=500, temperature=0,
            )
            return chat.choices[0].message.content
        except GroqRateLimitError:
            wait = 10 * (2**attempt) + random.uniform(1,5)
            print(f"  [KEY-{key_idx} 429] retry {attempt+1}, wait {wait:.0f}s", flush=True)
            time.sleep(wait)
    raise Exception(f"Key-{key_idx} exhausted")

write_lock = threading.Lock()

def worker(key_idx, api_key, items, writer, outf):
    client = Groq(api_key=api_key)
    ok = 0
    for idx, row in items:
        try:
            prompt = build_prompt(row["app_name"], row.get("review_text",""), row.get("rating",""))
            raw = call_groq(client, prompt, key_idx)
            ex = json.loads(clean_json(raw))
            friction = ex.get("friction_type",[])
            if isinstance(friction, list): friction = "|".join(friction)
            out = {
                "app_name":row["app_name"],"app_id":row.get("app_id",""),
                "review_id":row["review_id"],"rating":row.get("rating",""),
                "date":row.get("date",""),"review_text":row.get("review_text",""),
                "drop_off_stage":ex.get("drop_off_stage","unknown"),
                "friction_type":friction,"emotion":ex.get("emotion","unknown"),
                "gave_up":ex.get("gave_up"),"trust_signal":ex.get("trust_signal","unknown"),
                "effort_complained":ex.get("effort_complained"),
                "language":ex.get("language","unknown"),
                "literacy_hint":ex.get("literacy_hint","unknown"),
                "device_hint":ex.get("device_hint","unknown"),
                "income_hint":ex.get("income_hint","unknown"),
                "region_hint":ex.get("region_hint","unknown"),
                "key_quote":ex.get("key_quote",""),
                "confidence":ex.get("confidence","low"),
                "api_used":f"groq-key{key_idx}",
            }
            with write_lock:
                writer.writerow(out)
                outf.flush()
            ok += 1
        except Exception as e:
            print(f"  [KEY-{key_idx}] ERROR: {str(e)[:80]}", flush=True)
        time.sleep(2.5)
    print(f"  [KEY-{key_idx}] DONE: {ok} extracted", flush=True)

# Load done IDs
done_ids = set()
with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
    for r in csv.DictReader(f): done_ids.add(r.get("review_id",""))

# Load remaining
rows = list(csv.DictReader(open(INPUT_FILE,"r",encoding="utf-8")))
work = [(i,r) for i,r in enumerate(rows) if r.get("review_id","") not in done_ids]
print(f"\n{'='*50}")
print(f"  FINISHING: {len(work)} remaining reviews")
print(f"  Keys: {len(KEYS)} (skipping exhausted key-0)")
print(f"  ETA: ~{len(work)/(len(KEYS)*20):.0f} min")
print(f"{'='*50}\n")

# Split work round-robin
key_work = [[] for _ in KEYS]
for i, item in enumerate(work):
    key_work[i % len(KEYS)].append(item)

with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as outf:
    writer = csv.DictWriter(outf, fieldnames=OUTPUT_FIELDS)
    threads = []
    for i, key in enumerate(KEYS):
        t = threading.Thread(target=worker, args=(i, key, key_work[i], writer, outf))
        threads.append(t)
        t.start()
        time.sleep(0.3)
    for t in threads: t.join()

# Final count
with open(OUTPUT_FILE,"r",encoding="utf-8") as f:
    final = sum(1 for _ in f) - 1
print(f"\n{'='*50}")
print(f"  COMPLETE: {final}/1209 total extracted")
print(f"{'='*50}")
