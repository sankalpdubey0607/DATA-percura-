"""
scraper_playstore.py
--------------------
Scrapes reviews from the Google Play Store for a list of apps.
Output: raw_reviews.csv

Each app is tagged with a product category so Person B
can group behavioral signals by vertical when building archetypes.
"""

import csv
import time
import os
from datetime import datetime

from google_play_scraper import reviews, Sort


# ── Category mapping (all apps from the context doc) ─────────────────────────
CATEGORY_MAP = {
    # Phase 1 apps (active)
    "PhonePe":       "fintech",
    "Paytm":         "fintech",
    "Meesho":        "ecommerce",
    "BYJU's":        "edtech",
    "Zomato":        "food_delivery",
    "UrbanCompany":  "services",
    # Future expansion
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

# ── App targets (Phase 1: 6 apps, 2+ categories) ────────────────────────────
APPS = {
    "PhonePe":      "com.phonepe.app",
    "Paytm":        "net.one97.paytm",
    "Meesho":       "com.meesho.supply",
    "BYJU's":       "com.byjus.thelearningapp",
    "Zomato":       "com.application.zomato",
    "UrbanCompany": "com.urbanclap.urbanclap",
}

REVIEWS_PER_APP = 500
OUTPUT_FILE     = "raw_reviews.csv"
FIELDNAMES      = ["app_name", "app_id", "category", "review_id", "rating", "date", "review_text"]


# ── Main scraper ─────────────────────────────────────────────────────────────
def scrape_playstore(output_file: str = OUTPUT_FILE, reviews_per_app: int = REVIEWS_PER_APP) -> int:
    """Scrape Play Store reviews and write to CSV. Returns total rows written."""

    total_written = 0

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for app_name, app_id in APPS.items():
            category = CATEGORY_MAP.get(app_name, "unknown")
            print(f"\n[Play Store] Scraping '{app_name}' ({app_id}) | category={category} ...")

            collected = []
            continuation_token = None
            attempts = 0
            empty_batches = 0
            MAX_EMPTY = 3  # stop if we get 3 consecutive empty batches

            while len(collected) < reviews_per_app and attempts < 10:
                try:
                    batch, continuation_token = reviews(
                        app_id,
                        lang="en",
                        country="in",
                        sort=Sort.NEWEST,
                        count=min(200, reviews_per_app - len(collected)),
                        continuation_token=continuation_token,
                    )
                    collected.extend(batch)
                    print(f"  -> fetched {len(batch)} | total so far: {len(collected)}")

                    if len(batch) == 0:
                        empty_batches += 1
                        if empty_batches >= MAX_EMPTY:
                            print(f"  -> {MAX_EMPTY} empty batches in a row. Stopping for {app_name}.")
                            break
                    else:
                        empty_batches = 0

                    if not continuation_token:
                        print(f"  -> no more pages for {app_name}")
                        break
                    time.sleep(1)
                except Exception as e:
                    attempts += 1
                    print(f"  [WARN] Error fetching {app_name} (attempt {attempts}): {e}")
                    time.sleep(3)

            # Trim to exact target
            collected = collected[:reviews_per_app]

            for r in collected:
                # Normalise date to ISO string
                raw_date = r.get("at", "")
                if isinstance(raw_date, datetime):
                    date_str = raw_date.strftime("%Y-%m-%d")
                else:
                    date_str = str(raw_date)

                writer.writerow({
                    "app_name":    app_name,
                    "app_id":      app_id,
                    "category":    category,
                    "review_id":   r.get("reviewId", ""),
                    "rating":      r.get("score", ""),
                    "date":        date_str,
                    "review_text": r.get("content", "").replace("\n", " ").strip(),
                })

            total_written += len(collected)
            print(f"  ✔ Written {len(collected)} reviews for {app_name}")
            time.sleep(2)

    print(f"\n[Play Store] Done. Total reviews written: {total_written} → {output_file}")
    return total_written


if __name__ == "__main__":
    scrape_playstore()
