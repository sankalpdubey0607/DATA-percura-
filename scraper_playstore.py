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
import sys
from datetime import datetime

# Force UTF-8 stdout on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

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

REVIEWS_PER_RATING = 300          # per star-rating per app
TARGET_RATINGS     = [1, 2, 3]    # 1* 2* 3* -- captures rage, disappointment, AND nuanced friction
OUTPUT_FILE        = "raw_reviews.csv"
FIELDNAMES         = ["app_name", "app_id", "category", "review_id", "rating", "date", "review_text"]


# ── Main scraper ─────────────────────────────────────────────────────────────
def _scrape_rating(app_id: str, app_name: str, target_rating: int,
                   max_reviews: int) -> list:
    """Scrape reviews for a single (app, rating) pair using filter_score_with."""
    collected = []
    continuation_token = None
    attempts = 0
    empty_batches = 0
    MAX_EMPTY = 3

    while len(collected) < max_reviews and attempts < 10:
        try:
            batch, continuation_token = reviews(
                app_id,
                lang="en",
                country="in",
                sort=Sort.NEWEST,
                count=min(200, max_reviews - len(collected)),
                filter_score_with=target_rating,
                continuation_token=continuation_token,
            )
            collected.extend(batch)
            print(f"      fetched {len(batch):3d} | total {len(collected)}/{max_reviews}")

            if len(batch) == 0:
                empty_batches += 1
                if empty_batches >= MAX_EMPTY:
                    print(f"      {MAX_EMPTY} empty batches -- stopping for {app_name} {target_rating}-star")
                    break
            else:
                empty_batches = 0

            if not continuation_token:
                print(f"      no more pages")
                break
            time.sleep(1)
        except Exception as e:
            attempts += 1
            print(f"      [WARN] attempt {attempts}: {e}")
            time.sleep(3)

    return collected[:max_reviews]


def scrape_playstore(output_file: str = OUTPUT_FILE,
                     reviews_per_rating: int = REVIEWS_PER_RATING,
                     target_ratings: list = None) -> int:
    """Scrape 1* + 2* + 3* Play Store reviews and write to CSV.
    Returns total rows written."""

    if target_ratings is None:
        target_ratings = TARGET_RATINGS

    total_written = 0
    seen_ids = set()  # dedup across rating tiers

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for app_name, app_id in APPS.items():
            category = CATEGORY_MAP.get(app_name, "unknown")
            app_total = 0

            for star in target_ratings:
                print(f"\n[Play Store] {app_name} | {star}-star reviews (target={reviews_per_rating}) ...")
                batch = _scrape_rating(app_id, app_name, star, reviews_per_rating)

                written = 0
                for r in batch:
                    rid = r.get("reviewId", "")
                    if rid in seen_ids:
                        continue
                    seen_ids.add(rid)

                    raw_date = r.get("at", "")
                    if isinstance(raw_date, datetime):
                        date_str = raw_date.strftime("%Y-%m-%d")
                    else:
                        date_str = str(raw_date)

                    writer.writerow({
                        "app_name":    app_name,
                        "app_id":      app_id,
                        "category":    category,
                        "review_id":   rid,
                        "rating":      r.get("score", ""),
                        "date":        date_str,
                        "review_text": r.get("content", "").replace("\n", " ").strip(),
                    })
                    written += 1

                app_total += written
                print(f"      OK {written} unique {star}-star reviews")
                time.sleep(2)

            total_written += app_total
            print(f"  OK {app_name} total: {app_total} reviews")

    print(f"\n[Play Store] Done. Total reviews written: {total_written} -> {output_file}")
    return total_written


if __name__ == "__main__":
    scrape_playstore()
