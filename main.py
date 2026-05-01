"""
main.py
-------
Percura Behavioral Data Extraction Pipeline
Runs all 4 steps in sequence:
  1. Scrape Google Play Store reviews  -> raw_reviews.csv
  2. Scrape Reddit posts               -> raw_reddit.csv
  3. AI behavioral extraction          -> extracted_behaviors.csv
  4. Summary report                    -> summary_report.txt + summary_data.json

Usage:
    python main.py                  # run all steps
    python main.py --skip-scrape    # skip scrapers (use existing CSVs)
    python main.py --only-extract   # skip scrapers, run extraction + summary
    python main.py --only-summary   # only regenerate the summary report
"""

import argparse
import os
import sys
import time

from dotenv import load_dotenv

# Force UTF-8 output on Windows to avoid UnicodeEncodeError with box-drawing chars
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Load env vars immediately
load_dotenv()


def _check_env() -> bool:
    """Validate required environment variables are set."""
    required = {
        "GROQ_API_KEY": "Groq API key (for behavioral extraction)",
    }
    missing = [f"  {var}  ({desc})" for var, desc in required.items()
               if not os.getenv(var) or os.getenv(var, "").startswith("your_")]
    if missing:
        print("\n[ERROR] Missing or placeholder environment variables in .env:")
        for m in missing:
            print(m)
        print("\nCreate a .env file in this folder based on .env.example")
        return False
    return True


def _banner(step: int, title: str) -> None:
    print(f"\n{'=' * 57}")
    print(f"  STEP {step}: {title}")
    print(f"{'=' * 57}")


# ── Step runners ──────────────────────────────────────────────────────────────
def run_playstore_scraper():
    from scraper_playstore import scrape_playstore
    return scrape_playstore()


def run_reddit_scraper():
    from scraper_reddit import scrape_reddit
    return scrape_reddit()


def run_extractor():
    from extractor import extract_behaviors
    return extract_behaviors()


def run_summary():
    from summariser import summarise
    return summarise()


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Percura behavioral data extraction pipeline"
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip Play Store and Reddit scrapers; use existing CSVs",
    )
    parser.add_argument(
        "--skip-reddit",
        action="store_true",
        help="Skip only the Reddit scraper",
    )
    parser.add_argument(
        "--only-extract",
        action="store_true",
        help="Skip scrapers, run extraction + summary only (same as --skip-scrape)",
    )
    parser.add_argument(
        "--only-summary",
        action="store_true",
        help="Only regenerate the summary report from existing extracted_behaviors.csv",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("\n" + "=" * 57)
    print("  PERCURA -- Behavioral Signal Extraction Pipeline v2")
    print("=" * 57)
    print(f"  Working directory: {os.getcwd()}")
    print(f"  AI Model: Groq (llama-3.3-70b-versatile)")

    # Validate env vars (always needed for extraction step)
    if not _check_env():
        sys.exit(1)

    start_time = time.time()

    if args.only_summary:
        _banner(4, "Summary Report")
        run_summary()
        elapsed = time.time() - start_time
        print(f"\n  Summary generated in {elapsed:.1f}s")
        return

    skip_scrape = args.skip_scrape or args.only_extract

    # ── Step 1: Play Store ────────────────────────────────────────────────────
    if not skip_scrape:
        _banner(1, "Google Play Store Scraper")
        try:
            count = run_playstore_scraper()
            print(f"\n  [OK] Play Store done -- {count} reviews collected")
        except Exception as e:
            print(f"\n  [FAIL] Play Store scraper failed: {e}")
            print("    Continuing to next step ...")
    else:
        print("\n[Step 1] Skipped (using existing raw_reviews.csv)")

    # ── Step 2: Reddit ────────────────────────────────────────────────────────
    if not skip_scrape and not args.skip_reddit:
        _banner(2, "Reddit Scraper")
        try:
            count = run_reddit_scraper()
            print(f"\n  [OK] Reddit done -- {count} posts collected")
        except Exception as e:
            print(f"\n  [FAIL] Reddit scraper failed: {e}")
            print("    Continuing to next step ...")
    else:
        print("\n[Step 2] Skipped")

    # ── Step 3: AI Extraction ─────────────────────────────────────────────────
    if not os.path.exists("raw_reviews.csv"):
        print("\n[ERROR] raw_reviews.csv not found. Cannot run extraction. Exiting.")
        sys.exit(1)

    _banner(3, "AI Behavioral Extraction (Groq)")
    try:
        stats = run_extractor()
        print(f"\n  [OK] Extraction done -- {stats.get('success', 0)} rows extracted, "
              f"{stats.get('errors', 0)} errors, "
              f"{stats.get('skipped_low_quality', 0)} skipped (low quality)")
    except Exception as e:
        print(f"\n  [FAIL] Extraction failed: {e}")
        sys.exit(1)

    # ── Step 4: Summary ───────────────────────────────────────────────────────
    _banner(4, "Summary Report")
    try:
        run_summary()
    except Exception as e:
        print(f"\n  [FAIL] Summary failed: {e}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 57}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  raw_reviews.csv           -- raw Play Store reviews")
    print(f"  raw_reddit.csv            -- raw Reddit posts")
    print(f"  extracted_behaviors.csv   -- AI-extracted signals")
    print(f"  errors.csv                -- extraction errors")
    print(f"  summary_report.txt        -- human-readable summary")
    print(f"  summary_data.json         -- machine-readable for Person B")
    print(f"{'=' * 57}\n")


if __name__ == "__main__":
    main()
