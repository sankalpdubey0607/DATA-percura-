"""
scraper_reddit.py
-----------------
Scrapes posts + top comments from targeted Indian subreddits.
Output: raw_reddit.csv

(API-Free Version: Uses Reddit's raw JSON endpoints via requests to bypass the need for an official API key.)
"""

import csv
import os
import time
import requests
from datetime import datetime

# ── Config ───────────────────────────────────────────────────────────────────
SUBREDDITS = ["india", "IndiaInvestments", "IndianApps", "bangalore"]

SEARCH_QUERIES = [
    "app problem",
    "app delete",
    "app confusing",
    "app fraud",
]

OUTPUT_FILE = "raw_reddit.csv"
POSTS_PER_QUERY = 25          # per subreddit × per query
TOP_COMMENTS    = 3

FIELDNAMES = [
    "source", "subreddit", "post_id", "title", "body",
    "comment_1", "comment_2", "comment_3",
    "upvotes", "date", "url",
]

# Using a standard browser User-Agent helps avoid Reddit's automated rate limits
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def _ts_to_date(ts) -> str:
    try:
        return datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d")
    except Exception:
        return str(ts)

def fetch_top_comments(post_id: str, subreddit: str) -> list[str]:
    """Fetch the top-level comments for a specific post."""
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return []
        
        data = resp.json()
        comments_list = data[1].get("data", {}).get("children", [])
        
        # Filter for actual comments (ignore 'more' placeholders)
        valid_comments = []
        for c in comments_list:
            if c.get("kind") == "t1":
                body = c["data"].get("body", "").replace("\n", " ").strip()
                if body and body != "[deleted]":
                    valid_comments.append(body)
                    
        return valid_comments[:TOP_COMMENTS]
    except Exception:
        return []

# ── Main scraper ─────────────────────────────────────────────────────────────
def scrape_reddit(output_file: str = OUTPUT_FILE) -> int:
    seen_ids = set()
    total_written = 0

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for subreddit_name in SUBREDDITS:
            print(f"\n[Reddit] Scraping r/{subreddit_name} …")

            for query in SEARCH_QUERIES:
                print(f"  → searching: \"{query}\"")
                
                search_url = f"https://www.reddit.com/r/{subreddit_name}/search.json"
                params = {
                    "q": query,
                    "restrict_sr": "1",
                    "sort": "relevance",
                    "t": "year",
                    "limit": POSTS_PER_QUERY
                }

                try:
                    resp = requests.get(search_url, headers=HEADERS, params=params, timeout=15)
                    if resp.status_code == 429:
                        print("  ⚠ Rate limited by Reddit. Sleeping for 10 seconds...")
                        time.sleep(10)
                        resp = requests.get(search_url, headers=HEADERS, params=params, timeout=15)
                        
                    if resp.status_code != 200:
                        print(f"  ⚠ Failed to search (Status {resp.status_code}).")
                        continue
                        
                    results = resp.json().get("data", {}).get("children", [])
                    
                    batch_count = 0
                    for submission in results:
                        data = submission.get("data", {})
                        post_id = data.get("id")
                        
                        if not post_id or post_id in seen_ids:
                            continue
                        seen_ids.add(post_id)

                        # Wait slightly to respect Reddit's servers
                        time.sleep(1.5)
                        
                        comments = fetch_top_comments(post_id, subreddit_name)
                        while len(comments) < TOP_COMMENTS:
                            comments.append("")

                        writer.writerow({
                            "source":     "reddit",
                            "subreddit":  subreddit_name,
                            "post_id":    post_id,
                            "title":      data.get("title", "").replace("\n", " ").strip(),
                            "body":       data.get("selftext", "").replace("\n", " ").strip(),
                            "comment_1":  comments[0],
                            "comment_2":  comments[1],
                            "comment_3":  comments[2],
                            "upvotes":    data.get("score", 0),
                            "date":       _ts_to_date(data.get("created_utc", 0)),
                            "url":        f"https://reddit.com{data.get('permalink', '')}",
                        })
                        batch_count  += 1
                        total_written += 1

                    print(f"    ✔ {batch_count} new posts extracted")
                    time.sleep(2)

                except Exception as e:
                    print(f"  ⚠ Error on r/{subreddit_name} query='{query}': {e}")
                    time.sleep(3)

    print(f"\n[Reddit] Done. Total posts written: {total_written} → {output_file}")
    return total_written


if __name__ == "__main__":
    scrape_reddit()
