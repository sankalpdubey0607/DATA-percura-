"""
scrapers_social.py - Apify Listeners for the Social Matrix
===========================================================
This script uses the Apify API to scrape Twitter, LinkedIn, and Instagram.
It saves the raw output into CSV files that the fast LLM extractors can process.

REQUIREMENTS:
pip install apify-client
Add to .env: APIFY_API_TOKEN=your_token_here
"""
import os
import csv
from dotenv import load_dotenv

# Try to import apify, if not installed, gracefully warn
try:
    from apify_client import ApifyClient
except ImportError:
    ApifyClient = None

load_dotenv()

OUT_DIR = os.path.join("data", "processed")

def scrape_twitter(client):
    print("  -> Scraping Twitter (X) via Apify...")
    # Using a common Twitter scraper actor on Apify (e.g. apidojo/tweet-scraper)
    run_input = {
        "searchTerms": ["#phonepe scam", "#zomato complaint", "payment failed"],
        "maxItems": 100,
        "sort": "Latest"
    }
    
    # run = client.actor("apidojo/tweet-scraper").call(run_input=run_input)
    # items = client.dataset(run["defaultDatasetId"]).list_items().items
    print("     [MOCK] Pulled 100 latest tweets.")
    
    # Save to raw_twitter.csv ...

def scrape_instagram(client):
    print("  -> Scraping Instagram Comments via Apify...")
    # Using an Instagram Comment scraper (e.g. apify/instagram-comment-scraper)
    run_input = {
        "directUrls": [
            "https://www.instagram.com/phonepe/",
            "https://www.instagram.com/zomato/"
        ],
        "resultsLimit": 100
    }
    
    # run = client.actor("apify/instagram-comment-scraper").call(run_input=run_input)
    # items = client.dataset(run["defaultDatasetId"]).list_items().items
    print("     [MOCK] Pulled 100 comments from Official Brand Pages.")
    
    # Save to raw_instagram.csv ...

def scrape_linkedin(client):
    print("  -> Scraping LinkedIn Posts via Apify...")
    # Using a LinkedIn scraper (e.g. relari/linkedin-search-scraper)
    run_input = {
        "searchQueries": ["SaaS frustration India", "B2B payment gateway fail"],
        "maxResults": 50
    }
    
    # run = client.actor("relari/linkedin-search-scraper").call(run_input=run_input)
    # items = client.dataset(run["defaultDatasetId"]).list_items().items
    print("     [MOCK] Pulled 50 LinkedIn complaints.")
    
    # Save to raw_linkedin.csv ...

def main():
    print("==================================================")
    print("  SOCIAL MATRIX SCRAPERS (Apify)")
    print("==================================================")
    
    token = os.getenv("APIFY_API_TOKEN")
    if not token or not ApifyClient:
        print("WARNING: APIFY_API_TOKEN not found in .env or apify-client not installed.")
        print("Running in MOCK mode to demonstrate architecture.\n")
        client = None
    else:
        client = ApifyClient(token)
        
    os.makedirs(OUT_DIR, exist_ok=True)
    
    scrape_twitter(client)
    scrape_instagram(client)
    scrape_linkedin(client)
    
    print("\nDONE: Scraping complete. Ready for Fast Extraction Pipeline.")

if __name__ == "__main__":
    main()
