import requests
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

def test_reddit():
    print("Testing Reddit search...")
    url = "https://www.reddit.com/r/india/search.json"
    params = {"q": "app problem", "restrict_sr": "1", "limit": 2}
    
    resp = requests.get(url, headers=HEADERS, params=params)
    print("Search status:", resp.status_code)
    
    if resp.status_code == 200:
        data = resp.json()
        posts = data.get("data", {}).get("children", [])
        print(f"Found {len(posts)} posts.")
        for p in posts:
            post_id = p["data"]["id"]
            print("Post:", p["data"]["title"])
            
            # Fetch comments
            print(f"Testing comments for {post_id}...")
            c_url = f"https://www.reddit.com/r/india/comments/{post_id}.json"
            c_resp = requests.get(c_url, headers=HEADERS)
            print("Comments status:", c_resp.status_code)
            if c_resp.status_code == 200:
                c_data = c_resp.json()
                comments = c_data[1]["data"]["children"]
                print(f"Got {len(comments)} top-level comments.")
            time.sleep(2)
            
test_reddit()
