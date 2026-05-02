import csv

r = list(csv.DictReader(open('raw_reviews.csv', 'r', encoding='utf-8')))
rd = list(csv.DictReader(open('raw_reddit.csv', 'r', encoding='utf-8')))
print(f"Raw reviews: {len(r)}")
print(f"Raw reddit: {len(rd)}")
print(f"Total: {len(r) + len(rd)}")

short = sum(1 for x in r if len((x.get("review_text") or "").strip()) < 15)
print(f"Reviews under 15 chars (previously filtered out): {short}")

empty = sum(1 for x in r if len((x.get("review_text") or "").strip()) < 2)
print(f"Reviews under 2 chars (will still be filtered): {empty}")
