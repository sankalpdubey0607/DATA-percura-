import csv

with open('data/processed/extracted_behaviors.csv', 'r', encoding='utf-8') as f:
    reader = list(csv.DictReader(f))
    for r in reader[-10:]:
        print(f"Stage: {r['drop_off_stage']:<12} | Friction: {r['friction_type']:<20} | Emotion: {r['emotion']:<12} | Gave Up: {r['gave_up']}")
