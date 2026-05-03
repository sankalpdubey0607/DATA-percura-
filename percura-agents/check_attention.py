import json

d = json.load(open('config/archetype_params.json'))
vals = [p['params']['attention_budget'] for p in d['patterns']]
print(f"Attention budget range: {min(vals)} to {max(vals)}")

low = [p for p in d['patterns'] if p['params']['attention_budget'] < 10]
print(f"Patterns with attention_budget < 10: {len(low)}")
for p in low:
    print(f"  {p['id']}: {p['params']['attention_budget']}")

# Trust prior range
tvals = [p['params']['trust_prior'] for p in d['patterns']]
print(f"\nTrust prior range: {min(tvals)} to {max(tvals)}")

# Drop off stage diversity
stages = set()
for p in d['patterns']:
    drop = p['params']['primary_drop_off_stage']
    if isinstance(drop, dict):
        stages.update(drop.values())
    else:
        stages.add(drop)
print(f"\nDrop-off stages in use: {stages}")

# Friction trigger diversity
all_frictions = set()
for p in d['patterns']:
    all_frictions.update(p['params']['top_friction_triggers'])
print(f"Friction triggers in use: {all_frictions}")
