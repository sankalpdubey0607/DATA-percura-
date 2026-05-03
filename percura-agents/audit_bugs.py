import json

d = json.load(open('config/archetype_params.json'))

# Count distinct combos
combos = {}
for p in d['patterns']:
    drop = str(p['params']['primary_drop_off_stage'])
    if drop not in combos:
        combos[drop] = []
    combos[drop].append(p['id'])

print(f"Distinct drop_off combos: {len(combos)}\n")
for combo, ids in combos.items():
    print(f"Combo ({len(ids)} archetypes): {combo}")
    print(f"  Examples: {ids[:3]}\n")

# Verify literacy_hint in CSV
import pandas as pd
df = pd.read_csv('../data/processed/extracted_behaviors.csv')
lh = df['literacy_hint'].astype(str).str.lower().str.strip()
location_vals = lh[lh.isin(['tier_2','metro','tier_3','rural'])]
print(f"\n=== BUG 1 CHECK ===")
print(f"Polluted literacy_hint rows: {len(location_vals)}")
print(f"literacy_hint distribution:\n{lh.value_counts()}")

# Trust prior range
tvals = [p['params']['trust_prior'] for p in d['patterns']]
print(f"\n=== TRUST PRIOR ===")
print(f"Range: {min(tvals)} to {max(tvals)}")
print(f"Unique values: {len(set(tvals))}")
