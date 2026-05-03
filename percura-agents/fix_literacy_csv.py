"""
fix_literacy_csv.py — One-time fix for extracted_behaviors.csv
Reclassifies 249 polluted literacy_hint rows and saves back.
"""
import pandas as pd
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "extracted_behaviors.csv")

df = pd.read_csv(CSV_PATH, encoding="utf-8")
df["literacy_hint"] = df["literacy_hint"].astype(str).str.lower().str.strip()

location_values = {"tier_2", "metro", "tier_3", "rural"}
polluted = df["literacy_hint"].isin(location_values)
print(f"Polluted rows before fix: {polluted.sum()}")

# Reclassify using language as proxy
lang = df.loc[polluted, "language"].astype(str).str.lower().str.strip()
new_lit = pd.Series("medium", index=df.loc[polluted].index)
new_lit[lang.isin(["english"])] = "high"
new_lit[lang.isin(["hindi", "regional", "tamil", "telugu", "kannada",
                    "malayalam", "bengali", "marathi", "gujarati"])] = "low"
new_lit[lang.isin(["hinglish"])] = "medium"

df.loc[polluted, "literacy_hint"] = new_lit

# Verify
polluted_after = df["literacy_hint"].isin(location_values).sum()
print(f"Polluted rows after fix: {polluted_after}")
print(f"\nNew literacy_hint distribution:")
print(df["literacy_hint"].value_counts())

# Save
df.to_csv(CSV_PATH, index=False, encoding="utf-8")
print(f"\nSaved clean CSV to {CSV_PATH}")
