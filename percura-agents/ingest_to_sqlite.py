"""
ingest_to_sqlite.py
===================
Converts the ~1M row nvidia_personas_clean.csv into an indexed SQLite database.
This prevents RAM exhaustion (OOM) when the downstream Nvidia simulation engine
attempts to query and sample personas for simulation runs.

Indexes are created on key demographic and behavioral columns for rapid querying.
"""
import os
import sqlite3
import pandas as pd
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "nvidia_personas_clean.csv")
DB_PATH = os.path.join(BASE_DIR, "data", "processed", "nvidia_personas.db")
CHUNK_SIZE = 50_000

def ingest_to_db():
    print("=" * 60)
    print("  PERCURA DATABASE INGESTION")
    print(f"  Source: {CSV_PATH}")
    print(f"  Target: {DB_PATH}")
    print("=" * 60)

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run Sorter and Auditor first.")
        return

    # Remove old DB if it exists
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("  Removed existing database.")

    conn = sqlite3.connect(DB_PATH)
    
    # Get total rows for tqdm
    # Quick row count
    print("  Counting rows in CSV...")
    total_rows = sum(1 for _ in open(CSV_PATH, 'r', encoding='utf-8')) - 1
    
    print(f"\n  Ingesting {total_rows:,} rows into SQLite in chunks of {CHUNK_SIZE:,}...")
    
    chunks = pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE, encoding="utf-8")
    
    for i, chunk in enumerate(tqdm(chunks, total=(total_rows//CHUNK_SIZE)+1, desc="  Ingesting")):
        # Write to SQLite table 'personas'
        chunk.to_sql("personas", conn, if_exists="append" if i > 0 else "replace", index=False)
        
    print("\n  Creating indices for fast simulation querying...")
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX idx_archetype ON personas (archetype);")
    cursor.execute("CREATE INDEX idx_age_band ON personas (age_band);")
    cursor.execute("CREATE INDEX idx_region ON personas (region_mapped);")
    cursor.execute("CREATE INDEX idx_literacy ON personas (literacy_mapped);")
    cursor.execute("CREATE INDEX idx_occupation ON personas (occupation_mapped);")
    conn.commit()
    conn.close()
    
    print("=" * 60)
    print("  SUCCESS: SQLite Database built and indexed.")
    print("  Simulation engine can now query rows directly without loading to RAM:")
    print("  e.g., SELECT * FROM personas WHERE archetype = '...' ORDER BY RANDOM() LIMIT 10")
    print("=" * 60)

if __name__ == "__main__":
    ingest_to_db()
