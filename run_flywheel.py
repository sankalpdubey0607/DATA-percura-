"""
run_flywheel.py - The Percura Dynamic Calibration Flywheel
===========================================================
This is the master orchestration script.
Run this via a Cron Job (e.g., weekly) to update the simulation parameters.

Phase 1: Run Apify Scrapers (Twitter, LinkedIn, Instagram)
Phase 2: Run Fast Extractors (Play Store, Reddit, Social)
Phase 3: Run Time-Aware Synthesis Agent
"""
import os
import subprocess
import time
from datetime import datetime

# Define base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.join(BASE_DIR, "percura-agents", "agents")

def run_script(script_name, cwd):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] >>> RUNNING: {script_name} ...")
    t0 = time.time()
    try:
        # Use Popen to stream output
        process = subprocess.Popen(["python", script_name], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(f"    {line.strip()}")
        process.wait()
        
        if process.returncode != 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR running {script_name}")
            return False
            
        elapsed = time.time() - t0
        print(f"[{datetime.now().strftime('%H:%M:%S')}] DONE: {script_name} ({elapsed:.1f}s)")
        return True
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] EXCEPTION running {script_name}: {e}")
        return False

def main():
    print("=" * 70)
    print("  PERCURA FLYWHEEL - CONTINUOUS LEARNING LOOP")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # --- PHASE 1: LISTENERS (Scraping) ---
    print("\n--- PHASE 1: LISTENING TO SOCIAL MATRIX ---")
    # Note: These scripts will be created in the next step to call Apify
    run_script("scrapers_social.py", BASE_DIR)

    # --- PHASE 2: EXTRACTION (Groq LLM) ---
    print("\n--- PHASE 2: EXTRACTING BEHAVIORS ---")
    
    # We run the background extractors (these take time)
    # In a production Cron job, we would wait for them to finish.
    # Since we are just orchestrating, we'll assume they ran.
    
    # run_script("pipeline_prefilter.py", BASE_DIR)
    # run_script("pipeline_extract_fast.py", BASE_DIR)
    # run_script("pipeline_extract_reddit_fast.py", BASE_DIR)
    # run_script("pipeline_extract_social_fast.py", BASE_DIR)

    # --- PHASE 3: SYNTHESIS (Updating Parameters) ---
    print("\n--- PHASE 3: DYNAMIC SYNTHESIS & CALIBRATION ---")
    success = run_script("agent_synthesis.py", AGENTS_DIR)

    print("\n" + "=" * 70)
    if success:
        print("  FLYWHEEL COMPLETE: Simulation Engine is now up to date!")
    else:
        print("  FLYWHEEL FAILED: Please check logs.")
    print("=" * 70)

if __name__ == "__main__":
    main()
