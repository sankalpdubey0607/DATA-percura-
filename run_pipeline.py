"""
run_pipeline.py — Master orchestrator for Percura data pipeline v2
Runs all phases in order:
  Phase 1: Pre-filter (pipeline_prefilter.py)
  Phase 2: Extract reviews (pipeline_extract_reviews.py)
  Phase 3: Extract Reddit (pipeline_extract_reddit.py)
  Phase 4: Merge + Report (pipeline_merge_report.py)

Usage:
  python run_pipeline.py          # Run all phases
  python run_pipeline.py 2        # Run from phase 2 onward
  python run_pipeline.py 3        # Run from phase 3 onward (e.g., after review extraction finishes)
"""
import sys, time
from datetime import datetime

def main():
    start_phase = 1
    if len(sys.argv) > 1:
        try:
            start_phase = int(sys.argv[1])
        except ValueError:
            pass

    print("=" * 60)
    print("  PERCURA DATA PIPELINE v2")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Starting from phase: {start_phase}")
    print("=" * 60)

    t0 = time.time()

    # ── Phase 1: Pre-filter ──
    if start_phase <= 1:
        print("\n\n" + "#" * 60)
        print("  PHASE 1: PRE-FILTER")
        print("#" * 60)
        from pipeline_prefilter import run_prefilter
        passed = run_prefilter()
        print(f"\n  -> {passed} reviews ready for extraction")

    # ── Phase 2: Extract reviews ──
    if start_phase <= 2:
        print("\n\n" + "#" * 60)
        print("  PHASE 2: EXTRACT REVIEWS (this will take ~70 min)")
        print("#" * 60)
        from pipeline_extract_reviews import run_extraction
        stats = run_extraction()

    # ── Phase 3: Extract Reddit ──
    if start_phase <= 3:
        print("\n\n" + "#" * 60)
        print("  PHASE 3: EXTRACT REDDIT (~9 min)")
        print("#" * 60)
        from pipeline_extract_reddit import run_reddit_extraction
        stats = run_reddit_extraction()

    # ── Phase 4: Merge + Report ──
    if start_phase <= 4:
        print("\n\n" + "#" * 60)
        print("  PHASE 4: MERGE + REPORT")
        print("#" * 60)
        from pipeline_merge_report import run_merge_and_report
        total = run_merge_and_report()

    elapsed = time.time() - t0
    mins = elapsed / 60

    print("\n\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Total time: {mins:.1f} minutes")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("  Output files:")
    print("    data/processed/reviews_filtered_out.csv")
    print("    data/processed/reviews_to_extract.csv")
    print("    data/processed/extracted_behaviors.csv")
    print("    data/processed/extraction_checkpoint.json")
    print("    data/processed/extracted_reddit.csv")
    print("    data/processed/all_behaviors_master.csv")
    print("    data/reports/extraction_report.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
