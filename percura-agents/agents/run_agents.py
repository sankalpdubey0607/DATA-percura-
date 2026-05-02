"""
run_agents.py - Percura Agent Pipeline Orchestrator
====================================================
Runs both agents in sequence:
  1. Agent 1 (Sorter)  -> nvidia_personas_sorted.csv
  2. Agent 2 (Auditor) -> nvidia_personas_clean.csv
  3. Print final summary

Usage:
  python run_agents.py          # Run both agents
  python run_agents.py 1        # Run Agent 1 only
  python run_agents.py 2        # Run Agent 2 only (requires Agent 1 output)
"""
import sys
import time
from datetime import datetime


def main():
    start_agent = 1
    end_agent = 2
    if len(sys.argv) > 1:
        try:
            start_agent = int(sys.argv[1])
            end_agent = start_agent
        except ValueError:
            pass

    print("=" * 60)
    print("  PERCURA AGENT PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    t0 = time.time()
    sorter_result = None
    auditor_result = None

    # ── Agent 1: Sorter ──
    if start_agent <= 1 and end_agent >= 1:
        print("\n" + "#" * 60)
        print("  RUNNING AGENT 1: SORTING AGENT")
        print("#" * 60)
        from agent_sorter import run_sorter
        sorter_result = run_sorter()

    # ── Agent 2: Auditor ──
    if start_agent <= 2 and end_agent >= 2:
        print("\n\n" + "#" * 60)
        print("  RUNNING AGENT 2: AUDIT & VERIFICATION AGENT")
        print("#" * 60)
        from agent_auditor import run_auditor
        auditor_result = run_auditor()

    # ── Final Summary ──
    elapsed = time.time() - t0
    print("\n\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Total time: {elapsed/60:.1f} minutes")

    if sorter_result:
        print(f"\n  Agent 1 (Sorter):")
        print(f"    Personas processed:   {sorter_result['total_processed']:,}")
        print(f"    Unique archetypes:    {sorter_result['n_archetypes']}")
        print(f"    Default params used:  {sorter_result['default_pct']:.1f}%")
        print(f"    Time: {sorter_result['elapsed']:.1f}s")

    if auditor_result:
        print(f"\n  Agent 2 (Auditor):")
        print(f"    Input rows:           {auditor_result['total_input']:,}")
        print(f"    Clean (sim-ready):    {auditor_result['total_clean']:,}")
        print(f"    Removed:              {auditor_result['total_removed']:,}")
        print(f"    Flagged for review:   {auditor_result['total_flagged']:,}")
        print(f"    Quality score:        {auditor_result['quality_score']:.1f}%")
        print(f"    Safe archetypes:      {auditor_result['safe_archetypes']}")
        print(f"    Time: {auditor_result['elapsed']:.1f}s")

        # Simulation estimate
        clean = auditor_result['total_clean']
        batch_500_time_min = 500 * 4 / 60  # ~4s per persona simulation call
        total_batches = clean // 500
        print(f"\n  Simulation Estimate:")
        print(f"    Clean personas:       {clean:,}")
        print(f"    Batch size:           500 personas")
        print(f"    Total batches:        {total_batches:,}")
        print(f"    Est. time per batch:  {batch_500_time_min:.1f} min")
        print(f"    Est. total sim time:  {total_batches * batch_500_time_min / 60:.1f} hours")

    print(f"\n  Output files:")
    print(f"    data/processed/nvidia_personas_sorted.csv")
    print(f"    data/processed/nvidia_personas_clean.csv")
    print(f"    data/processed/nvidia_personas_removed.csv")
    print(f"    data/processed/nvidia_personas_flagged.csv")
    print(f"    data/reports/sorter_report.txt")
    print(f"    data/reports/audit_report.txt")
    print(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
