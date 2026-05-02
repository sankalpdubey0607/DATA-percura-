# Percura Agent Pipeline

Sorts 1M Nvidia personas into behavioral archetypes and audits data quality for simulation readiness.

## Quick Start

```bash
cd percura-agents
pip install -r requirements.txt
cd agents
python run_agents.py        # Run both agents
python run_agents.py 1      # Run Agent 1 only (Sorter)
python run_agents.py 2      # Run Agent 2 only (Auditor)
```

## Architecture

### Agent 1: Sorting Agent (`agent_sorter.py`)
Reads the [Nvidia Nemotron-Personas-India](https://huggingface.co/datasets/nvidia/Nemotron-Personas-India) dataset (en_IN, 1M rows) and assigns every persona to an archetype based on:
- `education_level` -> literacy (illiterate/primary/secondary/graduate)
- `occupation` -> occupation type (no_income/informal/blue_collar/white_collar/professional)
- `district` + `state` -> region (metro/tier_2/tier_3/rural)
- `age` -> age band (youth/young/middle/senior)
- Device inferred from occupation

Archetype format: `{age_band}-{region}-{literacy}-{occupation}-{device}`

### Agent 2: Audit Agent (`agent_auditor.py`)
Runs 6 checks on every row:
1. **Completeness** - key fields present, valid age range
2. **Duplicate** - exact and near-duplicate detection
3. **Behavioral Usefulness** - parameter bounds check
4. **Archetype Balance** - flags under/oversized archetypes
5. **Consistency** - cross-field logic validation
6. **Simulation Viability** - all required fields for simulation

### Config (`config/archetype_params.json`)
Single source of truth for all behavioral parameters. Update this file to change archetype behavior without modifying code.

## Output Files

| File | Description |
|------|-------------|
| `nvidia_personas_sorted.csv` | All personas with archetype + behavioral params |
| `nvidia_personas_clean.csv` | Passed all 6 audits - simulation ready |
| `nvidia_personas_removed.csv` | Failed audit - with removal reasons |
| `nvidia_personas_flagged.csv` | Passed but has consistency flags |
| `sorter_report.txt` | Agent 1 statistics |
| `audit_report.txt` | Agent 2 statistics + recommendations |

## Performance
- Chunked processing: 50k rows/chunk
- Checkpointing every 500k rows
- Memory: <4GB RAM
- Estimated time: 10-15 minutes total

## Dataset
- Source: `nvidia/Nemotron-Personas-India` (en_IN split)
- 1,000,000 personas, 28 columns
- Auto-downloaded via HuggingFace `datasets` library on first run
