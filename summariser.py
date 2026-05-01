"""
summariser.py
-------------
Reads extracted_behaviors.csv and produces:
  1. summary_report.txt  — human-readable report for Person C / beta founder briefings
  2. summary_data.json   — machine-readable for Person B's archetype pipeline

Improvements over v1:
  - Filters to confidence=medium|high (excludes low-signal noise)
  - Cross-tabulation: friction_type × drop_off_stage
  - Cross-tabulation: literacy_hint × emotion
  - Per-category breakdown (fintech, edtech, ecommerce, etc.)
  - Per-source breakdown (playstore vs reddit)
  - Top friction triggers per app
  - JSON export for automation
"""

import csv
import json
import os
from collections import Counter, defaultdict


EXTRACTED_FILE = "extracted_behaviors.csv"
SUMMARY_FILE   = "summary_report.txt"
JSON_FILE      = "summary_data.json"


def _count_field(rows: list[dict], field: str) -> Counter:
    return Counter(r.get(field, "unknown") for r in rows)


def _count_multivalue(rows: list[dict], field: str, sep: str = "|") -> Counter:
    """For pipe-separated multi-value fields like friction_type."""
    c: Counter = Counter()
    for r in rows:
        raw = r.get(field, "")
        for val in raw.split(sep):
            val = val.strip()
            if val:
                c[val] += 1
    return c


def _cross_tab(rows: list[dict], field_a: str, field_b: str, 
               a_is_multi: bool = False, sep: str = "|") -> Counter:
    """Cross-tabulate two fields. Returns Counter of (val_a, val_b) tuples."""
    c: Counter = Counter()
    for r in rows:
        val_b = r.get(field_b, "unknown")
        if not val_b or val_b == "unknown":
            continue

        if a_is_multi:
            raw_a = r.get(field_a, "")
            for val_a in raw_a.split(sep):
                val_a = val_a.strip()
                if val_a and val_a != "unknown":
                    c[(val_a, val_b)] += 1
        else:
            val_a = r.get(field_a, "unknown")
            if val_a and val_a != "unknown":
                c[(val_a, val_b)] += 1
    return c


def summarise(
    extracted_file: str = EXTRACTED_FILE,
    summary_file: str   = SUMMARY_FILE,
    json_file: str      = JSON_FILE,
) -> None:
    if not os.path.exists(extracted_file):
        print(f"[Summary] File not found: {extracted_file}")
        return

    with open(extracted_file, "r", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    total = len(all_rows)
    print(f"\n[Summary] Total records: {total}")

    # ── Filter to medium + high confidence only ──────────────────────────────
    quality_rows = [r for r in all_rows if r.get("confidence", "").lower() in ("medium", "high")]
    quality_count = len(quality_rows)
    low_count = total - quality_count
    print(f"  Quality rows (medium/high confidence): {quality_count}")
    print(f"  Low confidence (excluded from analysis): {low_count}")

    # ── Global field counts (on quality rows) ────────────────────────────────
    sections = {}
    sections["drop_off_stage"]    = _count_field(quality_rows, "drop_off_stage")
    sections["emotion"]           = _count_field(quality_rows, "emotion")
    sections["trust_signal"]      = _count_field(quality_rows, "trust_signal")
    sections["gave_up"]           = _count_field(quality_rows, "gave_up")
    sections["language"]          = _count_field(quality_rows, "language")
    sections["confidence"]        = _count_field(quality_rows, "confidence")
    sections["effort_complained"] = _count_field(quality_rows, "effort_complained")
    sections["literacy_hint"]     = _count_field(quality_rows, "literacy_hint")
    sections["device_hint"]       = _count_field(quality_rows, "device_hint")
    sections["income_hint"]       = _count_field(quality_rows, "income_hint")
    sections["region_hint"]       = _count_field(quality_rows, "region_hint")
    sections["friction_type"]     = _count_multivalue(quality_rows, "friction_type")

    # ── Per-app breakdowns ───────────────────────────────────────────────────
    per_app_emotion: dict[str, Counter] = defaultdict(Counter)
    per_app_friction: dict[str, Counter] = defaultdict(Counter)
    per_app_dropoff: dict[str, Counter] = defaultdict(Counter)
    for r in quality_rows:
        app = r.get("app_name", "unknown")
        per_app_emotion[app][r.get("emotion", "unknown")] += 1
        per_app_dropoff[app][r.get("drop_off_stage", "unknown")] += 1
        for f in r.get("friction_type", "").split("|"):
            f = f.strip()
            if f:
                per_app_friction[app][f] += 1

    # ── Per-category breakdowns ──────────────────────────────────────────────
    per_category_emotion: dict[str, Counter] = defaultdict(Counter)
    per_category_friction: dict[str, Counter] = defaultdict(Counter)
    per_category_dropoff: dict[str, Counter] = defaultdict(Counter)
    for r in quality_rows:
        cat = r.get("category", "unknown")
        per_category_emotion[cat][r.get("emotion", "unknown")] += 1
        per_category_dropoff[cat][r.get("drop_off_stage", "unknown")] += 1
        for f in r.get("friction_type", "").split("|"):
            f = f.strip()
            if f:
                per_category_friction[cat][f] += 1

    # ── Per-source breakdown ─────────────────────────────────────────────────
    per_source: dict[str, int] = Counter(r.get("source", "playstore") for r in quality_rows)

    # ── Cross-tabulations ────────────────────────────────────────────────────
    friction_x_dropoff = _cross_tab(quality_rows, "friction_type", "drop_off_stage", a_is_multi=True)
    literacy_x_emotion = _cross_tab(quality_rows, "literacy_hint", "emotion")

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD TEXT REPORT
    # ══════════════════════════════════════════════════════════════════════════
    lines = []
    lines.append("=" * 65)
    lines.append("  PERCURA BEHAVIORAL SIGNAL SUMMARY REPORT")
    lines.append("=" * 65)
    lines.append(f"  Total records:                       {total}")
    lines.append(f"  Quality records (medium/high conf):   {quality_count}")
    lines.append(f"  Low confidence (excluded):            {low_count}")
    lines.append("")

    # Source breakdown
    lines.append("── Data Sources ──")
    for src, cnt in per_source.most_common():
        lines.append(f"  {str(src):<30} {cnt:>5}")
    lines.append("")

    # Field-level breakdowns
    label_map = {
        "drop_off_stage":    "Drop-off Stage (where users struggled)",
        "emotion":           "Emotions Detected",
        "friction_type":     "Friction Types (multi-select)",
        "gave_up":           "Gave Up (deleted / stopped using)",
        "trust_signal":      "Trust Signals",
        "effort_complained": "Complained About Effort",
        "language":          "Language of Reviews",
        "confidence":        "Extraction Confidence",
        "literacy_hint":     "User Literacy Hint",
        "device_hint":       "Device Hint",
        "income_hint":       "Income Hint",
        "region_hint":       "Region Hint",
    }

    for key, counter in sections.items():
        lines.append(f"── {label_map.get(key, key)} ──")
        for value, count in counter.most_common():
            pct = f"{count / quality_count * 100:.1f}%" if quality_count else "0%"
            lines.append(f"  {str(value):<30} {count:>5}  ({pct})")
        lines.append("")

    # ── Cross-tab: Friction × Drop-off (top 15) ─────────────────────────────
    lines.append("── Cross-Tab: Friction Type × Drop-off Stage (top 15) ──")
    lines.append("  (Which frictions cause which dropoffs)")
    for (friction, stage), cnt in friction_x_dropoff.most_common(15):
        lines.append(f"  {friction:<25} → {stage:<20} {cnt:>4}")
    lines.append("")

    # ── Cross-tab: Literacy × Emotion (top 15) ──────────────────────────────
    lines.append("── Cross-Tab: Literacy Hint × Emotion (top 15) ──")
    lines.append("  (How literacy level correlates with frustration)")
    for (lit, emo), cnt in literacy_x_emotion.most_common(15):
        lines.append(f"  {lit:<15} × {emo:<20} {cnt:>4}")
    lines.append("")

    # ── Per-category breakdown ───────────────────────────────────────────────
    lines.append("── Breakdown by Product Category ──")
    for cat in sorted(per_category_emotion.keys()):
        cat_total = sum(per_category_emotion[cat].values())
        lines.append(f"\n  [{cat.upper()}] ({cat_total} reviews)")

        lines.append("    Top emotions:")
        for emo, cnt in per_category_emotion[cat].most_common(5):
            lines.append(f"      {str(emo):<25} {cnt}")

        lines.append("    Top friction types:")
        for f, cnt in per_category_friction.get(cat, Counter()).most_common(5):
            lines.append(f"      {str(f):<25} {cnt}")

        lines.append("    Top drop-off stages:")
        for s, cnt in per_category_dropoff.get(cat, Counter()).most_common(5):
            lines.append(f"      {str(s):<25} {cnt}")
    lines.append("")

    # ── Per-app breakdown ────────────────────────────────────────────────────
    lines.append("── Breakdown by App ──")
    for app in sorted(per_app_emotion.keys()):
        app_total = sum(per_app_emotion[app].values())
        lines.append(f"\n  [{app}] ({app_total} reviews)")

        lines.append("    Emotions:")
        for emo, cnt in per_app_emotion[app].most_common(5):
            lines.append(f"      {str(emo):<25} {cnt}")

        lines.append("    Frictions:")
        for f, cnt in per_app_friction.get(app, Counter()).most_common(5):
            lines.append(f"      {str(f):<25} {cnt}")

        lines.append("    Drop-off stages:")
        for s, cnt in per_app_dropoff.get(app, Counter()).most_common(3):
            lines.append(f"      {str(s):<25} {cnt}")
    lines.append("")

    # ── Top key quotes ───────────────────────────────────────────────────────
    quotes = [
        (r.get("app_name", ""), r.get("key_quote", "").strip())
        for r in quality_rows
        if r.get("key_quote", "").strip() and len(r.get("key_quote", "").strip()) > 5
    ]
    if quotes:
        lines.append("── Sample Key Quotes (top 20) ──")
        for app, q in quotes[:20]:
            lines.append(f'  [{app}] "{q}"')
        lines.append("")

    lines.append("=" * 65)
    report = "\n".join(lines)

    # Print to terminal
    print(report)

    # Save text report
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[Summary] Text report saved → {summary_file}")

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD JSON EXPORT (for Person B's automation)
    # ══════════════════════════════════════════════════════════════════════════
    json_data = {
        "meta": {
            "total_records": total,
            "quality_records": quality_count,
            "low_confidence_excluded": low_count,
            "sources": dict(per_source),
        },
        "global_counts": {
            key: dict(counter.most_common())
            for key, counter in sections.items()
        },
        "cross_tabs": {
            "friction_x_dropoff": [
                {"friction": f, "stage": s, "count": c}
                for (f, s), c in friction_x_dropoff.most_common(30)
            ],
            "literacy_x_emotion": [
                {"literacy": l, "emotion": e, "count": c}
                for (l, e), c in literacy_x_emotion.most_common(30)
            ],
        },
        "per_category": {
            cat: {
                "total": sum(per_category_emotion[cat].values()),
                "top_emotions": dict(per_category_emotion[cat].most_common(5)),
                "top_frictions": dict(per_category_friction.get(cat, Counter()).most_common(5)),
                "top_dropoffs": dict(per_category_dropoff.get(cat, Counter()).most_common(5)),
            }
            for cat in sorted(per_category_emotion.keys())
        },
        "per_app": {
            app: {
                "total": sum(per_app_emotion[app].values()),
                "top_emotions": dict(per_app_emotion[app].most_common(5)),
                "top_frictions": dict(per_app_friction.get(app, Counter()).most_common(5)),
                "top_dropoffs": dict(per_app_dropoff.get(app, Counter()).most_common(3)),
            }
            for app in sorted(per_app_emotion.keys())
        },
    }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"[Summary] JSON data saved → {json_file}")


if __name__ == "__main__":
    summarise()
