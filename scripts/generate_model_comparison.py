#!/usr/bin/env python3
"""Aggregate surrogate metrics into publication-style comparison tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS = ROOT / "results" / "design_baseline_metrics.json"
DEFAULT_CSV = ROOT / "results" / "model_comparison_summary.csv"
DEFAULT_JSON = ROOT / "results" / "model_comparison_summary.json"


def load_comparison_rows(metrics_path: Path) -> list[dict]:
    if not metrics_path.exists():
        return []
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    if "comparison_table" in data:
        return data["comparison_table"]
    rows = []
    for block, split in (("holdout_results", "holdout"), ("cross_validation_results", "cv_mean")):
        for entry in data.get(block, []):
            model = entry["model"]
            for target, metrics in entry.get("targets", {}).items():
                row = {"model": model, "target": target, "split": split}
                for key in ("mae", "rmse", "r2", "mape", "relative_pct_error"):
                    val = metrics.get(key)
                    if val is None and split == "cv_mean":
                        val = metrics.get(f"{key}_mean")
                    row[key] = val
                rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Generate consolidated surrogate comparison tables.")
    p.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    args = p.parse_args()

    rows = load_comparison_rows(args.metrics)
    if not rows:
        raise FileNotFoundError(f"No comparison data found in {args.metrics}")

    df = pd.DataFrame(rows)
    summary = {
        "source_metrics": str(args.metrics.relative_to(ROOT)),
        "n_rows": int(len(df)),
        "models": sorted(df["model"].unique().tolist()),
        "targets": sorted(df["target"].unique().tolist()),
        "rows": rows,
    }

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] comparison CSV  -> {args.out_csv}")
    print(f"[DONE] comparison JSON -> {args.out_json}")


if __name__ == "__main__":
    main()
