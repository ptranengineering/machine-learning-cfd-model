#!/usr/bin/env python3
"""Generate research readiness assessment from measured pipeline artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS = ROOT / "results" / "design_baseline_metrics.json"
DEFAULT_QUALITY = ROOT / "datasets" / "processed" / "aero_design_quality_report.csv"
DEFAULT_OUT_JSON = ROOT / "results" / "research_readiness_report.json"
DEFAULT_OUT_MD = ROOT / "results" / "research_readiness_report.md"


def main() -> None:
    p = argparse.ArgumentParser(description="Generate research readiness assessment report.")
    p.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--quality-report", type=Path, default=DEFAULT_QUALITY)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = p.parse_args()

    findings: dict = {"weaknesses": [], "recommendations": [], "experiments": []}
    metrics = {}
    if args.metrics.exists():
        metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
        n = metrics.get("n_samples", 0)
        readiness = metrics.get("readiness", {})
        status = readiness.get("status", "UNKNOWN")
        findings["n_samples"] = n
        findings["readiness_status"] = status
        findings["selected_model"] = metrics.get("selected_model")

        if n < 200:
            findings["weaknesses"].append(f"Dataset size ({n}) is below recommended 200–500 converged RANS cases.")
            findings["recommendations"].append("Run N_SAMPLES=500 ./scripts/run_large_design_sweep.sh")
        if status != "PASS":
            findings["weaknesses"].append("Surrogate readiness gate FAIL — R² and/or MAE % of range below targets.")
            findings["experiments"].append("Learning curve study at N = 50, 100, 200, 500 with RANS data.")
    else:
        findings["weaknesses"].append("No design_baseline_metrics.json found — train surrogate after dataset build.")

    if args.quality_report.exists():
        qdf = pd.read_csv(args.quality_report)
        passed = int(qdf["quality_pass"].sum()) if "quality_pass" in qdf.columns else 0
        failed = int((~qdf["quality_pass"]).sum()) if "quality_pass" in qdf.columns else 0
        findings["quality_passed"] = passed
        findings["quality_failed"] = failed
        if failed > 0:
            rate = failed / max(passed + failed, 1)
            findings["weaknesses"].append(f"{failed} CFD cases failed quality gate ({rate:.0%} rejection rate).")
    else:
        findings["weaknesses"].append("No design quality report — build dataset with --dataset-type design.")

    findings["model_limitations"] = [
        "Linear regression: poor in transonic nonlinear regions.",
        "Random Forest / Extra Trees: no uncertainty quantification; may overfit sparse data.",
        "Gaussian Process: O(N³) training cost; limited to ~300 samples without approximation.",
        "Neural Network (design_nn): not wired into optimizer by default.",
    ]
    findings["expected_dataset_size"] = "500+ converged RANS cases for ~5% MAE % of range readiness target."
    findings["conference_experiments"] = [
        "Euler vs. RANS ablation on identical LHS samples.",
        "Surrogate comparison table (LR / RF / ET / GP / NN) with unified metrics.",
        "Optimization validation: surrogate vs. fresh CFD at optimum and random holdout points.",
        "Feature engineering ablation (6D vs. 13D engineered inputs).",
    ]

    md_lines = [
        "# Research Readiness Assessment",
        "",
        f"- **Samples in training set:** {findings.get('n_samples', 'unknown')}",
        f"- **Readiness status:** {findings.get('readiness_status', 'unknown')}",
        f"- **Selected model:** {findings.get('selected_model', 'unknown')}",
        "",
        "## Remaining Technical Weaknesses",
        "",
    ]
    for w in findings["weaknesses"]:
        md_lines.append(f"- {w}")
    md_lines.extend(["", "## Recommended Dataset Size", "", f"- {findings['expected_dataset_size']}", "", "## Model Limitations", ""])
    for m in findings["model_limitations"]:
        md_lines.append(f"- {m}")
    md_lines.extend(["", "## Additional Experiments Before Conference Submission", ""])
    for e in findings["conference_experiments"]:
        md_lines.append(f"- {e}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(findings, indent=2), encoding="utf-8")
    args.out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[DONE] readiness JSON -> {args.out_json}")
    print(f"[DONE] readiness MD   -> {args.out_md}")


if __name__ == "__main__":
    main()
