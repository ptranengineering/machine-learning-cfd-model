#!/usr/bin/env python3
"""Generate all six publication figures and log results."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "results" / "figures" / "generation_log.txt"
PY = ROOT / ".venv" / "bin" / "python"

STEPS = [
    ("Fig 1", [str(PY), str(ROOT / "ml" / "plot_pipeline_architecture.py")]),
    ("Fig 2", [str(PY), str(ROOT / "ml" / "learning_curve_experiment.py"), "--n-repeats", "2"]),
    ("Fig 3-4", [str(PY), str(ROOT / "ml" / "train_design_baseline.py"), "--engineer-features", "--save-figures"]),
    ("Model comparison", [str(PY), str(ROOT / "scripts" / "generate_model_comparison.py")]),
    ("Fig 5", [str(PY), str(ROOT / "ml" / "optimize_design.py"), "--save-figures", "--n-seeds", "5"]),
    ("Fig 6", [str(PY), str(ROOT / "ml" / "plot_design_space.py")]),
    ("Readiness report", [str(PY), str(ROOT / "scripts" / "generate_research_readiness_report.py")]),
]


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for name, cmd in STEPS:
        lines.append(f"=== {name} ===")
        lines.append(" ".join(cmd))
        try:
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
            lines.append(f"exit={proc.returncode}")
            if proc.stdout:
                lines.append(proc.stdout.strip())
            if proc.stderr:
                lines.append("STDERR: " + proc.stderr.strip())
        except Exception as e:
            lines.append(f"ERROR: {e}")
        lines.append("")

    fig_dir = ROOT / "results" / "figures"
    if fig_dir.exists():
        lines.append("=== Output files ===")
        for f in sorted(fig_dir.glob("*.png")):
            lines.append(f"{f.name} ({f.stat().st_size} bytes)")
    else:
        lines.append("=== No figures directory created ===")

    LOG.write_text("\n".join(lines), encoding="utf-8")
    print(LOG.read_text(encoding="utf-8"))
    return 0 if fig_dir.exists() and any(fig_dir.glob("*.png")) else 1


if __name__ == "__main__":
    sys.exit(main())
