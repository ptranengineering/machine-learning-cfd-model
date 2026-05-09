#!/usr/bin/env python3
"""
Desktop popup optimizer UI (tkinter) for the design surrogate.

Run from repository root:
    python ml/popup_optimize_design.py
"""

from __future__ import annotations

import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import joblib
import numpy as np

ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from optimize_design import RANGE_PARAM_NAMES, default_bounds, run_surrogate_optimization


ROOT = ML_DIR.parent
MODEL_PATH = ROOT / "results" / "models" / "design_rf_model.joblib"

# Generous but physically possible UI limits (sliders + entry clamp). Mach stays >0 and <1
# for typical subsonic Euler/RANS setups; surrogate may be poor far outside training data.
PARAM_META = {
    "geometry_thickness": ("Thickness", 0.03, 0.35, 0.001),
    "geometry_camber": ("Camber", 0.00, 0.18, 0.001),
    "geometry_camber_pos": ("Camber Position", 0.05, 0.95, 0.005),
    "aoa": ("AoA (deg)", -15.0, 35.0, 0.1),
    "mach": ("Mach", 0.05, 0.99, 0.005),
    "reynolds": ("Reynolds", 1.0e4, 1.0e8, 5.0e3),
}


def _scale_resolution(step: float, key: str) -> float:
    if key == "reynolds":
        return max(1000.0, step)
    if step >= 0.01:
        return step
    return 0.001


class BoundRow:
    """
    One parameter: min/max each as [label | horizontal Scale | Entry].
    Uses classic tk.Frame + tk.Scale so sliders render reliably on Linux/WSL/X11.
    """

    def __init__(self, parent: ttk.Frame, key: str, default_lo: float, default_hi: float) -> None:
        self.key = key
        label, gmin, gmax, step = PARAM_META[key]
        self.gmin = float(gmin)
        self.gmax = float(gmax)
        self.step = float(step)
        res = _scale_resolution(self.step, key)

        outer = ttk.LabelFrame(parent, text=f"{label} ({key})", padding=10)
        outer.pack(fill="x", padx=4, pady=6)

        # Classic tk container — avoids ttk/Scale layout glitches on some platforms
        inner = tk.Frame(outer, bg="#f0f0f0")
        inner.pack(fill="x")

        self.min_var = tk.StringVar(value=f"{default_lo:g}")
        self.max_var = tk.StringVar(value=f"{default_hi:g}")

        scale_len = 420

        # --- Min row
        min_row = tk.Frame(inner, bg="#f0f0f0")
        min_row.pack(fill="x", pady=(0, 6))
        tk.Label(min_row, text="Min", width=5, anchor="w", bg="#f0f0f0").pack(side=tk.LEFT, padx=(0, 6))
        self.min_scale = tk.Scale(
            min_row,
            from_=self.gmin,
            to=self.gmax,
            orient=tk.HORIZONTAL,
            resolution=res,
            length=scale_len,
            showvalue=True,
            sliderlength=18,
            width=14,
            tickinterval=0,
            command=self._on_min_slide,
        )
        self.min_scale.set(default_lo)
        self.min_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        min_entry = ttk.Entry(min_row, textvariable=self.min_var, width=18)
        min_entry.pack(side=tk.LEFT)
        for seq in ("<FocusOut>", "<Return>"):
            min_entry.bind(seq, lambda _e: self._sync_from_entries())

        # --- Max row
        max_row = tk.Frame(inner, bg="#f0f0f0")
        max_row.pack(fill="x")
        tk.Label(max_row, text="Max", width=5, anchor="w", bg="#f0f0f0").pack(side=tk.LEFT, padx=(0, 6))
        self.max_scale = tk.Scale(
            max_row,
            from_=self.gmin,
            to=self.gmax,
            orient=tk.HORIZONTAL,
            resolution=res,
            length=scale_len,
            showvalue=True,
            sliderlength=18,
            width=14,
            tickinterval=0,
            command=self._on_max_slide,
        )
        self.max_scale.set(default_hi)
        self.max_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        max_entry = ttk.Entry(max_row, textvariable=self.max_var, width=18)
        max_entry.pack(side=tk.LEFT)
        for seq in ("<FocusOut>", "<Return>"):
            max_entry.bind(seq, lambda _e: self._sync_from_entries())

    def _on_min_slide(self, _value: str) -> None:
        mn = float(self.min_scale.get())
        mx = float(self.max_scale.get())
        if mn >= mx:
            mx = min(self.gmax, mn + self.step)
            self.max_scale.set(mx)
        self.min_var.set(f"{mn:g}")
        self.max_var.set(f"{mx:g}")

    def _on_max_slide(self, _value: str) -> None:
        mn = float(self.min_scale.get())
        mx = float(self.max_scale.get())
        if mx <= mn:
            mn = max(self.gmin, mx - self.step)
            self.min_scale.set(mn)
        self.min_var.set(f"{float(self.min_scale.get()):g}")
        self.max_var.set(f"{float(self.max_scale.get()):g}")

    def _sync_from_entries(self) -> None:
        try:
            mn = float(self.min_var.get().strip())
            mx = float(self.max_var.get().strip())
        except ValueError:
            return
        mn = max(self.gmin, min(self.gmax, mn))
        mx = max(self.gmin, min(self.gmax, mx))
        if mx <= mn:
            mx = min(self.gmax, mn + self.step)
        self.min_scale.set(mn)
        self.max_scale.set(mx)
        self.min_var.set(f"{mn:g}")
        self.max_var.set(f"{mx:g}")

    def values(self) -> tuple[float, float]:
        self._sync_from_entries()
        return float(self.min_scale.get()), float(self.max_scale.get())


class PopupOptimizerApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Aero Surrogate Optimizer")
        self.root.geometry("1024x900")
        self.root.minsize(880, 700)

        if not MODEL_PATH.exists():
            messagebox.showerror("Model not found", f"Missing model:\n{MODEL_PATH}\n\nTrain first.")
            self.root.destroy()
            raise SystemExit(1)

        self.pack = joblib.load(MODEL_PATH)
        d_lo, d_hi = default_bounds()

        # --- Top: scrollable bounds + settings + run button
        top_wrap = ttk.Frame(self.root, padding=(8, 8, 8, 4))
        top_wrap.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        header_lbl = ttk.Label(
            top_wrap,
            text=(
                "Design search bounds — trackpad/wheel scrolls everywhere in this pane "
                "(including the scrollbar strip). Ctrl+wheel over horizontal sliders adjusts the page scroll."
            ),
            font=("", 11, "bold"),
        )
        header_lbl.pack(anchor="w")

        # yscrollincrement: pixels per yview_scroll(..., "units"). Default 0 makes wheel feel broken/jerky.
        canvas = tk.Canvas(
            top_wrap,
            highlightthickness=0,
            bg="#e8e8e8",
            yscrollincrement=18,
        )
        vsb = ttk.Scrollbar(top_wrap, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)

        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll_inner = ttk.Frame(canvas, padding=4)
        scroll_win = canvas.create_window((0, 0), window=scroll_inner, anchor="nw")

        def _refresh_scrollregion(_event: tk.Event | None = None) -> None:
            canvas.update_idletasks()
            box = canvas.bbox("all")
            if box:
                canvas.configure(scrollregion=box)

        def _on_canvas_configure(event: tk.Event) -> None:
            # Match inner width to canvas so content uses full width
            canvas.itemconfigure(scroll_win, width=event.width)
            _refresh_scrollregion()

        scroll_inner.bind("<Configure>", _refresh_scrollregion)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _yview_scroll_units(steps: int) -> None:
            if steps == 0:
                return
            canvas.yview_scroll(steps, "units")
            canvas.after_idle(_refresh_scrollregion)

        def _wheel(event: tk.Event) -> str | None:
            # macOS: delta is small line steps
            if sys.platform == "darwin":
                _yview_scroll_units(int(-1 * event.delta))
                return "break"
            d = int(getattr(event, "delta", 0) or 0)
            if d == 0:
                return "break"
            # Windows: trackpads send many small deltas — use 1 unit each for smooth motion;
            # mouse wheel uses larger jumps (multiples of ~120).
            if abs(d) >= 60:
                steps = int(round(-d / 120.0))
                if steps == 0:
                    steps = -1 if d > 0 else 1
                steps = max(-12, min(12, steps))
            else:
                steps = -1 if d > 0 else 1
            _yview_scroll_units(steps)
            return "break"

        def _wheel_linux_up(_event: tk.Event) -> str | None:
            _yview_scroll_units(-2)
            return "break"

        def _wheel_linux_down(_event: tk.Event) -> str | None:
            _yview_scroll_units(2)
            return "break"

        # Scrollbar + header live *outside* scroll_inner — without these binds the trackpad stops when the pointer slips onto them.
        for _pane in (vsb, header_lbl):
            _pane.bind("<MouseWheel>", _wheel)
            _pane.bind("<Button-4>", _wheel_linux_up)
            _pane.bind("<Button-5>", _wheel_linux_down)

        def _bind_scroll_wheel(w: tk.Misc) -> None:
            """Trackpad/wheel on most inner widgets scrolls the canvas; skip tk.Scale so its own wheel still adjusts value."""
            if not isinstance(w, tk.Scale):
                w.bind("<MouseWheel>", _wheel)
                w.bind("<Button-4>", _wheel_linux_up)
                w.bind("<Button-5>", _wheel_linux_down)
            for ch in w.winfo_children():
                _bind_scroll_wheel(ch)

        canvas.bind("<Enter>", lambda _e: canvas.focus_set())
        canvas.bind("<MouseWheel>", _wheel)
        canvas.bind("<Button-4>", _wheel_linux_up)
        canvas.bind("<Button-5>", _wheel_linux_down)

        self.rows: dict[str, BoundRow] = {}
        for i, k in enumerate(RANGE_PARAM_NAMES):
            self.rows[k] = BoundRow(scroll_inner, k, float(d_lo[i]), float(d_hi[i]))

        ctrl = ttk.LabelFrame(scroll_inner, text="Optimization settings", padding=12)
        ctrl.pack(fill="x", padx=4, pady=10)

        self.objective = tk.StringVar(value="max_cl_cd")
        obj_row = ttk.Frame(ctrl)
        obj_row.pack(fill="x", pady=(0, 8))
        ttk.Label(obj_row, text="Objective:").pack(side=tk.LEFT, padx=(0, 12))
        ttk.Radiobutton(obj_row, text="Max CL/CD", value="max_cl_cd", variable=self.objective).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(obj_row, text="Max CL", value="max_cl", variable=self.objective).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(obj_row, text="Min CD", value="min_cd", variable=self.objective).pack(side=tk.LEFT, padx=4)

        self.min_cl = tk.StringVar(value="0.7")
        self.max_cd = tk.StringVar(value="0.2")
        self.iters = tk.StringVar(value="25")
        self.init_samples = tk.StringVar(value="30")
        self.candidate_pool = tk.StringVar(value="700")
        self.seed = tk.StringVar(value="42")

        fields = [
            ("Min CL", self.min_cl),
            ("Max CD", self.max_cd),
            ("Iterations", self.iters),
            ("Init samples", self.init_samples),
            ("Candidate pool", self.candidate_pool),
            ("Seed", self.seed),
        ]
        grid_f = ttk.Frame(ctrl)
        grid_f.pack(fill="x")
        for idx, (name, var) in enumerate(fields):
            r, col = divmod(idx, 3)
            ttk.Label(grid_f, text=name).grid(row=r * 2, column=col, sticky="nw", padx=(6, 10), pady=(6, 2))
            ttk.Entry(grid_f, textvariable=var, width=16).grid(
                row=r * 2 + 1, column=col, sticky="ew", padx=(6, 14), pady=(0, 10)
            )
        for c in range(3):
            grid_f.columnconfigure(c, weight=1)

        self.run_btn = ttk.Button(scroll_inner, text="Run optimization", command=self.run_optimization)
        self.run_btn.pack(fill="x", padx=4, pady=(4, 12))

        # Wheel/trackpad over any control in the scroll region scrolls the canvas (not only empty canvas).
        _bind_scroll_wheel(scroll_inner)

        # Over horizontal Scales the wheel normally moves the slider; Ctrl+wheel still scrolls the page.
        for _r in self.rows.values():
            for _sc in (_r.min_scale, _r.max_scale):
                _sc.bind("<Control-MouseWheel>", _wheel)
                _sc.bind("<Control-Button-4>", _wheel_linux_up)
                _sc.bind("<Control-Button-5>", _wheel_linux_down)

        # --- Bottom: result (separate scroll, always visible strip)
        out_frame = ttk.LabelFrame(self.root, text="Result", padding=8)
        out_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False)
        out_scroll = ttk.Scrollbar(out_frame, orient=tk.VERTICAL)
        self.out_text = tk.Text(out_frame, height=12, wrap="word", yscrollcommand=out_scroll.set)
        out_scroll.config(command=self.out_text.yview)
        out_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.out_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.out_text.insert("end", "Ready. Adjust bounds above (scroll if needed), then Run optimization.\n")
        self.out_text.configure(state="disabled")

    def _set_output(self, msg: str) -> None:
        self.out_text.configure(state="normal")
        self.out_text.delete("1.0", "end")
        self.out_text.insert("end", msg)
        self.out_text.configure(state="disabled")

    def run_optimization(self) -> None:
        try:
            lo = []
            hi = []
            for k in RANGE_PARAM_NAMES:
                mn, mx = self.rows[k].values()
                if mx <= mn:
                    raise ValueError(f"{k}: max must be > min")
                lo.append(mn)
                hi.append(mx)

            result = run_surrogate_optimization(
                self.pack,
                np.array(lo, dtype=float),
                np.array(hi, dtype=float),
                objective=self.objective.get(),
                min_cl=float(self.min_cl.get()),
                max_cd=float(self.max_cd.get()),
                iters=int(self.iters.get()),
                init_samples=int(self.init_samples.get()),
                candidate_pool=int(self.candidate_pool.get()),
                seed=int(self.seed.get()),
            )

            out_file = ROOT / "results" / "popup_design_optimization_result.json"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
            self._set_output(json.dumps(result, indent=2) + f"\n\nSaved to: {out_file}\n")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Optimization failed", str(e))

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    PopupOptimizerApp().run()
