/**
 * Surrogate optimizer UI — syncs sliders + number inputs, POST /api/optimize
 */

const PARAMS = [
  {
    key: "geometry_thickness",
    title: "Thickness",
    desc: "Fraction of chord",
    globalMin: 0.06,
    globalMax: 0.22,
    step: 0.001,
    defaultMin: 0.1,
    defaultMax: 0.16,
  },
  {
    key: "geometry_camber",
    title: "Camber",
    desc: "Max camber (chord fraction)",
    globalMin: 0,
    globalMax: 0.08,
    step: 0.001,
    defaultMin: 0,
    defaultMax: 0.04,
  },
  {
    key: "geometry_camber_pos",
    title: "Camber position",
    desc: "Chord fraction to max camber",
    globalMin: 0.2,
    globalMax: 0.65,
    step: 0.01,
    defaultMin: 0.3,
    defaultMax: 0.5,
  },
  {
    key: "aoa",
    title: "Angle of attack",
    desc: "Degrees",
    globalMin: -4,
    globalMax: 16,
    step: 0.1,
    defaultMin: 0,
    defaultMax: 6,
  },
  {
    key: "mach",
    title: "Mach",
    desc: "Freestream",
    globalMin: 0.5,
    globalMax: 0.92,
    step: 0.005,
    defaultMin: 0.68,
    defaultMax: 0.8,
  },
  {
    key: "reynolds",
    title: "Reynolds",
    desc: "Re",
    globalMin: 1e5,
    globalMax: 2e7,
    step: 1e4,
    defaultMin: 3e6,
    defaultMax: 10e6,
  },
];

function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  Object.assign(node, props);
  for (const c of children) node.append(c);
  return node;
}

function syncSliderToNum(slider, num, gMin, gMax) {
  const v = Number(num.value);
  if (Number.isFinite(v)) {
    const clamped = Math.min(gMax, Math.max(gMin, v));
    slider.value = String(clamped);
    if (clamped !== v) num.value = String(roundForStep(clamped, slider.step));
  }
}

function roundForStep(v, stepStr) {
  const step = Number(stepStr);
  if (!step || step >= 1) return Math.round(v);
  const decimals = Math.max(0, String(step).split(".")[1]?.length ?? 0);
  return Number(v.toFixed(decimals));
}

function buildBoundRows(container) {
  for (const p of PARAMS) {
    const block = el("div", { className: "bound-block" });
    block.append(
      el("header", {}, [
        el("span", { className: "title", textContent: p.title }),
        el("span", { className: "key", textContent: p.key }),
      ])
    );
    const desc = el("p", {
      className: "hint",
      style: { margin: "0 0 0.5rem", fontSize: "0.8rem" },
      textContent: p.desc,
    });
    block.append(desc);

    const minId = `${p.key}-min`;
    const maxId = `${p.key}-max`;
    const minSlideId = `${p.key}-min-slide`;
    const maxSlideId = `${p.key}-max-slide`;

    const row = el("div", { className: "bound-sliders" });

    const minLabel = el("label", {}, [
      document.createTextNode("Min"),
      el("input", {
        type: "range",
        id: minSlideId,
        min: p.globalMin,
        max: p.globalMax,
        step: p.step,
        value: p.defaultMin,
      }),
      el("input", {
        type: "number",
        id: minId,
        min: p.globalMin,
        max: p.globalMax,
        step: p.step,
        value: p.defaultMin,
      }),
    ]);

    const maxLabel = el("label", {}, [
      document.createTextNode("Max"),
      el("input", {
        type: "range",
        id: maxSlideId,
        min: p.globalMin,
        max: p.globalMax,
        step: p.step,
        value: p.defaultMax,
      }),
      el("input", {
        type: "number",
        id: maxId,
        min: p.globalMin,
        max: p.globalMax,
        step: p.step,
        value: p.defaultMax,
      }),
    ]);

    row.append(minLabel, maxLabel);
    block.append(row);
    container.append(block);

    const minSlide = block.querySelector(`#${minSlideId}`);
    const maxSlide = block.querySelector(`#${maxSlideId}`);
    const minNum = block.querySelector(`#${minId}`);
    const maxNum = block.querySelector(`#${maxId}`);

    minSlide.addEventListener("input", () => {
      minNum.value = minSlide.value;
      if (Number(maxNum.value) <= Number(minNum.value)) {
        const bump = Number(minNum.value) + Number(p.step);
        maxNum.value = String(Math.min(p.globalMax, bump));
        maxSlide.value = maxNum.value;
      }
    });
    maxSlide.addEventListener("input", () => {
      maxNum.value = maxSlide.value;
      if (Number(maxNum.value) <= Number(minNum.value)) {
        const bump = Number(maxNum.value) - Number(p.step);
        minNum.value = String(Math.max(p.globalMin, bump));
        minSlide.value = minNum.value;
      }
    });
    minNum.addEventListener("change", () => syncSliderToNum(minSlide, minNum, p.globalMin, p.globalMax));
    maxNum.addEventListener("change", () => syncSliderToNum(maxSlide, maxNum, p.globalMin, p.globalMax));
  }
}

function collectBounds() {
  const bounds = {};
  for (const p of PARAMS) {
    const minV = Number(document.getElementById(`${p.key}-min`).value);
    const maxV = Number(document.getElementById(`${p.key}-max`).value);
    if (!Number.isFinite(minV) || !Number.isFinite(maxV)) {
      throw new Error(`Invalid numbers for ${p.key}`);
    }
    if (maxV <= minV) {
      throw new Error(`${p.title}: max must be greater than min`);
    }
    bounds[p.key] = [minV, maxV];
  }
  return bounds;
}

function selectedObjective() {
  const r = document.querySelector('input[name="objective"]:checked');
  return r ? r.value : "max_cl_cd";
}

async function fetchHealth() {
  const pill = document.getElementById("health-pill");
  const runBtn = document.getElementById("run-btn");
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.ok) {
      pill.textContent = "Model loaded";
      pill.className = "health-pill ok";
      runBtn.disabled = false;
    } else {
      pill.textContent = data.error || "Model missing";
      pill.className = "health-pill bad";
      runBtn.disabled = true;
    }
  } catch {
    pill.textContent = "Server unreachable";
    pill.className = "health-pill bad";
    runBtn.disabled = true;
  }
}

async function loadDefaultsFromApi() {
  try {
    const res = await fetch("/api/defaults");
    if (!res.ok) return;
    const data = await res.json();
    const b = data.bounds;
    if (!b) return;
    for (const p of PARAMS) {
      const pair = b[p.key];
      if (!pair || pair.length !== 2) continue;
      const minEl = document.getElementById(`${p.key}-min`);
      const maxEl = document.getElementById(`${p.key}-max`);
      const minSl = document.getElementById(`${p.key}-min-slide`);
      const maxSl = document.getElementById(`${p.key}-max-slide`);
      if (minEl && maxEl) {
        minEl.value = pair[0];
        maxEl.value = pair[1];
        if (minSl) minSl.value = String(Math.min(Number(maxSl.max), Math.max(Number(minSl.min), pair[0])));
        if (maxSl) maxSl.value = String(Math.min(Number(maxSl.max), Math.max(Number(maxSl.min), pair[1])));
      }
    }
  } catch {
    /* keep UI defaults */
  }
}

function showResult(data) {
  document.getElementById("results-empty").hidden = true;
  const content = document.getElementById("results-content");
  content.hidden = false;

  const pred = data.predicted || {};
  document.getElementById("out-cl").textContent =
    pred.CL != null ? Number(pred.CL).toFixed(5) : "—";
  document.getElementById("out-cd").textContent =
    pred.CD != null ? Number(pred.CD).toFixed(5) : "—";
  document.getElementById("out-clcd").textContent =
    pred.CL_CD != null ? Number(pred.CL_CD).toFixed(4) : "—";
  document.getElementById("out-score").textContent =
    data.best_internal_score != null ? String(data.best_internal_score) : "—";

  const geo = data.best_geometry_flow || {};
  const dl = document.getElementById("out-geometry");
  dl.innerHTML = "";
  const order = [
    "geometry_thickness",
    "geometry_camber",
    "geometry_camber_pos",
    "aoa",
    "mach",
    "reynolds",
  ];
  const labels = {
    geometry_thickness: "Thickness",
    geometry_camber: "Camber",
    geometry_camber_pos: "Camber pos",
    aoa: "AoA (°)",
    mach: "Mach",
    reynolds: "Re",
  };
  for (const k of order) {
    if (geo[k] == null) continue;
    dl.appendChild(el("dt", { textContent: labels[k] || k }));
    dl.appendChild(el("dd", { textContent: String(geo[k]) }));
  }

  document.getElementById("out-json").textContent = JSON.stringify(data, null, 2);
}

function showError(msg) {
  const e = document.getElementById("error-msg");
  e.textContent = msg;
  e.hidden = false;
}

function clearError() {
  const e = document.getElementById("error-msg");
  e.hidden = true;
  e.textContent = "";
}

document.addEventListener("DOMContentLoaded", () => {
  const container = document.getElementById("bound-rows");
  buildBoundRows(container);

  document.getElementById("run-btn").addEventListener("click", async () => {
    clearError();
    let bounds;
    try {
      bounds = collectBounds();
    } catch (err) {
      showError(err.message || String(err));
      return;
    }

    const body = {
      bounds,
      objective: selectedObjective(),
      min_cl: Number(document.getElementById("min-cl").value),
      max_cd: Number(document.getElementById("max-cd").value),
      iters: Number(document.getElementById("iters").value),
      init_samples: Number(document.getElementById("init-samples").value),
      candidate_pool: Number(document.getElementById("candidate-pool").value),
      seed: Number(document.getElementById("seed").value),
    };

    const btn = document.getElementById("run-btn");
    btn.disabled = true;
    btn.textContent = "Running…";

    try {
      const res = await fetch("/api/optimize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail = data.detail;
        let msg = res.statusText || "Request failed";
        if (typeof detail === "string") msg = detail;
        else if (Array.isArray(detail))
          msg = detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
        else if (detail && typeof detail === "object") msg = JSON.stringify(detail);
        showError(msg);
        return;
      }
      showResult(data);
    } catch (err) {
      showError(err.message || "Network error");
    } finally {
      btn.disabled = false;
      btn.textContent = "Run optimization";
      fetchHealth();
    }
  });

  fetchHealth().then(() => loadDefaultsFromApi());
});
