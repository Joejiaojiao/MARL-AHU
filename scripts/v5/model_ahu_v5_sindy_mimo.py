"""
AHU v5 — SINDy-style MIMO state-space predictor.

Design:
  Inputs  (exogenous, always from real data): T_out, T_ret
  Outputs (AHU internal states, predicted):   T_rec, T_coil, T_sup

Feature library (SINDy candidate functions, evaluated at time t):
  - Level-1 : T_out, T_ret, T_rec, T_coil, T_sup          (5 terms)
  - Level-2 : all degree-2 monomials of the 5 variables    (15 terms)
  - Time    : sin/cos of time-of-day, sin/cos of day-of-year (4 terms)
  Total: 24 candidate features per sample.

Model: Lasso regression (sparsity via L1) trained independently for each
output's delta:  delta_i = X_i(t+1) - X_i(t).
Prediction:      X_i(t+1) = X_i(t) + Lasso_i( Theta(x(t)) )

Multi-step rollout:
  - T_out, T_ret  : always taken from real data (treated as known disturbances)
  - T_rec, T_coil, T_sup : auto-regressed from model predictions

Evaluation horizons: 1, 2, 4, 8, 12 steps (5 min → 1 h).
"""

from __future__ import annotations

import argparse
import csv
import html as html_mod
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

# ── constants ─────────────────────────────────────────────────────────────────

EXOG   = ["T_out", "T_ret"]                    # known future (real values used in rollout)
STATES = ["T_rec", "T_coil", "T_sup"]          # predicted outputs
ALL_VARS = EXOG + STATES                        # order matters for feature library

HORIZONS = [1, 2, 4, 8, 12]                    # steps (× 5 min)

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AHU v5 SINDy-MIMO predictor")
    p.add_argument("--split-dir",  default="data_preprocessing/splits")
    p.add_argument("--out-dir",    default="model_outputs/v5")
    p.add_argument("--freq-min",   type=int,   default=5)
    p.add_argument("--alpha",      type=float, default=0.001,
                   help="Lasso regularisation strength (L1).")
    p.add_argument("--lasso-iter", type=int,   default=2000,
                   help="Coordinate-descent iterations for Lasso.")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()

# ── data loading ──────────────────────────────────────────────────────────────

def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def to_float(row: dict[str, str], col: str) -> float:
    return float(row[col])


@dataclass
class Sample:
    timestamp: str
    x_now:  dict[str, float]   # all 5 vars at t
    x_next: dict[str, float]   # all 5 vars at t+1  (ground truth)
    delta:  dict[str, float]   # x_next - x_now  for STATES only


def build_samples(rows: list[dict[str, str]], freq_min: int) -> list[Sample]:
    step = timedelta(minutes=freq_min)
    parsed = [datetime.fromisoformat(r["timestamp"]) for r in rows]
    samples: list[Sample] = []
    for i in range(len(rows) - 1):
        if parsed[i + 1] - parsed[i] != step:
            continue
        try:
            now  = {v: to_float(rows[i],     v) for v in ALL_VARS}
            nxt  = {v: to_float(rows[i + 1], v) for v in ALL_VARS}
        except (KeyError, ValueError):
            continue
        delta = {v: nxt[v] - now[v] for v in STATES}
        samples.append(Sample(rows[i + 1]["timestamp"], now, nxt, delta))
    return samples

# ── SINDy feature library ─────────────────────────────────────────────────────

def time_features(ts: str) -> list[float]:
    dt = datetime.fromisoformat(ts)
    # use previous timestamp (x_now corresponds to rows[i])
    mday = dt.hour * 60 + dt.minute
    dangle = 2.0 * math.pi * mday / 1440.0
    yangle = 2.0 * math.pi * dt.timetuple().tm_yday / 365.25
    return [math.sin(dangle), math.cos(dangle),
            math.sin(yangle), math.cos(yangle)]


def build_theta(x: dict[str, float], ts: str) -> list[float]:
    """
    Candidate function library Theta(x):
      degree-1 terms: all 5 variables
      degree-2 terms: all pairs (including squares) of 5 variables  -> 15 terms
      time features:  sin/cos day-angle, sin/cos year-angle          ->  4 terms
    Total: 5 + 15 + 4 = 24 features.
    """
    vals = [x[v] for v in ALL_VARS]           # [T_out, T_ret, T_rec, T_coil, T_sup]
    feats: list[float] = list(vals)            # degree-1
    for i in range(len(vals)):                 # degree-2
        for j in range(i, len(vals)):
            feats.append(vals[i] * vals[j])
    feats.extend(time_features(ts))
    return feats


def feature_names() -> list[str]:
    names = list(ALL_VARS)
    for i, vi in enumerate(ALL_VARS):
        for vj in ALL_VARS[i:]:
            names.append(f"{vi}*{vj}")
    names += ["sin_day", "cos_day", "sin_year", "cos_year"]
    return names

# ── Lasso (coordinate descent, no external libs) ──────────────────────────────

@dataclass
class LassoModel:
    coef: list[float]
    intercept: float
    means: list[float]
    stds:  list[float]
    alpha: float
    n_iter: int


def _soft_threshold(x: float, lam: float) -> float:
    if x > lam:
        return x - lam
    if x < -lam:
        return x + lam
    return 0.0


def fit_lasso(
    X: list[list[float]],
    y: list[float],
    alpha: float,
    max_iter: int,
    tol: float = 1e-6,
) -> LassoModel:
    n, p = len(X), len(X[0])

    # standardise features (column-wise)
    means = [sum(X[i][j] for i in range(n)) / n for j in range(p)]
    stds: list[float] = []
    for j in range(p):
        var = sum((X[i][j] - means[j]) ** 2 for i in range(n)) / max(1, n - 1)
        stds.append(math.sqrt(var) if var > 1e-12 else 1.0)

    # Xs[j] = column j as a flat list (column-major for fast dot products)
    Xs_col = [[( X[i][j] - means[j]) / stds[j] for i in range(n)] for j in range(p)]

    intercept = sum(y) / n
    coef = [0.0] * p

    # residual vector r = y - intercept - X @ coef  (initially y - intercept)
    r = [y[i] - intercept for i in range(n)]

    # column squared norms (all 1.0 after standardisation, but compute explicitly)
    col_norm2 = [sum(v * v for v in Xs_col[j]) for j in range(p)]

    lam = alpha * n     # Lasso objective = (1/2n)||r||^2 + alpha*||coef||_1
                        # coordinate update threshold = alpha * n / col_norm2[j]

    for _ in range(max_iter):
        max_change = 0.0
        for j in range(p):
            cj = coef[j]
            col = Xs_col[j]
            # restore residual contribution of feature j
            if cj != 0.0:
                for i in range(n):
                    r[i] += cj * col[i]
            # OLS update for this coordinate
            rho = sum(r[i] * col[i] for i in range(n))
            new_cj = _soft_threshold(rho, lam) / col_norm2[j]
            # update residual
            if new_cj != 0.0:
                for i in range(n):
                    r[i] -= new_cj * col[i]
            change = abs(new_cj - cj)
            if change > max_change:
                max_change = change
            coef[j] = new_cj
        if max_change < tol:
            break

    # refit intercept
    intercept = sum(y[i] - sum(coef[j] * Xs_col[j][i] for j in range(p))
                    for i in range(n)) / n

    return LassoModel(coef, intercept, means, stds, alpha, max_iter)


def predict_lasso(model: LassoModel, x_feat: list[float]) -> float:
    xs = [(x_feat[j] - model.means[j]) / model.stds[j] for j in range(len(x_feat))]
    return model.intercept + sum(c * v for c, v in zip(model.coef, xs))

# ── training ──────────────────────────────────────────────────────────────────

def train_models(
    samples: list[Sample],
    alpha: float,
    lasso_iter: int,
) -> dict[str, LassoModel]:
    """Train one Lasso model per output state."""
    X = [build_theta(s.x_now, s.timestamp) for s in samples]
    models: dict[str, LassoModel] = {}
    for state in STATES:
        y = [s.delta[state] for s in samples]
        print(f"  Fitting Lasso for delta_{state}  "
              f"(n={len(y)}, features={len(X[0])}, alpha={alpha}) ...")
        models[state] = fit_lasso(X, y, alpha, lasso_iter)
        n_active = sum(1 for c in models[state].coef if abs(c) > 1e-10)
        print(f"    -> active terms: {n_active}/{len(X[0])}")
    return models

# ── single-step prediction ─────────────────────────────────────────────────────

def predict_one_step(
    models: dict[str, LassoModel],
    x_now: dict[str, float],
    ts_now: str,
) -> dict[str, float]:
    """Predict x(t+1) given x(t). Returns full state dict."""
    theta = build_theta(x_now, ts_now)
    x_next = dict(x_now)          # copy; exog will be overwritten from real data later
    for state in STATES:
        delta_pred = predict_lasso(models[state], theta)
        x_next[state] = x_now[state] + delta_pred
    return x_next

# ── multi-step rollout ─────────────────────────────────────────────────────────

def rollout(
    models: dict[str, LassoModel],
    samples: list[Sample],
    horizon: int,
) -> list[dict[str, float]]:
    """
    For each sample i, roll out H steps starting from x(i).
    T_out, T_ret at future steps taken from real data (samples[i+h].x_next).
    Returns predicted x at step i+horizon (for all samples where i+horizon exists).
    """
    preds: list[dict[str, float]] = []
    for i in range(len(samples) - horizon):
        # check consecutive: samples are already gap-filtered, but rollout
        # needs all intermediate steps to be consecutive too
        x = dict(samples[i].x_now)
        ts = samples[i].timestamp
        valid = True
        for h in range(horizon):
            x_pred = predict_one_step(models, x, ts)
            # inject real exogenous at t+h+1
            future = samples[i + h].x_next     # = samples[i+h+1].x_now if consecutive
            x_pred[EXOG[0]] = future[EXOG[0]]
            x_pred[EXOG[1]] = future[EXOG[1]]
            x = x_pred
            ts = samples[i + h].timestamp
        preds.append(x)
    return preds

# ── metrics ───────────────────────────────────────────────────────────────────

def metrics(true: list[float], pred: list[float]) -> dict[str, float]:
    n = len(true)
    errors = [p - t for t, p in zip(true, pred)]
    mae  = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    mean_y = sum(true) / n
    sst = sum((t - mean_y) ** 2 for t in true)
    sse = sum(e * e for e in errors)
    r2  = 1.0 - sse / sst if sst > 0 else float("nan")
    bias = sum(errors) / n
    return {"n": n, "mae": mae, "rmse": rmse, "r2": r2, "bias": bias}


def persistence_pred(samples: list[Sample], horizon: int, state: str) -> list[float]:
    """Persistence baseline: X(t+H) ≈ X(t)."""
    return [samples[i].x_now[state] for i in range(len(samples) - horizon)]


def true_vals(samples: list[Sample], horizon: int, state: str) -> list[float]:
    return [samples[i + horizon].x_now[state] for i in range(len(samples) - horizon)]

# ── report ────────────────────────────────────────────────────────────────────

def build_report(
    train_samples: list[Sample],
    val_samples:   list[Sample],
    models: dict[str, LassoModel],
    alpha: float,
    args: argparse.Namespace,
) -> list[str]:
    feat_names = feature_names()
    lines = [
        "AHU v5 SINDy-MIMO model report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Alpha (Lasso L1): {alpha}",
        f"Frequency: {args.freq_min} min",
        f"Exogenous (real values in rollout): {EXOG}",
        f"Predicted states: {STATES}",
        f"Feature library size: {len(feat_names)}",
        "",
        "Active Lasso coefficients per output:",
    ]
    for state in STATES:
        m = models[state]
        active = [(feat_names[j], m.coef[j])
                  for j in range(len(m.coef)) if abs(m.coef[j]) > 1e-10]
        lines.append(f"  delta_{state}  intercept={m.intercept:+.6f}  "
                     f"active={len(active)}/{len(m.coef)}")
        for name, c in sorted(active, key=lambda x: -abs(x[1])):
            lines.append(f"    {name:20s}  {c:+.6f}")
        lines.append("")

    for split_name, samples in [("train", train_samples), ("validation", val_samples)]:
        lines.append(f"{'='*60}")
        lines.append(f"Split: {split_name}  ({len(samples)} samples)")
        for H in HORIZONS:
            if H >= len(samples):
                continue
            preds_all = rollout(models, samples, H)
            lines.append(f"  Horizon H={H:2d} ({H*args.freq_min} min)")
            for state in STATES:
                tv = true_vals(samples, H, state)
                pv = [p[state] for p in preds_all]
                persv = persistence_pred(samples, H, state)
                m_pred = metrics(tv, pv)
                m_pers = metrics(tv, persv)
                lines.append(
                    f"    {state:8s}  "
                    f"Lasso  MAE={m_pred['mae']:.4f} RMSE={m_pred['rmse']:.4f} "
                    f"R2={m_pred['r2']:.4f} bias={m_pred['bias']:+.4f} | "
                    f"Persistence MAE={m_pers['mae']:.4f} RMSE={m_pers['rmse']:.4f}"
                )
        lines.append("")
    return lines

# ── HTML output ───────────────────────────────────────────────────────────────

def build_plot_data(
    val_samples: list[Sample],
    models: dict[str, LassoModel],
    freq_min: int,
) -> dict:
    """Build JSON payload for 1-step and multi-step HTML plots."""
    h1_preds = rollout(models, val_samples, 1)
    h4_preds = rollout(models, val_samples, 4)
    h12_preds = rollout(models, val_samples, 12)

    n = min(len(h1_preds), len(h4_preds), len(h12_preds))
    timestamps = [val_samples[i + 1].timestamp for i in range(n)]

    series: dict = {}
    for state in STATES:
        true = [val_samples[i + 1].x_now[state] for i in range(n)]
        pers = [val_samples[i].x_now[state]      for i in range(n)]
        series[state] = {
            "true":        true,
            "persistence": pers,
            "lasso_h1":    [h1_preds[i][state]  for i in range(n)],
            "lasso_h4":    [h4_preds[i][state]  for i in range(n)],
            "lasso_h12":   [h12_preds[i][state] for i in range(n)],
        }
    return {"timestamps": timestamps, "states": series, "freqMin": freq_min}


def render_html(plot_data: dict, out_path: Path, alpha: float) -> str:
    data_json = json.dumps(plot_data, separators=(",", ":")).replace("</", "<\\/")
    states = STATES
    colors = {
        "true":        "#111827",
        "persistence": "#9ca3af",
        "lasso_h1":    "#2563eb",
        "lasso_h4":    "#16a34a",
        "lasso_h12":   "#dc2626",
    }
    labels = {
        "true":        "True",
        "persistence": "Persistence",
        "lasso_h1":    "Lasso H=1 (5 min)",
        "lasso_h4":    "Lasso H=4 (20 min)",
        "lasso_h12":   "Lasso H=12 (1 h)",
    }
    series_keys = ["persistence", "lasso_h1", "lasso_h4", "lasso_h12"]

    # metric table rows
    metric_rows = []
    for state in states:
        sd = plot_data["states"][state]
        tv = sd["true"]
        for sk in series_keys:
            pv = sd[sk]
            errs = [p - t for t, p in zip(tv, pv)]
            mae  = sum(abs(e) for e in errs) / len(errs)
            rmse = math.sqrt(sum(e*e for e in errs) / len(errs))
            bias = sum(errs) / len(errs)
            metric_rows.append(
                f"<tr><td>{state}</td><td>{labels[sk]}</td>"
                f"<td>{mae:.4f}</td><td>{rmse:.4f}</td><td>{bias:+.4f}</td></tr>"
            )

    # panel sections
    sections = []
    for state in states:
        key = f"{state}_values"
        sections.append(f"""
    <section class="panel">
      <div class="panel-head">
        <div>
          <div class="panel-title">{state} — value prediction</div>
          <div class="window-label" data-window="{key}"></div>
        </div>
        <div class="tools">
          <button type="button" data-action="zoom-in"  data-target="{key}">Zoom in</button>
          <button type="button" data-action="zoom-out" data-target="{key}">Zoom out</button>
          <button type="button" data-action="reset"    data-target="{key}">Reset</button>
        </div>
      </div>
      <canvas data-chart="{key}" data-state="{state}" height="360"></canvas>
      <div class="readout" data-readout="{key}"></div>
    </section>""")

    legend_html = "".join(
        f'<span class="legend-item">'
        f'<span class="swatch" style="background:{colors[s]}"></span>{labels[s]}'
        f'</span>'
        for s in ["true"] + series_keys
    )

    first = html_mod.escape(plot_data["timestamps"][0])
    last  = html_mod.escape(plot_data["timestamps"][-1])

    script = _make_script(data_json, colors, series_keys)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>AHU v5 SINDy-MIMO — Validation Predictions</title>
  <style>
    :root{{font-family:Arial,Helvetica,sans-serif;background:#f3f4f6;color:#111827}}
    body{{margin:0;padding:28px}}
    main{{max-width:1120px;margin:0 auto}}
    h1{{margin:0 0 8px;font-size:26px}}
    .meta{{color:#4b5563;font-size:14px;line-height:1.6}}
    .legend{{display:flex;gap:18px;margin:12px 0;flex-wrap:wrap;font-size:14px}}
    .legend-item{{display:inline-flex;align-items:center;gap:7px}}
    .swatch{{width:28px;height:3px;border-radius:999px;display:inline-block}}
    .panel{{background:#fff;border:1px solid #d1d5db;border-radius:8px;margin:14px 0;padding:14px 16px 12px}}
    .panel-head{{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:8px}}
    .panel-title{{font-size:16px;font-weight:700}}
    .window-label{{color:#6b7280;font-size:12px;margin-top:3px}}
    .tools{{display:flex;gap:6px}}
    button{{border:1px solid #cbd5e1;background:#fff;color:#111827;border-radius:6px;padding:6px 9px;font-size:12px;cursor:pointer}}
    button:hover{{background:#f3f4f6}}
    canvas{{width:100%;height:360px;display:block;border:1px solid #e5e7eb;cursor:grab;touch-action:none}}
    canvas:active{{cursor:grabbing}}
    .readout{{min-height:18px;margin-top:8px;color:#374151;font-size:13px;font-variant-numeric:tabular-nums}}
    table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid #d1d5db;border-radius:8px;overflow:hidden;font-size:14px;margin-bottom:18px}}
    th,td{{padding:10px 12px;border-bottom:1px solid #e5e7eb;text-align:right;white-space:nowrap}}
    th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
    th{{background:#f9fafb;font-weight:700}}
  </style>
</head>
<body>
<main>
  <h1>AHU v5 SINDy-MIMO — Validation Predictions</h1>
  <p class="meta">
    Outputs: T_rec, T_coil, T_sup &nbsp;|&nbsp;
    Exogenous (real): T_out, T_ret &nbsp;|&nbsp;
    Lasso &alpha;={alpha} &nbsp;|&nbsp;
    Time range: {first} to {last} &nbsp;|&nbsp;
    Samples: {len(plot_data['timestamps'])}
  </p>
  <div class="legend">{legend_html}</div>
  <table>
    <thead><tr><th>State</th><th>Model</th><th>MAE</th><th>RMSE</th><th>Bias</th></tr></thead>
    <tbody>{''.join(metric_rows)}</tbody>
  </table>
  {''.join(sections)}
</main>
{script}
</body>
</html>
"""


def _make_script(data_json: str, colors: dict, series_keys: list[str]) -> str:
    colors_json = json.dumps(colors)
    skeys_json  = json.dumps(series_keys)
    # JS is kept as a plain string (no f-string) to avoid brace-escaping issues.
    js = (
        "const D=" + data_json + ";\n"
        "const COLORS=" + colors_json + ";\n"
        "const SKEYS=" + skeys_json + ";\n"
        "const charts=new Map();\n"
        "function clamp(v,a,b){return Math.max(a,Math.min(b,v));}\n"
        "function fmtTs(t){return t.slice(0,16);}\n"
        "function makeChart(canvas){\n"
        "  const n=D.timestamps.length;\n"
        "  return{key:canvas.dataset.chart,canvas,ctx:canvas.getContext('2d'),\n"
        "    state:canvas.dataset.state,start:0,end:n-1,\n"
        "    hoverIndex:null,dragging:false,dragStartX:0,\n"
        "    dragStartStart:0,dragStartEnd:n-1,cssWidth:0,cssHeight:0};}\n"
        "function chartArea(c){return{left:66,top:22,width:c.cssWidth-90,height:c.cssHeight-74};}\n"
        "function visBounds(c){\n"
        "  const sd=D.states[c.state];\n"
        "  const l=Math.max(0,Math.floor(c.start)),r=Math.min(D.timestamps.length-1,Math.ceil(c.end));\n"
        "  let mn=Infinity,mx=-Infinity;\n"
        "  for(const s of['true',...SKEYS]){const a=sd[s];for(let i=l;i<=r;i++){if(a[i]<mn)mn=a[i];if(a[i]>mx)mx=a[i];}}\n"
        "  const mg=(mx-mn)*0.1||0.5;return{min:mn-mg,max:mx+mg};}\n"
        "function resizeCanvas(c){\n"
        "  const rect=c.canvas.getBoundingClientRect();const dpr=window.devicePixelRatio||1;\n"
        "  const w=Math.max(320,Math.floor(rect.width));const h=Number(c.canvas.getAttribute('height'))||360;\n"
        "  c.canvas.width=Math.floor(w*dpr);c.canvas.height=Math.floor(h*dpr);\n"
        "  c.ctx.setTransform(dpr,0,0,dpr,0,0);c.cssWidth=w;c.cssHeight=h;}\n"
        "function xFor(c,i,a){return a.left+((i-c.start)/(c.end-c.start||1))*a.width;}\n"
        "function iForX(c,x,a){return c.start+((x-a.left)/a.width)*(c.end-c.start);}\n"
        "function yFor(v,a,b){return a.top+a.height-((v-b.min)/(b.max-b.min||1))*a.height;}\n"
        "function drawLine(ctx,c,arr,a,b,color,lw){\n"
        "  ctx.save();ctx.strokeStyle=color;ctx.lineWidth=lw;ctx.beginPath();\n"
        "  const l=Math.max(0,Math.floor(c.start)),r=Math.min(D.timestamps.length-1,Math.ceil(c.end));\n"
        "  let started=false;\n"
        "  for(let i=l;i<=r;i++){const x=xFor(c,i,a);const y=yFor(arr[i],a,b);\n"
        "    if(!started){ctx.moveTo(x,y);started=true;}else{ctx.lineTo(x,y);}}\n"
        "  ctx.stroke();ctx.restore();}\n"
        "function drawChart(c){\n"
        "  const ctx=c.ctx;const w=c.cssWidth;const h=c.cssHeight;if(!w||!h)return;\n"
        "  const a=chartArea(c);const b=visBounds(c);const sd=D.states[c.state];\n"
        "  ctx.clearRect(0,0,w,h);ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);\n"
        "  ctx.strokeStyle='#e5e7eb';ctx.lineWidth=1;\n"
        "  ctx.fillStyle='#6b7280';ctx.font='12px Arial';ctx.textAlign='right';ctx.textBaseline='middle';\n"
        "  for(let t=0;t<=4;t++){\n"
        "    const v=b.min+(b.max-b.min)*t/4;const y=yFor(v,a,b);\n"
        "    ctx.beginPath();ctx.moveTo(a.left,y);ctx.lineTo(a.left+a.width,y);ctx.stroke();\n"
        "    ctx.fillText(v.toFixed(2),a.left-8,y);}\n"
        "  drawLine(ctx,c,sd['true'],a,b,'#111827',1.8);\n"
        "  for(const s of SKEYS)drawLine(ctx,c,sd[s],a,b,COLORS[s],1.4);\n"
        "  ctx.strokeStyle='#9ca3af';ctx.strokeRect(a.left,a.top,a.width,a.height);\n"
        "  const li=Math.max(0,Math.round(c.start)),ri=Math.min(D.timestamps.length-1,Math.round(c.end));\n"
        "  ctx.fillStyle='#6b7280';ctx.textAlign='left';ctx.textBaseline='top';\n"
        "  ctx.fillText(fmtTs(D.timestamps[li]),a.left,a.top+a.height+12);\n"
        "  ctx.textAlign='right';ctx.fillText(fmtTs(D.timestamps[ri]),a.left+a.width,a.top+a.height+12);\n"
        "  if(c.hoverIndex!==null){\n"
        "    const i=clamp(c.hoverIndex,li,ri);const x=xFor(c,i,a);\n"
        "    ctx.strokeStyle='#374151';ctx.beginPath();ctx.moveTo(x,a.top);ctx.lineTo(x,a.top+a.height);ctx.stroke();\n"
        "    const parts=[fmtTs(D.timestamps[i]),'True='+sd['true'][i].toFixed(3)];\n"
        "    for(const s of SKEYS)parts.push(s+'='+sd[s][i].toFixed(3));\n"
        "    const ro=document.querySelector('[data-readout=\"'+c.key+'\"]');if(ro)ro.textContent=parts.join(' | ');}\n"
        "  const lbl=document.querySelector('[data-window=\"'+c.key+'\"]');\n"
        "  if(lbl)lbl.textContent=fmtTs(D.timestamps[li])+' to '+fmtTs(D.timestamps[ri])+' ('+(ri-li+1)+' pts)';}\n"
        "function zoom(c,ci,factor){\n"
        "  const n=D.timestamps.length;const os=c.end-c.start;const ns=clamp(os*factor,20,n-1);\n"
        "  let s=ci-ns/2,e=ci+ns/2;\n"
        "  if(s<0){e-=s;s=0;}if(e>n-1){s-=e-(n-1);e=n-1;}\n"
        "  c.start=clamp(s,0,n-1);c.end=clamp(e,0,n-1);drawChart(c);}\n"
        "function attachEvents(c){\n"
        "  c.canvas.addEventListener('wheel',(e)=>{\n"
        "    e.preventDefault();const rect=c.canvas.getBoundingClientRect();\n"
        "    const ci=clamp(iForX(c,e.clientX-rect.left,chartArea(c)),0,D.timestamps.length-1);\n"
        "    zoom(c,ci,e.deltaY<0?0.75:1.35);},{passive:false});\n"
        "  c.canvas.addEventListener('pointerdown',(e)=>{\n"
        "    c.dragging=true;c.dragStartX=e.clientX;c.dragStartStart=c.start;\n"
        "    c.dragStartEnd=c.end;c.canvas.setPointerCapture(e.pointerId);});\n"
        "  c.canvas.addEventListener('pointermove',(e)=>{\n"
        "    const rect=c.canvas.getBoundingClientRect();const a=chartArea(c);\n"
        "    const x=e.clientX-rect.left;\n"
        "    if(c.dragging){\n"
        "      const dx=e.clientX-c.dragStartX;const span=c.dragStartEnd-c.dragStartStart;\n"
        "      const shift=-dx/a.width*span;\n"
        "      c.start=c.dragStartStart+shift;c.end=c.dragStartEnd+shift;\n"
        "      if(c.start<0){c.end-=c.start;c.start=0;}\n"
        "      if(c.end>D.timestamps.length-1){c.start-=c.end-(D.timestamps.length-1);c.end=D.timestamps.length-1;}}\n"
        "    const nh=clamp(Math.round(iForX(c,x,a)),0,D.timestamps.length-1);\n"
        "    if(c.dragging||nh!==c.hoverIndex){c.hoverIndex=nh;drawChart(c);}});\n"
        "  c.canvas.addEventListener('pointerup',(e)=>{\n"
        "    c.dragging=false;if(c.canvas.hasPointerCapture(e.pointerId))c.canvas.releasePointerCapture(e.pointerId);});\n"
        "  c.canvas.addEventListener('pointerleave',()=>{\n"
        "    c.dragging=false;c.hoverIndex=null;\n"
        "    const ro=document.querySelector('[data-readout=\"'+c.key+'\"]');if(ro)ro.textContent='';\n"
        "    drawChart(c);});\n"
        "  c.canvas.addEventListener('dblclick',()=>{c.start=0;c.end=D.timestamps.length-1;drawChart(c);});}\n"
        "function init(){\n"
        "  document.querySelectorAll('canvas[data-chart]').forEach((canvas)=>{\n"
        "    const c=makeChart(canvas);charts.set(c.key,c);attachEvents(c);resizeCanvas(c);drawChart(c);});\n"
        "  document.querySelectorAll('button[data-action]').forEach((btn)=>btn.addEventListener('click',()=>{\n"
        "    const c=charts.get(btn.dataset.target);if(!c)return;\n"
        "    const ci=(c.start+c.end)/2;\n"
        "    if(btn.dataset.action==='zoom-in')zoom(c,ci,0.5);\n"
        "    if(btn.dataset.action==='zoom-out')zoom(c,ci,2.0);\n"
        "    if(btn.dataset.action==='reset'){c.start=0;c.end=D.timestamps.length-1;drawChart(c);}}));\n"
        "  window.addEventListener('resize',()=>{for(const c of charts.values()){resizeCanvas(c);drawChart(c);}});}\n"
        "init();\n"
    )
    return "<script>\n" + js + "</script>"

# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    split_dir = Path(args.split_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    train_rows = load_rows(split_dir / "train_5min.csv")
    val_rows   = load_rows(split_dir / "validation_5min.csv")

    print("Building samples ...")
    train_samples = build_samples(train_rows, args.freq_min)
    val_samples   = build_samples(val_rows,   args.freq_min)
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Val   samples: {len(val_samples)}")

    print("Training Lasso models ...")
    models = train_models(train_samples, args.alpha, args.lasso_iter)

    print("Building report ...")
    report_lines = build_report(train_samples, val_samples, models, args.alpha, args)
    report_path  = out_dir / "ahu_v5_sindy_mimo_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote {report_path}")

    print("Generating HTML visualisation ...")
    plot_data = build_plot_data(val_samples, models, args.freq_min)
    html_str  = render_html(plot_data, out_dir / "ahu_v5_sindy_mimo_validation.html", args.alpha)
    html_path = out_dir / "ahu_v5_sindy_mimo_validation.html"
    html_path.write_text(html_str, encoding="utf-8")
    print(f"Wrote {html_path}")

    print("Done.")


if __name__ == "__main__":
    main()
