"""
AHU v7 — Physics-structured SINDy with two subsystems.

Subsystem 1 — Heat Exchanger (produces T_rec):
  States:   T_rec                        (auto-regressed in rollout)
  Exog:     T_out, T_ret                 (always real)
  Control:  u_heat_recovery              (always real)
  Feature vector z1 = [T_rec, T_out, T_ret, u_heat_recovery]  — 4 variables
  Library:  degree-1 (4) + degree-2 (10) + time (4) = 18 features
  Target:   delta_T_rec

Subsystem 2 — Heating Coil (produces T_sup):
  States:   T_sup                        (auto-regressed in rollout)
  Exog:     T_rec (predicted by SS1), T_coil  (always real — sensor, not predicted)
  Control:  u_heat                       (always real)
  Feature vector z2 = [T_sup, T_rec, T_coil, u_heat]  — 4 variables
  Library:  degree-1 (4) + degree-2 (10) + time (4) = 18 features
  Target:   delta_T_sup

Multi-step rollout:
  - T_out, T_ret, T_coil: real values at each future step (exogenous)
  - u_heat_recovery, u_heat: real values at each future step (known controls)
  - T_rec: auto-regressed from SS1 predictions
  - T_sup: auto-regressed from SS2 predictions (uses predicted T_rec)

Evaluation horizons: H = 1, 4, 12 steps (5 min, 20 min, 60 min)

Gap handling: only rollout windows where all H steps are consecutive are evaluated.
"""

from __future__ import annotations

import argparse
import csv
import html as html_mod
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

# ── constants ─────────────────────────────────────────────────────────────────

# Subsystem 1 — Heat Exchanger
SS1_STATE = "T_rec"
SS1_EXOG  = ["T_out", "T_ret"]
SS1_CTRL  = ["u_heat_recovery"]
SS1_VARS  = [SS1_STATE] + SS1_EXOG + SS1_CTRL   # 4 variables → 18 features

# Subsystem 2 — Heating Coil
SS2_STATE = "T_sup"
SS2_EXOG  = ["T_rec", "T_coil"]   # T_rec comes from SS1 in rollout; T_coil always real
SS2_CTRL  = ["u_heat"]
SS2_VARS  = [SS2_STATE] + SS2_EXOG + SS2_CTRL   # 4 variables → 18 features

# All columns needed from CSV
ALL_COLS = ["T_rec", "T_sup", "T_out", "T_ret", "T_coil",
            "u_heat_recovery", "u_heat"]

HORIZONS = [1, 4, 12]

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AHU v7 physics-structured SINDy")
    p.add_argument("--split-dir",   default="data_preprocessing/splits")
    p.add_argument("--out-dir",     default="model_outputs/v7")
    p.add_argument("--freq-min",    type=int,   default=5)
    p.add_argument("--alpha-ss1",   type=float, default=0.0001,
                   help="Lasso alpha for subsystem 1 (heat exchanger).")
    p.add_argument("--alpha-ss2",   type=float, default=0.0001,
                   help="Lasso alpha for subsystem 2 (heating coil).")
    p.add_argument("--lasso-iter",  type=int,   default=500)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()

# ── data ──────────────────────────────────────────────────────────────────────

def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@dataclass
class Sample:
    timestamp:   str
    vals_now:    dict[str, float]   # all ALL_COLS at t
    vals_next:   dict[str, float]   # all ALL_COLS at t+1
    delta_rec:   float              # T_rec(t+1) - T_rec(t)
    delta_sup:   float              # T_sup(t+1) - T_sup(t)
    max_horizon: int                # consecutive steps available after this sample


def build_samples(rows: list[dict[str, str]], freq_min: int) -> list[Sample]:
    step   = timedelta(minutes=freq_min)
    parsed = [datetime.fromisoformat(r["timestamp"]) for r in rows]

    consecutive = [parsed[i + 1] - parsed[i] == step for i in range(len(rows) - 1)]

    run_fwd = [0] * len(rows)
    for i in range(len(rows) - 2, -1, -1):
        run_fwd[i] = (1 + run_fwd[i + 1]) if consecutive[i] else 0

    samples: list[Sample] = []
    for i in range(len(rows) - 1):
        if not consecutive[i]:
            continue
        try:
            now  = {c: float(rows[i][c])     for c in ALL_COLS}
            nxt  = {c: float(rows[i + 1][c]) for c in ALL_COLS}
        except (KeyError, ValueError):
            continue
        samples.append(Sample(
            timestamp   = rows[i + 1]["timestamp"],
            vals_now    = now,
            vals_next   = nxt,
            delta_rec   = nxt["T_rec"] - now["T_rec"],
            delta_sup   = nxt["T_sup"] - now["T_sup"],
            max_horizon = run_fwd[i],
        ))
    return samples

# ── SINDy feature library ─────────────────────────────────────────────────────

def time_feats(ts: str) -> list[float]:
    dt    = datetime.fromisoformat(ts)
    mday  = dt.hour * 60 + dt.minute
    d_ang = 2 * math.pi * mday / 1440.0
    y_ang = 2 * math.pi * dt.timetuple().tm_yday / 365.25
    return [math.sin(d_ang), math.cos(d_ang), math.sin(y_ang), math.cos(y_ang)]


def build_theta(var_vals: list[float], ts: str) -> list[float]:
    """degree-1 (n) + degree-2 (n*(n+1)/2) + time (4) features."""
    feats: list[float] = list(var_vals)
    for i in range(len(var_vals)):
        for j in range(i, len(var_vals)):
            feats.append(var_vals[i] * var_vals[j])
    feats.extend(time_feats(ts))
    return feats


def theta_names(var_names: list[str]) -> list[str]:
    names = list(var_names)
    for i, vi in enumerate(var_names):
        for vj in var_names[i:]:
            names.append(f"{vi}*{vj}")
    names += ["sin_day", "cos_day", "sin_year", "cos_year"]
    return names


def theta_ss1(s: Sample) -> list[float]:
    return build_theta([s.vals_now[v] for v in SS1_VARS], s.timestamp)


def theta_ss2(s: Sample) -> list[float]:
    return build_theta([s.vals_now[v] for v in SS2_VARS], s.timestamp)

# ── Lasso (coordinate descent) ────────────────────────────────────────────────

@dataclass
class LassoModel:
    coef:      list[float]
    intercept: float
    means:     list[float]
    stds:      list[float]
    alpha:     float


def _soft(x: float, lam: float) -> float:
    return max(0.0, x - lam) - max(0.0, -x - lam)


def fit_lasso(
    X: list[list[float]], y: list[float],
    alpha: float, max_iter: int, tol: float = 1e-6,
) -> LassoModel:
    n, p = len(X), len(X[0])
    means = [sum(X[i][j] for i in range(n)) / n for j in range(p)]
    stds: list[float] = []
    for j in range(p):
        var = sum((X[i][j] - means[j]) ** 2 for i in range(n)) / max(1, n - 1)
        stds.append(math.sqrt(var) if var > 1e-12 else 1.0)

    Xs_col = [[(X[i][j] - means[j]) / stds[j] for i in range(n)] for j in range(p)]
    col_norm2 = [sum(v * v for v in Xs_col[j]) for j in range(p)]
    coef = [0.0] * p
    intercept = sum(y) / n
    r = [y[i] - intercept for i in range(n)]
    lam = alpha * n

    for _ in range(max_iter):
        max_change = 0.0
        for j in range(p):
            cj  = coef[j]
            col = Xs_col[j]
            if cj != 0.0:
                for i in range(n):
                    r[i] += cj * col[i]
            rho = sum(r[i] * col[i] for i in range(n))
            new_cj = 0.0 if col_norm2[j] < 1e-12 else _soft(rho, lam) / col_norm2[j]
            if new_cj != 0.0:
                for i in range(n):
                    r[i] -= new_cj * col[i]
            change = abs(new_cj - cj)
            if change > max_change:
                max_change = change
            coef[j] = new_cj
        if max_change < tol:
            break

    intercept = sum(
        y[i] - sum(coef[j] * Xs_col[j][i] for j in range(p)) for i in range(n)
    ) / n
    return LassoModel(coef, intercept, means, stds, alpha)


def predict_delta(model: LassoModel, theta: list[float]) -> float:
    xs = [(theta[j] - model.means[j]) / model.stds[j] for j in range(len(theta))]
    return model.intercept + sum(c * v for c, v in zip(model.coef, xs))

# ── rollout ───────────────────────────────────────────────────────────────────

@dataclass
class RolloutPred:
    t_rec: float
    t_sup: float


def rollout(
    ss1: LassoModel,
    ss2: LassoModel,
    samples: list[Sample],
    horizon: int,
) -> tuple[list[RolloutPred], list[int]]:
    """
    Returns (predictions, valid_start_indices).
    Only windows where all H steps are consecutive are included.
    """
    preds:   list[RolloutPred] = []
    indices: list[int]         = []

    for i in range(len(samples) - horizon):
        if samples[i].max_horizon < horizon:
            continue

        t_rec = samples[i].vals_now["T_rec"]
        t_sup = samples[i].vals_now["T_sup"]
        ts    = samples[i].timestamp

        for h in range(horizon):
            src = samples[i + h]
            # real exog/ctrl values at step t+h (from data)
            t_out  = src.vals_now["T_out"]
            t_ret  = src.vals_now["T_ret"]
            t_coil = src.vals_now["T_coil"]
            u_hr   = src.vals_now["u_heat_recovery"]
            u_heat = src.vals_now["u_heat"]

            # SS1: predict delta_T_rec
            z1    = [t_rec, t_out, t_ret, u_hr]
            th1   = build_theta(z1, ts)
            t_rec = t_rec + predict_delta(ss1, th1)

            # SS2: predict delta_T_sup using updated T_rec
            z2    = [t_sup, t_rec, t_coil, u_heat]
            th2   = build_theta(z2, ts)
            t_sup = t_sup + predict_delta(ss2, th2)

            ts = src.timestamp

        preds.append(RolloutPred(t_rec, t_sup))
        indices.append(i)

    return preds, indices

# ── metrics ───────────────────────────────────────────────────────────────────

def metrics(true: list[float], pred: list[float]) -> dict[str, float]:
    n    = len(true)
    errs = [p - t for t, p in zip(true, pred)]
    mae  = sum(abs(e) for e in errs) / n
    rmse = math.sqrt(sum(e * e for e in errs) / n)
    mean_y = sum(true) / n
    sst  = sum((t - mean_y) ** 2 for t in true)
    sse  = sum(e * e for e in errs)
    return {
        "n": n, "mae": mae, "rmse": rmse,
        "r2": 1.0 - sse / sst if sst > 0 else float("nan"),
        "bias": sum(errs) / n,
    }

# ── report ────────────────────────────────────────────────────────────────────

def build_report(
    train_s: list[Sample],
    val_s:   list[Sample],
    ss1:     LassoModel,
    ss2:     LassoModel,
    args:    argparse.Namespace,
) -> list[str]:
    names1 = theta_names(SS1_VARS)
    names2 = theta_names(SS2_VARS)

    lines = [
        "AHU v7 Physics-structured SINDy report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Frequency: {args.freq_min} min",
        "",
        "=== Subsystem 1: Heat Exchanger ===",
        f"  Variables: {SS1_VARS}",
        f"  Feature library: {len(names1)} candidates (4 vars: 4+10+4)",
        f"  Lasso alpha: {args.alpha_ss1}",
        f"  Target: delta_T_rec",
    ]
    active1 = [(names1[j], ss1.coef[j]) for j in range(len(ss1.coef)) if abs(ss1.coef[j]) > 1e-10]
    lines.append(f"  Active terms: {len(active1)}/{len(ss1.coef)}  intercept={ss1.intercept:+.6f}")
    for name, c in sorted(active1, key=lambda x: -abs(x[1])):
        lines.append(f"    {name:35s}  {c:+.8f}")

    lines += [
        "",
        "=== Subsystem 2: Heating Coil ===",
        f"  Variables: {SS2_VARS}",
        f"  Feature library: {len(names2)} candidates (4 vars: 4+10+4)",
        f"  Lasso alpha: {args.alpha_ss2}",
        f"  Target: delta_T_sup",
        f"  Note: T_rec in SS2 is predicted by SS1 during rollout; T_coil always real",
    ]
    active2 = [(names2[j], ss2.coef[j]) for j in range(len(ss2.coef)) if abs(ss2.coef[j]) > 1e-10]
    lines.append(f"  Active terms: {len(active2)}/{len(ss2.coef)}  intercept={ss2.intercept:+.6f}")
    for name, c in sorted(active2, key=lambda x: -abs(x[1])):
        lines.append(f"    {name:35s}  {c:+.8f}")

    lines += ["", "=== Prediction metrics (gap-aware rollout) ==="]

    for split_name, samples in [("train", train_s), ("validation", val_s)]:
        lines.append(f"\n--- {split_name} ({len(samples)} samples) ---")
        for H in HORIZONS:
            if H >= len(samples):
                continue
            preds, idx = rollout(ss1, ss2, samples, H)
            n_valid = len(idx)
            lines.append(f"  H={H:2d} ({H * args.freq_min} min ahead, {n_valid} valid windows)")

            for state, getter, pers_key in [
                ("T_rec", lambda p: p.t_rec, "T_rec"),
                ("T_sup", lambda p: p.t_sup, "T_sup"),
            ]:
                tv    = [samples[i + H].vals_now[pers_key] for i in idx]
                pv    = [getter(p) for p in preds]
                persv = [samples[i].vals_now[pers_key] for i in idx]
                mp    = metrics(tv, pv)
                mpe   = metrics(tv, persv)
                lines.append(
                    f"    {state:8s} "
                    f"Persistence MAE={mpe['mae']:.4f} RMSE={mpe['rmse']:.4f} | "
                    f"SINDy MAE={mp['mae']:.4f} RMSE={mp['rmse']:.4f} R2={mp['r2']:.4f}"
                )
    return lines

# ── HTML ──────────────────────────────────────────────────────────────────────

def build_plot_data(
    val_s: list[Sample],
    ss1:   LassoModel,
    ss2:   LassoModel,
    freq_min: int,
) -> dict:
    raw = {H: rollout(ss1, ss2, val_s, H) for H in HORIZONS}

    common_idx: set[int] | None = None
    for H in HORIZONS:
        _, idx = raw[H]
        common_idx = set(idx) if common_idx is None else common_idx & set(idx)
    base_idx = sorted(common_idx or [])

    timestamps = [val_s[i].timestamp for i in base_idx]

    series: dict = {}
    for state, pred_attr in [("T_rec", "t_rec"), ("T_sup", "t_sup")]:
        sd: dict[str, list[float]] = {}
        for H in HORIZONS:
            preds, idx = raw[H]
            idx_map = {i: k for k, i in enumerate(idx)}
            sd[f"true_h{H}"]        = [val_s[i + H].vals_now[state]            for i in base_idx]
            sd[f"persistence_h{H}"] = [val_s[i].vals_now[state]                for i in base_idx]
            sd[f"sindy_h{H}"]       = [getattr(preds[idx_map[i]], pred_attr)   for i in base_idx]
        series[state] = sd

    return {"timestamps": timestamps, "states": series,
            "freqMin": freq_min, "horizons": HORIZONS}


def render_html(plot_data: dict, args: argparse.Namespace) -> str:
    data_json = json.dumps(plot_data, separators=(",", ":")).replace("</", "<\\/")
    horizons  = plot_data["horizons"]
    freq_min  = plot_data["freqMin"]
    states    = list(plot_data["states"].keys())

    colors = {"true": "#111827", "persistence": "#9ca3af", "sindy": "#2563eb"}
    labels = {"true": "True", "persistence": "Persistence", "sindy": "SINDy"}

    # metric table
    metric_rows = []
    for H in horizons:
        for state in states:
            sd = plot_data["states"][state]
            tv = sd[f"true_h{H}"]
            for mk in ["persistence", "sindy"]:
                pv   = sd[f"{mk}_h{H}"]
                errs = [v - t for t, v in zip(tv, pv)]
                mae  = sum(abs(e) for e in errs) / len(errs)
                rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
                bias = sum(errs) / len(errs)
                metric_rows.append(
                    f"<tr><td>{state}</td>"
                    f"<td>{labels[mk]} H={H} ({H*freq_min}min)</td>"
                    f"<td>{mae:.4f}</td><td>{rmse:.4f}</td><td>{bias:+.4f}</td></tr>"
                )

    sections = []
    for state in states:
        for H in horizons:
            key   = f"{state}_h{H}"
            title = f"{state} — H={H} ({H*freq_min} min ahead)"
            sections.append(
                f'<section class="panel">'
                f'<div class="panel-head"><div>'
                f'<div class="panel-title">{title}</div>'
                f'<div class="window-label" data-window="{key}"></div>'
                f'</div><div class="tools">'
                f'<button type="button" data-action="zoom-in" data-target="{key}">Zoom in</button>'
                f'<button type="button" data-action="zoom-out" data-target="{key}">Zoom out</button>'
                f'<button type="button" data-action="reset" data-target="{key}">Reset</button>'
                f'</div></div>'
                f'<canvas data-chart="{key}" data-state="{state}" data-horizon="{H}" height="300"></canvas>'
                f'<div class="readout" data-readout="{key}"></div>'
                f'</section>'
            )

    legend_html = "".join(
        f'<span class="legend-item">'
        f'<span class="swatch" style="background:{colors[k]}"></span>{labels[k]}'
        f'</span>'
        for k in ["true", "persistence", "sindy"]
    )

    first = html_mod.escape(plot_data["timestamps"][0])
    last  = html_mod.escape(plot_data["timestamps"][-1])
    colors_json = json.dumps(colors)

    js = (
        "const D=" + data_json + ";\n"
        "const COLORS=" + colors_json + ";\n"
        "const HORIZONS=" + json.dumps(horizons) + ";\n"
        "const MKEYS=['persistence','sindy'];\n"
        "const charts=new Map();\n"
        "function clamp(v,a,b){return Math.max(a,Math.min(b,v));}\n"
        "function fmtTs(t){return t.slice(0,16);}\n"
        "function makeChart(canvas){\n"
        "  const n=D.timestamps.length;\n"
        "  const H=Number(canvas.dataset.horizon);\n"
        "  const state=canvas.dataset.state;\n"
        "  const sd=D.states[state];\n"
        "  return{key:canvas.dataset.chart,canvas,ctx:canvas.getContext('2d'),\n"
        "    state,H,\n"
        "    trueArr:sd['true_h'+H],\n"
        "    persArr:sd['persistence_h'+H],\n"
        "    sindyArr:sd['sindy_h'+H],\n"
        "    start:0,end:n-1,hoverIndex:null,dragging:false,\n"
        "    dragStartX:0,dragStartStart:0,dragStartEnd:n-1,cssWidth:0,cssHeight:0};}\n"
        "function chartArea(c){return{left:66,top:22,width:c.cssWidth-90,height:c.cssHeight-54};}\n"
        "function visBounds(c){\n"
        "  const l=Math.max(0,Math.floor(c.start)),r=Math.min(D.timestamps.length-1,Math.ceil(c.end));\n"
        "  let mn=Infinity,mx=-Infinity;\n"
        "  for(const arr of[c.trueArr,c.persArr,c.sindyArr]){\n"
        "    for(let i=l;i<=r;i++){if(arr[i]<mn)mn=arr[i];if(arr[i]>mx)mx=arr[i];}}\n"
        "  const mg=(mx-mn)*0.1||0.5;return{min:mn-mg,max:mx+mg};}\n"
        "function resizeCanvas(c){\n"
        "  const rect=c.canvas.getBoundingClientRect();const dpr=window.devicePixelRatio||1;\n"
        "  const w=Math.max(320,Math.floor(rect.width));const h=Number(c.canvas.getAttribute('height'))||300;\n"
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
        "  const a=chartArea(c);const b=visBounds(c);\n"
        "  ctx.clearRect(0,0,w,h);ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);\n"
        "  ctx.strokeStyle='#e5e7eb';ctx.lineWidth=1;\n"
        "  ctx.fillStyle='#6b7280';ctx.font='12px Arial';ctx.textAlign='right';ctx.textBaseline='middle';\n"
        "  for(let t=0;t<=4;t++){\n"
        "    const v=b.min+(b.max-b.min)*t/4;const y=yFor(v,a,b);\n"
        "    ctx.beginPath();ctx.moveTo(a.left,y);ctx.lineTo(a.left+a.width,y);ctx.stroke();\n"
        "    ctx.fillText(v.toFixed(2),a.left-8,y);}\n"
        "  drawLine(ctx,c,c.trueArr,a,b,COLORS.true,1.8);\n"
        "  drawLine(ctx,c,c.persArr,a,b,COLORS.persistence,1.4);\n"
        "  drawLine(ctx,c,c.sindyArr,a,b,COLORS.sindy,1.5);\n"
        "  ctx.strokeStyle='#9ca3af';ctx.strokeRect(a.left,a.top,a.width,a.height);\n"
        "  const li=Math.max(0,Math.round(c.start)),ri=Math.min(D.timestamps.length-1,Math.round(c.end));\n"
        "  ctx.fillStyle='#6b7280';ctx.textAlign='left';ctx.textBaseline='top';\n"
        "  ctx.fillText(fmtTs(D.timestamps[li]),a.left,a.top+a.height+8);\n"
        "  ctx.textAlign='right';ctx.fillText(fmtTs(D.timestamps[ri]),a.left+a.width,a.top+a.height+8);\n"
        "  if(c.hoverIndex!==null){\n"
        "    const i=clamp(c.hoverIndex,li,ri);const x=xFor(c,i,a);\n"
        "    ctx.strokeStyle='#374151';ctx.lineWidth=1;\n"
        "    ctx.beginPath();ctx.moveTo(x,a.top);ctx.lineTo(x,a.top+a.height);ctx.stroke();\n"
        "    const ro=document.querySelector('[data-readout=\"'+c.key+'\"]');\n"
        "    if(ro)ro.textContent=fmtTs(D.timestamps[i])\n"
        "      +' | True='+c.trueArr[i].toFixed(3)\n"
        "      +' | Pers='+c.persArr[i].toFixed(3)\n"
        "      +' | SINDy='+c.sindyArr[i].toFixed(3);}\n"
        "  const lbl=document.querySelector('[data-window=\"'+c.key+'\"]');\n"
        "  if(lbl)lbl.textContent=fmtTs(D.timestamps[li])+' to '+fmtTs(D.timestamps[ri])+' ('+(ri-li+1)+' pts)';}\n"
        "function zoom(c,ci,factor){\n"
        "  const n=D.timestamps.length;const os=c.end-c.start;const ns=clamp(os*factor,20,n-1);\n"
        "  let s=ci-ns/2,e=ci+ns/2;\n"
        "  if(s<0){e-=s;s=0;}if(e>n-1){s-=e-(n-1);e=n-1;}\n"
        "  c.start=clamp(s,0,n-1);c.end=clamp(e,0,n-1);drawChart(c);}\n"
        "function attachEvents(c){\n"
        "  c.canvas.addEventListener('wheel',(e)=>{\n"
        "    e.preventDefault();\n"
        "    const rect=c.canvas.getBoundingClientRect();\n"
        "    const ci=clamp(iForX(c,e.clientX-rect.left,chartArea(c)),0,D.timestamps.length-1);\n"
        "    zoom(c,ci,e.deltaY<0?0.75:1.35);},{passive:false});\n"
        "  c.canvas.addEventListener('pointerdown',(e)=>{\n"
        "    c.dragging=true;c.dragStartX=e.clientX;\n"
        "    c.dragStartStart=c.start;c.dragStartEnd=c.end;\n"
        "    c.canvas.setPointerCapture(e.pointerId);});\n"
        "  c.canvas.addEventListener('pointermove',(e)=>{\n"
        "    const rect=c.canvas.getBoundingClientRect();const a=chartArea(c);\n"
        "    const x=e.clientX-rect.left;\n"
        "    if(c.dragging){\n"
        "      const dx=e.clientX-c.dragStartX;\n"
        "      const span=c.dragStartEnd-c.dragStartStart;\n"
        "      const shift=-dx/a.width*span;\n"
        "      c.start=c.dragStartStart+shift;c.end=c.dragStartEnd+shift;\n"
        "      if(c.start<0){c.end-=c.start;c.start=0;}\n"
        "      if(c.end>D.timestamps.length-1){c.start-=c.end-(D.timestamps.length-1);c.end=D.timestamps.length-1;}}\n"
        "    const nh=clamp(Math.round(iForX(c,x,a)),0,D.timestamps.length-1);\n"
        "    if(c.dragging||nh!==c.hoverIndex){c.hoverIndex=nh;drawChart(c);}});\n"
        "  c.canvas.addEventListener('pointerup',(e)=>{\n"
        "    c.dragging=false;\n"
        "    if(c.canvas.hasPointerCapture(e.pointerId))c.canvas.releasePointerCapture(e.pointerId);});\n"
        "  c.canvas.addEventListener('pointerleave',()=>{\n"
        "    c.dragging=false;c.hoverIndex=null;\n"
        "    const ro=document.querySelector('[data-readout=\"'+c.key+'\"]');\n"
        "    if(ro)ro.textContent='';drawChart(c);});  \n"
        "  c.canvas.addEventListener('dblclick',()=>{\n"
        "    c.start=0;c.end=D.timestamps.length-1;drawChart(c);});}\n"
        "function init(){\n"
        "  document.querySelectorAll('canvas[data-chart]').forEach((canvas)=>{\n"
        "    const c=makeChart(canvas);charts.set(c.key,c);\n"
        "    attachEvents(c);resizeCanvas(c);drawChart(c);});\n"
        "  document.querySelectorAll('button[data-action]').forEach((btn)=>\n"
        "    btn.addEventListener('click',()=>{\n"
        "      const c=charts.get(btn.dataset.target);if(!c)return;\n"
        "      const ci=(c.start+c.end)/2;\n"
        "      if(btn.dataset.action==='zoom-in')zoom(c,ci,0.5);\n"
        "      if(btn.dataset.action==='zoom-out')zoom(c,ci,2.0);\n"
        "      if(btn.dataset.action==='reset'){c.start=0;c.end=D.timestamps.length-1;drawChart(c);}}));\n"
        "  window.addEventListener('resize',()=>{\n"
        "    for(const c of charts.values()){resizeCanvas(c);drawChart(c);}});}\n"
        "init();\n"
    )

    return (
        "<!doctype html><html lang='en'><head>"
        "<meta charset='utf-8'/>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'/>"
        "<title>AHU v7 Physics-structured SINDy Validation</title>"
        "<style>"
        ":root{font-family:Arial,Helvetica,sans-serif;background:#f3f4f6;color:#111827}"
        "body{margin:0;padding:28px}main{max-width:1120px;margin:0 auto}"
        "h1{margin:0 0 8px;font-size:26px}.meta{color:#4b5563;font-size:14px;line-height:1.6}"
        ".legend{display:flex;gap:18px;margin:12px 0;flex-wrap:wrap;font-size:14px}"
        ".legend-item{display:inline-flex;align-items:center;gap:7px}"
        ".swatch{width:28px;height:3px;border-radius:999px;display:inline-block}"
        ".panel{background:#fff;border:1px solid #d1d5db;border-radius:8px;margin:14px 0;padding:14px 16px 12px}"
        ".panel-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:8px}"
        ".panel-title{font-size:16px;font-weight:700}"
        ".window-label{color:#6b7280;font-size:12px;margin-top:3px}"
        ".tools{display:flex;gap:6px}"
        "button{border:1px solid #cbd5e1;background:#fff;color:#111827;border-radius:6px;padding:6px 9px;font-size:12px;cursor:pointer}"
        "button:hover{background:#f3f4f6}"
        "canvas{width:100%;height:300px;display:block;border:1px solid #e5e7eb;cursor:grab;touch-action:none}"
        "canvas:active{cursor:grabbing}"
        ".readout{min-height:18px;margin-top:8px;color:#374151;font-size:13px;font-variant-numeric:tabular-nums}"
        "table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #d1d5db;"
        "border-radius:8px;overflow:hidden;font-size:14px;margin-bottom:18px}"
        "th,td{padding:10px 12px;border-bottom:1px solid #e5e7eb;text-align:right;white-space:nowrap}"
        "th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}"
        "th{background:#f9fafb;font-weight:700}"
        "</style></head><body><main>"
        "<h1>AHU v7 — Physics-structured SINDy Validation</h1>"
        f"<p class='meta'>"
        f"SS1 (Heat Exchanger): T_rec ← [T_out, T_ret, u_heat_recovery] &nbsp;|&nbsp; "
        f"SS2 (Heating Coil): T_sup ← [T_rec, T_coil, u_heat] &nbsp;|&nbsp; "
        f"α_SS1={args.alpha_ss1} &nbsp;|&nbsp; α_SS2={args.alpha_ss2} &nbsp;|&nbsp; "
        f"Time: {first} to {last} &nbsp;|&nbsp; Samples: {len(plot_data['timestamps'])}"
        f"</p>"
        f"<div class='legend'>{legend_html}</div>"
        "<table><thead><tr>"
        "<th>State</th><th>Model</th><th>MAE</th><th>RMSE</th><th>Bias</th>"
        "</tr></thead><tbody>"
        + "".join(metric_rows)
        + "</tbody></table>"
        + "".join(sections)
        + "</main><script>" + js + "</script></body></html>"
    )

# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    train_rows = load_rows(Path(args.split_dir) / "train_5min.csv")
    val_rows   = load_rows(Path(args.split_dir) / "validation_5min.csv")

    print("Building samples ...")
    train_s = build_samples(train_rows, args.freq_min)
    val_s   = build_samples(val_rows,   args.freq_min)
    print(f"  Train: {len(train_s)}  Val: {len(val_s)}")

    # ── build feature matrices ──
    print("Building feature matrices ...")
    X1_train = [theta_ss1(s) for s in train_s]
    X2_train = [theta_ss2(s) for s in train_s]
    y1_train = [s.delta_rec for s in train_s]
    y2_train = [s.delta_sup for s in train_s]
    print(f"  SS1 features: {len(X1_train[0])}  SS2 features: {len(X2_train[0])}")

    # ── train Lasso ──
    print("\nTraining SS1 (Heat Exchanger) Lasso ...")
    ss1 = fit_lasso(X1_train, y1_train, args.alpha_ss1, args.lasso_iter)
    n1  = sum(1 for c in ss1.coef if abs(c) > 1e-10)
    print(f"  Active terms: {n1}/{len(ss1.coef)}")

    print("Training SS2 (Heating Coil) Lasso ...")
    ss2 = fit_lasso(X2_train, y2_train, args.alpha_ss2, args.lasso_iter)
    n2  = sum(1 for c in ss2.coef if abs(c) > 1e-10)
    print(f"  Active terms: {n2}/{len(ss2.coef)}")

    # ── report ──
    print("\nBuilding report ...")
    report = build_report(train_s, val_s, ss1, ss2, args)
    report_path = out_dir / "ahu_v7_report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {report_path}")

    # ── HTML ──
    print("Generating HTML ...")
    plot_data = build_plot_data(val_s, ss1, ss2, args.freq_min)
    html_str  = render_html(plot_data, args)
    html_path = out_dir / "ahu_v7_validation.html"
    html_path.write_text(html_str, encoding="utf-8")
    print(f"Wrote {html_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
