"""
AHU v7 — Physics-structured SINDy + NSS with two subsystems.

Extends v6 (SINDy-MIMO + NSS) by splitting into two physically meaningful
subsystems instead of one monolithic model:

Subsystem 1 — Heat Exchanger (produces T_rec):
  Variables: [T_rec, T_out, T_ret, u_heat_recovery]  — 4 vars → 18 features
  Models: SINDy (Lasso) + NSS (MLP)
  Target: delta_T_rec

Subsystem 2 — Heating Coil (produces T_sup):
  Variables: [T_sup, T_rec, T_coil, u_heat]  — 4 vars → 18 features
  Models: SINDy (Lasso) + NSS (MLP)
  Target: delta_T_sup
  Note: T_coil always real (sensor, not predicted); T_rec from SS1 in rollout

Multi-step rollout (both SINDy and NSS):
  - T_out, T_ret, T_coil: real values at each future step
  - u_heat_recovery, u_heat: real values at each future step
  - T_rec: auto-regressed from SS1 predictions
  - T_sup: auto-regressed from SS2 predictions

Evaluation horizons: H = 1, 4, 12 steps (5 min, 20 min, 60 min)
Gap handling: only rollout windows where all H steps are consecutive.
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

SS1_VARS = ["T_rec", "T_out", "T_ret", "u_heat_recovery"]
SS2_VARS = ["T_sup", "T_rec", "T_coil", "u_heat"]
ALL_COLS = ["T_rec", "T_sup", "T_out", "T_ret", "T_coil", "u_heat_recovery", "u_heat"]

HORIZONS = [1, 4, 12]

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AHU v7 physics-structured SINDy+NSS")
    p.add_argument("--split-dir",   default="data_preprocessing/splits")
    p.add_argument("--out-dir",     default="model_outputs/v7")
    p.add_argument("--freq-min",    type=int,   default=5)
    p.add_argument("--alpha-ss1",   type=float, default=0.0001)
    p.add_argument("--alpha-ss2",   type=float, default=0.0001)
    p.add_argument("--lasso-iter",  type=int,   default=500)
    p.add_argument("--nss-hidden",  type=str,   default="32,32,16",
                   help="Hidden layer sizes for each subsystem NSS MLP.")
    p.add_argument("--nss-epochs",  type=int,   default=50)
    p.add_argument("--nss-lr",      type=float, default=0.001)
    p.add_argument("--nss-batch",   type=int,   default=512)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()

# ── data ──────────────────────────────────────────────────────────────────────

def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@dataclass
class Sample:
    timestamp:   str
    vals_now:    dict[str, float]
    vals_next:   dict[str, float]
    delta_rec:   float
    delta_sup:   float
    max_horizon: int


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
            now = {c: float(rows[i][c])     for c in ALL_COLS}
            nxt = {c: float(rows[i + 1][c]) for c in ALL_COLS}
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

# ── feature library ───────────────────────────────────────────────────────────

def time_feats(ts: str) -> list[float]:
    dt    = datetime.fromisoformat(ts)
    mday  = dt.hour * 60 + dt.minute
    d_ang = 2 * math.pi * mday / 1440.0
    y_ang = 2 * math.pi * dt.timetuple().tm_yday / 365.25
    return [math.sin(d_ang), math.cos(d_ang), math.sin(y_ang), math.cos(y_ang)]


def build_theta(var_vals: list[float], ts: str) -> list[float]:
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

# ── Lasso ─────────────────────────────────────────────────────────────────────

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
    r = [yi - intercept for yi in y]
    lam = alpha * n

    for _ in range(max_iter):
        max_change = 0.0
        for j in range(p):
            cj = coef[j]
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


def predict_lasso_delta(model: LassoModel, theta: list[float]) -> float:
    xs = [(theta[j] - model.means[j]) / model.stds[j] for j in range(len(theta))]
    return model.intercept + sum(c * v for c, v in zip(model.coef, xs))

# ── NSS — MLP ─────────────────────────────────────────────────────────────────

@dataclass
class MLPModel:
    weights:   list[list[list[float]]]
    biases:    list[list[float]]
    means:     list[float]
    stds:      list[float]
    n_outputs: int


def _relu(x: float) -> float:
    return x if x > 0 else 0.0


def mlp_forward(model: MLPModel, theta: list[float]) -> list[float]:
    x = [(theta[j] - model.means[j]) / model.stds[j] for j in range(len(theta))]
    for W, b in zip(model.weights[:-1], model.biases[:-1]):
        x = [_relu(sum(W[o][i] * x[i] for i in range(len(x))) + b[o])
             for o in range(len(b))]
    W, b = model.weights[-1], model.biases[-1]
    return [sum(W[o][i] * x[i] for i in range(len(x))) + b[o] for o in range(len(b))]


def _he_init(fan_in: int, fan_out: int, rng: random.Random) -> list[list[float]]:
    std = math.sqrt(2.0 / fan_in)
    return [[rng.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]


def train_nss(
    X: list[list[float]], Y: list[list[float]],
    hidden_sizes: list[int], n_epochs: int,
    lr: float, batch_size: int, seed: int,
    label: str = "",
) -> MLPModel:
    rng  = random.Random(seed)
    n, p = len(X), len(X[0])
    n_out = len(Y[0])

    means = [sum(X[i][j] for i in range(n)) / n for j in range(p)]
    stds: list[float] = []
    for j in range(p):
        var = sum((X[i][j] - means[j]) ** 2 for i in range(n)) / max(1, n - 1)
        stds.append(math.sqrt(var) if var > 1e-12 else 1.0)
    Xs = [[(X[i][j] - means[j]) / stds[j] for j in range(p)] for i in range(n)]

    y_means = [sum(Y[i][k] for i in range(n)) / n for k in range(n_out)]
    y_stds: list[float] = []
    for k in range(n_out):
        var = sum((Y[i][k] - y_means[k]) ** 2 for i in range(n)) / max(1, n - 1)
        y_stds.append(math.sqrt(var) if var > 1e-12 else 1.0)
    Ys = [[(Y[i][k] - y_means[k]) / y_stds[k] for k in range(n_out)] for i in range(n)]

    layer_sizes = [p] + hidden_sizes + [n_out]
    n_layers    = len(layer_sizes) - 1
    W  = [_he_init(layer_sizes[l], layer_sizes[l + 1], rng) for l in range(n_layers)]
    b  = [[0.0] * layer_sizes[l + 1] for l in range(n_layers)]

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    mW = [[[0.0]*len(W[l][o]) for o in range(len(W[l]))] for l in range(n_layers)]
    vW = [[[0.0]*len(W[l][o]) for o in range(len(W[l]))] for l in range(n_layers)]
    mb = [[0.0]*len(b[l]) for l in range(n_layers)]
    vb = [[0.0]*len(b[l]) for l in range(n_layers)]
    t_adam = 0

    idx_list  = list(range(n))
    best_loss = float("inf")
    best_W: list | None = None
    best_b: list | None = None

    for epoch in range(n_epochs):
        rng.shuffle(idx_list)
        epoch_loss = 0.0
        n_batches  = 0

        for batch_start in range(0, n, batch_size):
            batch = idx_list[batch_start: batch_start + batch_size]
            if not batch:
                continue
            t_adam += 1
            B = len(batch)

            acts: list[list[list[float]]] = []
            cur = [Xs[i] for i in batch]
            acts.append(cur)
            for l in range(n_layers):
                nxt: list[list[float]] = []
                for xi in cur:
                    h = [sum(W[l][o][j] * xi[j] for j in range(len(xi))) + b[l][o]
                         for o in range(len(b[l]))]
                    if l < n_layers - 1:
                        h = [_relu(v) for v in h]
                    nxt.append(h)
                acts.append(nxt)
                cur = nxt

            preds_b  = acts[-1]
            targets_b = [Ys[i] for i in batch]
            loss = sum(
                (preds_b[bi][k] - targets_b[bi][k]) ** 2
                for bi in range(B) for k in range(n_out)
            ) / (B * n_out)
            epoch_loss += loss
            n_batches  += 1

            layer_deltas: list[list[list[float]]] = [[] for _ in range(n_layers)]
            layer_deltas[-1] = [
                [2.0 * (preds_b[bi][k] - targets_b[bi][k]) / (B * n_out)
                 for k in range(n_out)]
                for bi in range(B)
            ]
            for l in range(n_layers - 2, -1, -1):
                pd = layer_deltas[l + 1]
                cd: list[list[float]] = []
                for bi in range(B):
                    d = [
                        sum(W[l+1][o][j] * pd[bi][o] for o in range(len(pd[bi])))
                        * (1.0 if acts[l+1][bi][j] > 0 else 0.0)
                        for j in range(len(acts[l+1][bi]))
                    ]
                    cd.append(d)
                layer_deltas[l] = cd

            for l in range(n_layers):
                dW = [[sum(layer_deltas[l][bi][o] * acts[l][bi][j] for bi in range(B))
                       for j in range(len(W[l][o]))]
                      for o in range(len(W[l]))]
                db_l = [sum(layer_deltas[l][bi][o] for bi in range(B))
                        for o in range(len(b[l]))]
                for o in range(len(W[l])):
                    for j in range(len(W[l][o])):
                        mW[l][o][j] = beta1*mW[l][o][j] + (1-beta1)*dW[o][j]
                        vW[l][o][j] = beta2*vW[l][o][j] + (1-beta2)*dW[o][j]**2
                        mh = mW[l][o][j] / (1 - beta1**t_adam)
                        vh = vW[l][o][j] / (1 - beta2**t_adam)
                        W[l][o][j] -= lr * mh / (math.sqrt(vh) + eps)
                for o in range(len(b[l])):
                    mb[l][o] = beta1*mb[l][o] + (1-beta1)*db_l[o]
                    vb[l][o] = beta2*vb[l][o] + (1-beta2)*db_l[o]**2
                    mh = mb[l][o] / (1 - beta1**t_adam)
                    vh = vb[l][o] / (1 - beta2**t_adam)
                    b[l][o] -= lr * mh / (math.sqrt(vh) + eps)

        avg_loss = epoch_loss / max(1, n_batches)
        if avg_loss < best_loss:
            best_loss = avg_loss
            import copy
            best_W = copy.deepcopy(W)
            best_b = copy.deepcopy(b)
        if (epoch + 1) % 10 == 0:
            print(f"    {label} epoch {epoch+1:3d}/{n_epochs}  loss={avg_loss:.6f}", flush=True)

    final_W = best_W if best_W is not None else W
    final_b = best_b if best_b is not None else b
    for o in range(n_out):
        scale = y_stds[o]
        shift = y_means[o]
        for j in range(len(final_W[-1][o])):
            final_W[-1][o][j] *= scale
        final_b[-1][o] = final_b[-1][o] * scale + shift

    return MLPModel(final_W, final_b, means, stds, n_out)

# ── rollout ───────────────────────────────────────────────────────────────────

@dataclass
class RolloutPred:
    sindy_rec: float
    sindy_sup: float
    nss_rec:   float
    nss_sup:   float


def rollout(
    lasso1: LassoModel, lasso2: LassoModel,
    nss1:   MLPModel,   nss2:   MLPModel,
    samples: list[Sample],
    horizon: int,
) -> tuple[list[RolloutPred], list[int]]:
    preds:   list[RolloutPred] = []
    indices: list[int]         = []
    n_total = len(samples) - horizon

    for i in range(n_total):
        if i > 0 and i % 5000 == 0:
            print(f"    rollout {i}/{n_total} ({100*i//n_total}%)", flush=True)
        if samples[i].max_horizon < horizon:
            continue

        # separate state trackers for SINDy and NSS
        s_rec = samples[i].vals_now["T_rec"]
        s_sup = samples[i].vals_now["T_sup"]
        n_rec = samples[i].vals_now["T_rec"]
        n_sup = samples[i].vals_now["T_sup"]
        ts    = samples[i].timestamp

        for h in range(horizon):
            src    = samples[i + h]
            t_out  = src.vals_now["T_out"]
            t_ret  = src.vals_now["T_ret"]
            t_coil = src.vals_now["T_coil"]
            u_hr   = src.vals_now["u_heat_recovery"]
            u_heat = src.vals_now["u_heat"]

            # ── SINDy path ──
            th1_s  = build_theta([s_rec, t_out, t_ret, u_hr], ts)
            s_rec  = s_rec + predict_lasso_delta(lasso1, th1_s)
            th2_s  = build_theta([s_sup, s_rec, t_coil, u_heat], ts)
            s_sup  = s_sup + predict_lasso_delta(lasso2, th2_s)

            # ── NSS path ──
            th1_n   = build_theta([n_rec, t_out, t_ret, u_hr], ts)
            n_rec   = n_rec + mlp_forward(nss1, th1_n)[0]
            th2_n   = build_theta([n_sup, n_rec, t_coil, u_heat], ts)
            n_sup   = n_sup + mlp_forward(nss2, th2_n)[0]

            ts = src.timestamp

        preds.append(RolloutPred(s_rec, s_sup, n_rec, n_sup))
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
    train_s: list[Sample], val_s: list[Sample],
    lasso1: LassoModel, lasso2: LassoModel,
    nss1: MLPModel, nss2: MLPModel,
    args: argparse.Namespace,
    precomputed: dict,      # {"train": {H: (preds, idx)}, "validation": {H: (preds, idx)}}
) -> list[str]:
    names1 = theta_names(SS1_VARS)
    names2 = theta_names(SS2_VARS)

    lines = [
        "AHU v7 Physics-structured SINDy + NSS report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Frequency: {args.freq_min} min",
        f"NSS hidden: {args.nss_hidden}  epochs: {args.nss_epochs}  lr: {args.nss_lr}",
        "",
        "=== Subsystem 1: Heat Exchanger ===",
        f"  Variables: {SS1_VARS}",
        f"  Feature library: {len(names1)} candidates (4 vars: 4+10+4)",
        f"  Lasso alpha: {args.alpha_ss1}",
    ]
    active1 = [(names1[j], lasso1.coef[j]) for j in range(len(lasso1.coef)) if abs(lasso1.coef[j]) > 1e-10]
    lines.append(f"  SINDy active: {len(active1)}/{len(lasso1.coef)}  intercept={lasso1.intercept:+.6f}")
    for name, c in sorted(active1, key=lambda x: -abs(x[1])):
        lines.append(f"    {name:35s}  {c:+.8f}")

    lines += [
        "",
        "=== Subsystem 2: Heating Coil ===",
        f"  Variables: {SS2_VARS}",
        f"  Feature library: {len(names2)} candidates (4 vars: 4+10+4)",
        f"  Lasso alpha: {args.alpha_ss2}",
        f"  Note: T_coil always real; T_rec from SS1 in rollout",
    ]
    active2 = [(names2[j], lasso2.coef[j]) for j in range(len(lasso2.coef)) if abs(lasso2.coef[j]) > 1e-10]
    lines.append(f"  SINDy active: {len(active2)}/{len(lasso2.coef)}  intercept={lasso2.intercept:+.6f}")
    for name, c in sorted(active2, key=lambda x: -abs(x[1])):
        lines.append(f"    {name:35s}  {c:+.8f}")

    lines += ["", "=== Prediction metrics (gap-aware rollout) ==="]

    for split_name, samples in [("train", train_s), ("validation", val_s)]:
        lines.append(f"\n--- {split_name} ({len(samples)} samples) ---")
        for H in HORIZONS:
            if H >= len(samples):
                continue
            preds, idx = precomputed[split_name][H]
            lines.append(f"  H={H:2d} ({H * args.freq_min} min ahead, {len(idx)} valid windows)")
            for state, true_key, s_attr, n_attr in [
                ("T_rec", "T_rec", "sindy_rec", "nss_rec"),
                ("T_sup", "T_sup", "sindy_sup", "nss_sup"),
            ]:
                tv    = [samples[i + H].vals_now[true_key] for i in idx]
                persv = [samples[i].vals_now[true_key]     for i in idx]
                sv    = [getattr(p, s_attr) for p in preds]
                nv    = [getattr(p, n_attr) for p in preds]
                mp    = metrics(tv, persv)
                ms    = metrics(tv, sv)
                mn    = metrics(tv, nv)
                lines.append(
                    f"    {state:8s} "
                    f"Persistence MAE={mp['mae']:.4f} RMSE={mp['rmse']:.4f} | "
                    f"SINDy MAE={ms['mae']:.4f} RMSE={ms['rmse']:.4f} R2={ms['r2']:.4f} | "
                    f"NSS   MAE={mn['mae']:.4f} RMSE={mn['rmse']:.4f} R2={mn['r2']:.4f}"
                )
    return lines

# ── HTML ──────────────────────────────────────────────────────────────────────

def build_plot_data(
    val_s: list[Sample],
    freq_min: int,
    precomputed_val: dict,      # {H: (preds, idx)}
) -> dict:
    common_idx: set[int] | None = None
    for H in HORIZONS:
        _, idx = precomputed_val[H]
        common_idx = set(idx) if common_idx is None else common_idx & set(idx)
    base_idx = sorted(common_idx or [])
    timestamps = [val_s[i].timestamp for i in base_idx]

    series: dict = {}
    for state, true_key, s_attr, n_attr in [
        ("T_rec", "T_rec", "sindy_rec", "nss_rec"),
        ("T_sup", "T_sup", "sindy_sup", "nss_sup"),
    ]:
        sd: dict[str, list[float]] = {}
        for H in HORIZONS:
            preds, idx = precomputed_val[H]
            idx_map = {i: k for k, i in enumerate(idx)}
            sd[f"true_h{H}"]        = [val_s[i + H].vals_now[true_key]          for i in base_idx]
            sd[f"persistence_h{H}"] = [val_s[i].vals_now[true_key]              for i in base_idx]
            sd[f"sindy_h{H}"]       = [getattr(preds[idx_map[i]], s_attr)       for i in base_idx]
            sd[f"nss_h{H}"]         = [getattr(preds[idx_map[i]], n_attr)       for i in base_idx]
        series[state] = sd

    return {"timestamps": timestamps, "states": series,
            "freqMin": freq_min, "horizons": HORIZONS}


def render_html(plot_data: dict, args: argparse.Namespace) -> str:
    data_json = json.dumps(plot_data, separators=(",", ":")).replace("</", "<\\/")
    horizons  = plot_data["horizons"]
    freq_min  = plot_data["freqMin"]
    states    = list(plot_data["states"].keys())

    colors = {"true": "#111827", "persistence": "#9ca3af",
              "sindy": "#2563eb", "nss": "#dc2626"}
    labels = {"true": "True", "persistence": "Persistence",
              "sindy": "SINDy", "nss": "NSS"}

    metric_rows = []
    for H in horizons:
        for state in states:
            sd = plot_data["states"][state]
            tv = sd[f"true_h{H}"]
            for mk in ["persistence", "sindy", "nss"]:
                pv   = sd[f"{mk}_h{H}"]
                errs = [v - t for t, v in zip(tv, pv)]
                mae  = sum(abs(e) for e in errs) / len(errs)
                rmse = math.sqrt(sum(e*e for e in errs) / len(errs))
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
        for k in ["true", "persistence", "sindy", "nss"]
    )

    first = html_mod.escape(plot_data["timestamps"][0])
    last  = html_mod.escape(plot_data["timestamps"][-1])
    colors_json = json.dumps(colors)

    js = (
        "const D=" + data_json + ";\n"
        "const COLORS=" + colors_json + ";\n"
        "const HORIZONS=" + json.dumps(horizons) + ";\n"
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
        "    nssArr:sd['nss_h'+H],\n"
        "    start:0,end:n-1,hoverIndex:null,dragging:false,\n"
        "    dragStartX:0,dragStartStart:0,dragStartEnd:n-1,cssWidth:0,cssHeight:0};}\n"
        "function chartArea(c){return{left:66,top:22,width:c.cssWidth-90,height:c.cssHeight-54};}\n"
        "function visBounds(c){\n"
        "  const l=Math.max(0,Math.floor(c.start)),r=Math.min(D.timestamps.length-1,Math.ceil(c.end));\n"
        "  let mn=Infinity,mx=-Infinity;\n"
        "  for(const arr of[c.trueArr,c.persArr,c.sindyArr,c.nssArr]){\n"
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
        "  drawLine(ctx,c,c.nssArr,a,b,COLORS.nss,1.5);\n"
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
        "      +' | SINDy='+c.sindyArr[i].toFixed(3)\n"
        "      +' | NSS='+c.nssArr[i].toFixed(3);}\n"
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
        "    if(ro)ro.textContent='';drawChart(c);});\n"
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
        "<title>AHU v7 Physics-structured SINDy+NSS Validation</title>"
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
        "<h1>AHU v7 — Physics-structured SINDy + NSS Validation</h1>"
        f"<p class='meta'>"
        f"SS1 (Heat Exchanger): T_rec ← [T_out, T_ret, u_heat_recovery] &nbsp;|&nbsp; "
        f"SS2 (Heating Coil): T_sup ← [T_rec, T_coil, u_heat] &nbsp;|&nbsp; "
        f"NSS hidden=[{args.nss_hidden}] epochs={args.nss_epochs} &nbsp;|&nbsp; "
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
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    train_rows = load_rows(Path(args.split_dir) / "train_5min.csv")
    val_rows   = load_rows(Path(args.split_dir) / "validation_5min.csv")

    print("Building samples ...")
    train_s = build_samples(train_rows, args.freq_min)
    val_s   = build_samples(val_rows,   args.freq_min)
    print(f"  Train: {len(train_s)}  Val: {len(val_s)}")

    print("Building feature matrices ...")
    X1 = [theta_ss1(s) for s in train_s]
    X2 = [theta_ss2(s) for s in train_s]
    y1 = [s.delta_rec for s in train_s]
    y2 = [s.delta_sup for s in train_s]
    print(f"  SS1 features: {len(X1[0])}  SS2 features: {len(X2[0])}")

    print("\nTraining SS1 Lasso ...")
    lasso1 = fit_lasso(X1, y1, args.alpha_ss1, args.lasso_iter)
    print(f"  Active: {sum(1 for c in lasso1.coef if abs(c)>1e-10)}/{len(lasso1.coef)}")

    print("Training SS2 Lasso ...")
    lasso2 = fit_lasso(X2, y2, args.alpha_ss2, args.lasso_iter)
    print(f"  Active: {sum(1 for c in lasso2.coef if abs(c)>1e-10)}/{len(lasso2.coef)}")

    hidden_sizes = [int(h) for h in args.nss_hidden.split(",")]

    print(f"\nTraining SS1 NSS ({args.nss_hidden}, {args.nss_epochs} epochs) ...")
    nss1 = train_nss(X1, [[v] for v in y1], hidden_sizes,
                     args.nss_epochs, args.nss_lr, args.nss_batch, args.seed, "SS1")

    print(f"\nTraining SS2 NSS ({args.nss_hidden}, {args.nss_epochs} epochs) ...")
    nss2 = train_nss(X2, [[v] for v in y2], hidden_sizes,
                     args.nss_epochs, args.nss_lr, args.nss_batch, args.seed + 1, "SS2")

    # ── precompute all rollouts once ──
    print("\nRunning rollouts ...")
    precomputed: dict = {"train": {}, "validation": {}}
    for split_name, samples in [("train", train_s), ("validation", val_s)]:
        for H in HORIZONS:
            print(f"\n  [{split_name}] H={H} — rollout (SINDy+NSS) ...", flush=True)
            result = rollout(lasso1, lasso2, nss1, nss2, samples, H)
            print(f"  [{split_name}] H={H} done. valid={len(result[1])}", flush=True)
            precomputed[split_name][H] = result

    print("\nBuilding report ...")
    report = build_report(train_s, val_s, lasso1, lasso2, nss1, nss2, args, precomputed)
    report_path = out_dir / "ahu_v7_report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {report_path}")

    print("Generating HTML ...")
    plot_data = build_plot_data(val_s, args.freq_min, precomputed["validation"])
    html_str  = render_html(plot_data, args)
    html_path = out_dir / "ahu_v7_validation.html"
    html_path.write_text(html_str, encoding="utf-8")
    print(f"Wrote {html_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
