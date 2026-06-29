"""
AHU v6 — SINDy-MIMO + Neural State-Space (NSS), with control signals.

Architecture (mirrors Habib et al. 2024, Fullpaper-V02):

  State vector  x = [T_rec, T_coil, T_sup]          (AHU internal states)
  Exogenous     d = [T_out, T_ret]                   (outdoor / return air, known)
  Control input u = [u_heat, u_heat_recovery, u_FF1] (actuator signals)

  Full feature vector at time t:  z(t) = [x(t), d(t), u(t)]  — 8 variables

  Model A — SINDy:
    Candidate library  Theta(z): degree-1 (8) + degree-2 (36) + time (4) = 48 features
    Lasso regression (L1) for each output delta:
        delta_i(t) = X_i(t+1) - X_i(t) = Theta(z(t)) . xi_i
    Prediction: X_i(t+1) = X_i(t) + delta_i_pred

  Model B — NSS (Neural State-Space):
    A 3-layer MLP that maps z(t) -> delta(t):
        hidden layers: [64, 64, 32] neurons, ReLU activation
        output:        3 neurons (one per state delta)
        trained with MSE loss + Adam-style gradient descent
    Prediction same as SINDy: X(t+1) = X(t) + MLP(z(t))

Multi-step rollout (both models):
  - d (T_out, T_ret) : real values from data at each future step
  - u (u_heat, u_heat_recovery, u_FF1): real values from data at each future step
  - x (T_rec, T_coil, T_sup): auto-regressed from model predictions

Evaluation horizons: H = 1, 4, 12 steps (5 min, 20 min, 60 min)
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

STATES = ["T_rec", "T_coil", "T_sup"]          # predicted outputs
EXOG   = ["T_out", "T_ret"]                    # exogenous disturbances
CTRL   = ["u_heat", "u_heat_recovery", "u_FF1"] # control inputs
ALL_Z  = STATES + EXOG + CTRL                  # full feature vector (8 vars)

HORIZONS = [1, 4, 12]

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AHU v6 SINDy + NSS with control signals")
    p.add_argument("--split-dir",   default="data_preprocessing/splits")
    p.add_argument("--out-dir",     default="model_outputs/v6")
    p.add_argument("--freq-min",    type=int,   default=5)
    p.add_argument("--alpha",       type=float, default=0.0001,
                   help="Lasso L1 regularisation strength.")
    p.add_argument("--lasso-iter",  type=int,   default=500)
    p.add_argument("--nss-hidden",  type=str,   default="64,64,32",
                   help="Comma-separated hidden layer sizes for NSS MLP.")
    p.add_argument("--nss-epochs",  type=int,   default=50)
    p.add_argument("--nss-lr",      type=float, default=0.001,
                   help="Adam learning rate for NSS.")
    p.add_argument("--nss-batch",   type=int,   default=512)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()

# ── data ──────────────────────────────────────────────────────────────────────

def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@dataclass
class Sample:
    timestamp: str
    z_now:  dict[str, float]   # full z(t) = [x, d, u]
    x_next: dict[str, float]   # true x(t+1)
    delta:  dict[str, float]   # x(t+1) - x(t) for STATES
    max_horizon: int = 0        # max consecutive steps available after this sample


def build_samples(rows: list[dict[str, str]], freq_min: int) -> list[Sample]:
    step = timedelta(minutes=freq_min)
    parsed = [datetime.fromisoformat(r["timestamp"]) for r in rows]

    # first pass: build a boolean array of whether step i->i+1 is consecutive
    consecutive = [
        parsed[i + 1] - parsed[i] == step
        for i in range(len(rows) - 1)
    ]

    # second pass: for each row i, compute how many consecutive steps follow
    # run_fwd[i] = number of consecutive steps starting at i (i.e. i->i+1, i+1->i+2, ...)
    max_h = len(rows)
    run_fwd = [0] * len(rows)
    for i in range(len(rows) - 2, -1, -1):
        if consecutive[i]:
            run_fwd[i] = 1 + run_fwd[i + 1]
        else:
            run_fwd[i] = 0

    samples: list[Sample] = []
    for i in range(len(rows) - 1):
        if not consecutive[i]:
            continue
        try:
            z_now  = {v: float(rows[i][v])     for v in ALL_Z}
            x_next = {v: float(rows[i + 1][v]) for v in STATES}
        except (KeyError, ValueError):
            continue
        delta = {v: x_next[v] - z_now[v] for v in STATES}
        # run_fwd[i] >= 1 because consecutive[i] is True; max_horizon = steps ahead we can roll
        samples.append(Sample(rows[i + 1]["timestamp"], z_now, x_next, delta, run_fwd[i]))
    return samples

# ── SINDy feature library ─────────────────────────────────────────────────────

def time_feats(ts: str) -> list[float]:
    dt = datetime.fromisoformat(ts)
    mday   = dt.hour * 60 + dt.minute
    d_ang  = 2 * math.pi * mday / 1440.0
    y_ang  = 2 * math.pi * dt.timetuple().tm_yday / 365.25
    return [math.sin(d_ang), math.cos(d_ang), math.sin(y_ang), math.cos(y_ang)]


def build_theta(z: dict[str, float], ts: str) -> list[float]:
    """
    Degree-1 (8) + degree-2 (36) + time (4) = 48 candidate features.
    Crucially, cross-terms between states and control signals (e.g. T_rec * u_heat)
    are included, capturing how actuators affect temperature dynamics.
    """
    vals = [z[v] for v in ALL_Z]
    feats: list[float] = list(vals)                     # degree-1: 8 terms
    for i in range(len(vals)):                           # degree-2: 36 terms
        for j in range(i, len(vals)):
            feats.append(vals[i] * vals[j])
    feats.extend(time_feats(ts))                        # time: 4 terms
    return feats


def sindy_feature_names() -> list[str]:
    names = list(ALL_Z)
    for i, vi in enumerate(ALL_Z):
        for vj in ALL_Z[i:]:
            names.append(f"{vi}*{vj}")
    names += ["sin_day", "cos_day", "sin_year", "cos_year"]
    return names

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
            rho    = sum(r[i] * col[i] for i in range(n))
            if col_norm2[j] < 1e-12:
                new_cj = 0.0
            else:
                new_cj = _soft(rho, lam) / col_norm2[j]
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

# ── NSS — Neural State-Space MLP ──────────────────────────────────────────────

@dataclass
class MLPModel:
    """Simple feed-forward MLP: input -> [hidden layers] -> output (3 deltas)."""
    weights: list[list[list[float]]]   # weights[layer][out][in]
    biases:  list[list[float]]         # biases[layer][out]
    means:   list[float]               # input standardisation
    stds:    list[float]
    n_outputs: int


def _relu(x: float) -> float:
    return x if x > 0 else 0.0


def mlp_forward(model: MLPModel, theta: list[float]) -> list[float]:
    """Forward pass: returns list of n_outputs delta predictions."""
    x = [(theta[j] - model.means[j]) / model.stds[j] for j in range(len(theta))]
    for W, b in zip(model.weights[:-1], model.biases[:-1]):
        x = [_relu(sum(W[o][i] * x[i] for i in range(len(x))) + b[o])
             for o in range(len(b))]
    W, b = model.weights[-1], model.biases[-1]
    x = [sum(W[o][i] * x[i] for i in range(len(x))) + b[o]
         for o in range(len(b))]
    return x


def _he_init(fan_in: int, fan_out: int, rng: random.Random) -> list[list[float]]:
    std = math.sqrt(2.0 / fan_in)
    return [[rng.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]


def train_nss(
    X: list[list[float]],          # theta features for each sample
    Y: list[list[float]],          # deltas [n_samples x 3]
    hidden_sizes: list[int],
    n_epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> MLPModel:
    """
    Train NSS MLP with mini-batch gradient descent (Adam optimiser).
    Pure Python — no external libraries.
    """
    rng = random.Random(seed)
    n, p = len(X), len(X[0])
    n_out = len(Y[0])

    # standardise inputs
    means = [sum(X[i][j] for i in range(n)) / n for j in range(p)]
    stds: list[float] = []
    for j in range(p):
        var = sum((X[i][j] - means[j]) ** 2 for i in range(n)) / max(1, n - 1)
        stds.append(math.sqrt(var) if var > 1e-12 else 1.0)
    Xs = [[(X[i][j] - means[j]) / stds[j] for j in range(p)] for i in range(n)]

    # standardise outputs (improves convergence)
    y_means = [sum(Y[i][k] for i in range(n)) / n for k in range(n_out)]
    y_stds: list[float] = []
    for k in range(n_out):
        var = sum((Y[i][k] - y_means[k]) ** 2 for i in range(n)) / max(1, n - 1)
        y_stds.append(math.sqrt(var) if var > 1e-12 else 1.0)
    Ys = [[(Y[i][k] - y_means[k]) / y_stds[k] for k in range(n_out)] for i in range(n)]

    # build layer sizes
    layer_sizes = [p] + hidden_sizes + [n_out]
    n_layers = len(layer_sizes) - 1

    # initialise weights and biases (He initialisation)
    W = [_he_init(layer_sizes[l], layer_sizes[l + 1], rng) for l in range(n_layers)]
    b = [[0.0] * layer_sizes[l + 1] for l in range(n_layers)]

    # Adam state
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    mW = [[[0.0]*len(W[l][o]) for o in range(len(W[l]))] for l in range(n_layers)]
    vW = [[[0.0]*len(W[l][o]) for o in range(len(W[l]))] for l in range(n_layers)]
    mb = [[0.0]*len(b[l]) for l in range(n_layers)]
    vb = [[0.0]*len(b[l]) for l in range(n_layers)]
    t_adam = 0

    idx = list(range(n))
    best_loss = float("inf")
    best_W = None
    best_b = None

    for epoch in range(n_epochs):
        rng.shuffle(idx)
        epoch_loss = 0.0
        n_batches = 0

        for batch_start in range(0, n, batch_size):
            batch = idx[batch_start: batch_start + batch_size]
            if not batch:
                continue
            t_adam += 1

            # ── forward pass (store activations) ──
            acts: list[list[list[float]]] = []  # acts[layer][sample]
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

            # ── loss (MSE) ──
            preds = acts[-1]
            targets = [Ys[i] for i in batch]
            B = len(batch)
            loss = sum(
                (preds[bi][k] - targets[bi][k]) ** 2
                for bi in range(B) for k in range(n_out)
            ) / (B * n_out)
            epoch_loss += loss
            n_batches += 1

            # ── backward pass ──
            # output layer delta
            deltas: list[list[list[float]]] = [[] for _ in range(n_layers)]
            deltas[-1] = [
                [2.0 * (preds[bi][k] - targets[bi][k]) / (B * n_out)
                 for k in range(n_out)]
                for bi in range(B)
            ]

            for l in range(n_layers - 2, -1, -1):
                prev_delta = deltas[l + 1]
                cur_delta: list[list[float]] = []
                for bi in range(B):
                    d = [
                        sum(W[l + 1][o][j] * prev_delta[bi][o]
                            for o in range(len(prev_delta[bi])))
                        * (1.0 if acts[l + 1][bi][j] > 0 else 0.0)
                        for j in range(len(acts[l + 1][bi]))
                    ]
                    cur_delta.append(d)
                deltas[l] = cur_delta

            # ── Adam weight update ──
            for l in range(n_layers):
                dW = [[
                    sum(deltas[l][bi][o] * acts[l][bi][j] for bi in range(B))
                    for j in range(len(W[l][o]))]
                    for o in range(len(W[l]))
                ]
                db = [sum(deltas[l][bi][o] for bi in range(B))
                      for o in range(len(b[l]))]

                for o in range(len(W[l])):
                    for j in range(len(W[l][o])):
                        mW[l][o][j] = beta1 * mW[l][o][j] + (1 - beta1) * dW[o][j]
                        vW[l][o][j] = beta2 * vW[l][o][j] + (1 - beta2) * dW[o][j] ** 2
                        m_hat = mW[l][o][j] / (1 - beta1 ** t_adam)
                        v_hat = vW[l][o][j] / (1 - beta2 ** t_adam)
                        W[l][o][j] -= lr * m_hat / (math.sqrt(v_hat) + eps)
                for o in range(len(b[l])):
                    mb[l][o] = beta1 * mb[l][o] + (1 - beta1) * db[o]
                    vb[l][o] = beta2 * vb[l][o] + (1 - beta2) * db[o] ** 2
                    m_hat = mb[l][o] / (1 - beta1 ** t_adam)
                    v_hat = vb[l][o] / (1 - beta2 ** t_adam)
                    b[l][o] -= lr * m_hat / (math.sqrt(v_hat) + eps)

        avg_loss = epoch_loss / max(1, n_batches)
        if avg_loss < best_loss:
            best_loss = avg_loss
            import copy
            best_W = copy.deepcopy(W)
            best_b = copy.deepcopy(b)

        if (epoch + 1) % 10 == 0:
            print(f"    NSS epoch {epoch+1:3d}/{n_epochs}  loss={avg_loss:.6f}", flush=True)

    # build model with best weights; un-standardise output layer
    # instead of storing y_means/y_stds separately, bake them into the
    # output layer weights and biases so mlp_forward returns raw delta values
    final_W = best_W if best_W is not None else W
    final_b = best_b if best_b is not None else b
    for o in range(n_out):
        scale = y_stds[o]
        shift = y_means[o]
        for j in range(len(final_W[-1][o])):
            final_W[-1][o][j] *= scale
        final_b[-1][o] = final_b[-1][o] * scale + shift

    return MLPModel(final_W, final_b, means, stds, n_out)

# ── rollout (shared by both models) ───────────────────────────────────────────

def rollout_sindy(
    models: dict[str, LassoModel],
    samples: list[Sample],
    horizon: int,
) -> tuple[list[dict[str, float]], list[int]]:
    """Returns (predictions, indices) — only for samples where H consecutive steps exist."""
    preds: list[dict[str, float]] = []
    indices: list[int] = []
    n_total = len(samples) - horizon
    for i in range(n_total):
        if i > 0 and i % 5000 == 0:
            print(f"    SINDy rollout {i}/{n_total} ({100*i//n_total}%)", flush=True)
        if samples[i].max_horizon < horizon:
            continue
        z = dict(samples[i].z_now)
        ts = samples[i].timestamp
        for h in range(horizon):
            theta = build_theta(z, ts)
            x_pred: dict[str, float] = {}
            for state in STATES:
                x_pred[state] = z[state] + predict_lasso_delta(models[state], theta)
            src = samples[i + h + 1].z_now if i + h + 1 < len(samples) else samples[i + h].x_next
            for v in EXOG + CTRL:
                x_pred[v] = src[v]
            for state in STATES:
                z[state] = x_pred[state]
            for v in EXOG + CTRL:
                z[v] = x_pred[v]
            ts = samples[i + h].timestamp
        preds.append({s: z[s] for s in STATES})
        indices.append(i)
    return preds, indices


def rollout_nss(
    model: MLPModel,
    samples: list[Sample],
    horizon: int,
) -> tuple[list[dict[str, float]], list[int]]:
    """Returns (predictions, indices) — only for samples where H consecutive steps exist."""
    preds: list[dict[str, float]] = []
    indices: list[int] = []
    n_total = len(samples) - horizon
    for i in range(n_total):
        if i > 0 and i % 5000 == 0:
            print(f"    NSS rollout {i}/{n_total} ({100*i//n_total}%)", flush=True)
        if samples[i].max_horizon < horizon:
            continue
        z = dict(samples[i].z_now)
        ts = samples[i].timestamp
        for h in range(horizon):
            theta = build_theta(z, ts)
            deltas = mlp_forward(model, theta)
            x_pred: dict[str, float] = {
                state: z[state] + deltas[k]
                for k, state in enumerate(STATES)
            }
            src = samples[i + h + 1].z_now if i + h + 1 < len(samples) else samples[i + h].x_next
            for v in EXOG + CTRL:
                x_pred[v] = src[v]
            for state in STATES:
                z[state] = x_pred[state]
            for v in EXOG + CTRL:
                z[v] = x_pred[v]
            ts = samples[i + h].timestamp
        preds.append({s: z[s] for s in STATES})
        indices.append(i)
    return preds, indices

# ── metrics ───────────────────────────────────────────────────────────────────

def metrics(true: list[float], pred: list[float]) -> dict[str, float]:
    n = len(true)
    errs = [p - t for t, p in zip(true, pred)]
    mae  = sum(abs(e) for e in errs) / n
    rmse = math.sqrt(sum(e * e for e in errs) / n)
    mean_y = sum(true) / n
    sst = sum((t - mean_y) ** 2 for t in true)
    sse = sum(e * e for e in errs)
    return {
        "n": n, "mae": mae, "rmse": rmse,
        "r2": 1.0 - sse / sst if sst > 0 else float("nan"),
        "bias": sum(errs) / n,
    }


def true_at(samples: list[Sample], H: int, state: str, indices: list[int]) -> list[float]:
    return [samples[i + H].z_now[state] for i in indices]


def persistence_at(samples: list[Sample], H: int, state: str, indices: list[int]) -> list[float]:
    return [samples[i].z_now[state] for i in indices]

# ── report ────────────────────────────────────────────────────────────────────

def build_report(
    train_s: list[Sample], val_s: list[Sample],
    sindy_models: dict[str, LassoModel],
    nss_model: MLPModel,
    args: argparse.Namespace,
    precomputed: dict,          # {split: {H: (sindy_preds, sindy_idx, nss_preds, nss_idx)}}
) -> list[str]:
    feat_names = sindy_feature_names()
    lines = [
        "AHU v6 SINDy-MIMO + NSS report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Frequency: {args.freq_min} min",
        f"States (predicted): {STATES}",
        f"Exogenous (real in rollout): {EXOG}",
        f"Control inputs (real in rollout): {CTRL}",
        f"SINDy feature library: {len(feat_names)} candidates",
        f"SINDy Lasso alpha: {args.alpha}",
        f"NSS architecture: {len(ALL_Z)*2+1} -> {args.nss_hidden} -> 3",
        f"NSS epochs: {args.nss_epochs}  lr: {args.nss_lr}  batch: {args.nss_batch}",
        "",
        "=== SINDy active coefficients ===",
    ]
    for state in STATES:
        m = sindy_models[state]
        active = [(feat_names[j], m.coef[j])
                  for j in range(len(m.coef)) if abs(m.coef[j]) > 1e-10]
        lines.append(f"\n  delta_{state}  intercept={m.intercept:+.6f}"
                     f"  active={len(active)}/{len(m.coef)}")
        for name, c in sorted(active, key=lambda x: -abs(x[1]))[:15]:
            lines.append(f"    {name:30s}  {c:+.8f}")
        if len(active) > 15:
            lines.append(f"    ... ({len(active) - 15} more)")

    lines += ["", "=== Prediction metrics (gap-aware rollout) ==="]
    for split_name, samples in [("train", train_s), ("validation", val_s)]:
        lines.append(f"\n--- {split_name} ({len(samples)} samples) ---")
        for H in HORIZONS:
            if H >= len(samples):
                continue
            sindy_p, idx_s, nss_p, idx_n = precomputed[split_name][H]
            tv_all   = {s: true_at(samples, H, s, idx_s) for s in STATES}
            pers_all = {s: persistence_at(samples, H, s, idx_s) for s in STATES}
            tv_nss   = {s: true_at(samples, H, s, idx_n) for s in STATES}
            lines.append(f"  H={H:2d} ({H * args.freq_min} min ahead, {len(idx_s)} valid windows)")
            for state in STATES:
                tv      = tv_all[state]
                m_pers  = metrics(tv, pers_all[state])
                m_sindy = metrics(tv, [p[state] for p in sindy_p])
                m_nss   = metrics(tv_nss[state], [p[state] for p in nss_p])
                lines.append(
                    f"    {state:8s} "
                    f"Persistence MAE={m_pers['mae']:.4f} RMSE={m_pers['rmse']:.4f} | "
                    f"SINDy MAE={m_sindy['mae']:.4f} RMSE={m_sindy['rmse']:.4f} R2={m_sindy['r2']:.4f} | "
                    f"NSS   MAE={m_nss['mae']:.4f} RMSE={m_nss['rmse']:.4f} R2={m_nss['r2']:.4f}"
                )
    return lines

# ── HTML ──────────────────────────────────────────────────────────────────────

def build_plot_data(
    val_s: list[Sample],
    freq_min: int,
    precomputed_val: dict,      # {H: (sindy_preds, sindy_idx, nss_preds, nss_idx)}
) -> dict:
    # common valid indices across all horizons
    common_idx: set[int] | None = None
    for H in HORIZONS:
        _, idx_s, _, idx_n = precomputed_val[H]
        h_set = set(idx_s) & set(idx_n)
        common_idx = h_set if common_idx is None else common_idx & h_set
    base_idx = sorted(common_idx or [])

    timestamps = [val_s[i].timestamp for i in base_idx]

    series: dict = {}
    for state in STATES:
        sd: dict[str, list[float]] = {}
        for H in HORIZONS:
            preds_s, idx_s, preds_n, idx_n = precomputed_val[H]
            s_map = {i: k for k, i in enumerate(idx_s)}
            n_map = {i: k for k, i in enumerate(idx_n)}
            sd[f"true_h{H}"]        = [val_s[i + H].z_now[state]      for i in base_idx]
            sd[f"persistence_h{H}"] = [val_s[i].z_now[state]          for i in base_idx]
            sd[f"sindy_h{H}"]       = [preds_s[s_map[i]][state]       for i in base_idx]
            sd[f"nss_h{H}"]         = [preds_n[n_map[i]][state]       for i in base_idx]
        series[state] = sd
    return {"timestamps": timestamps, "states": series,
            "freqMin": freq_min, "horizons": HORIZONS}


def render_html(plot_data: dict, alpha: float, nss_hidden: str) -> str:
    data_json = json.dumps(plot_data, separators=(",", ":")).replace("</", "<\\/")
    horizons  = plot_data["horizons"]
    freq_min  = plot_data["freqMin"]

    colors = {
        "true":        "#111827",
        "persistence": "#9ca3af",
        "sindy":       "#2563eb",
        "nss":         "#dc2626",
    }
    labels = {
        "true": "True", "persistence": "Persistence",
        "sindy": "SINDy", "nss": "NSS",
    }

    # metric table
    metric_rows = []
    for H in horizons:
        for state in STATES:
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

    # panels: one per (state, horizon)
    sections = []
    for state in STATES:
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
        "const MKEYS=['persistence','sindy','nss'];\n"
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
        "<title>AHU v6 SINDy+NSS Validation</title>"
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
        f"<h1>AHU v6 SINDy + NSS — Validation</h1>"
        f"<p class='meta'>"
        f"States: T_rec, T_coil, T_sup &nbsp;|&nbsp; "
        f"Control inputs: u_heat, u_heat_recovery, u_FF1 &nbsp;|&nbsp; "
        f"Lasso α={alpha} &nbsp;|&nbsp; NSS hidden=[{nss_hidden}] &nbsp;|&nbsp; "
        f"Time: {first} to {last} &nbsp;|&nbsp; Samples: {len(plot_data['timestamps'])}"
        f"</p>"
        f"<div class='legend'>{legend_html}</div>"
        "<table><thead><tr>"
        "<th>State</th><th>Model</th><th>MAE</th><th>RMSE</th><th>Bias</th>"
        "</tr></thead><tbody>"
        + "".join(metric_rows) +
        "</tbody></table>"
        + "".join(sections) +
        "</main><script>" + js + "</script></body></html>"
    )

# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
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

    # ── build feature matrices ──
    print("Building feature matrices ...")
    X_train = [build_theta(s.z_now, s.timestamp) for s in train_s]
    n_feat  = len(X_train[0])
    print(f"  Feature dimension: {n_feat}")

    # ── SINDy ──
    print("\nTraining SINDy (Lasso) models ...")
    sindy_models: dict[str, LassoModel] = {}
    for state in STATES:
        y = [s.delta[state] for s in train_s]
        print(f"  delta_{state} ...")
        sindy_models[state] = fit_lasso(X_train, y, args.alpha, args.lasso_iter)
        n_active = sum(1 for c in sindy_models[state].coef if abs(c) > 1e-10)
        print(f"    active terms: {n_active}/{n_feat}")

    # ── NSS ──
    print("\nTraining NSS (MLP) ...")
    hidden_sizes = [int(h) for h in args.nss_hidden.split(",")]
    Y_train = [[s.delta[st] for st in STATES] for s in train_s]
    nss_model = train_nss(
        X_train, Y_train, hidden_sizes,
        args.nss_epochs, args.nss_lr, args.nss_batch, args.seed,
    )
    print(f"  NSS training complete.")

    # ── precompute all rollouts once ──
    print("\nRunning rollouts ...")
    precomputed: dict = {"train": {}, "validation": {}}
    for split_name, samples in [("train", train_s), ("validation", val_s)]:
        for H in HORIZONS:
            print(f"\n  [{split_name}] H={H} — SINDy rollout ...", flush=True)
            sp, si = rollout_sindy(sindy_models, samples, H)
            print(f"  [{split_name}] H={H} — NSS rollout ({len(si)} windows) ...", flush=True)
            np_, ni = rollout_nss(nss_model, samples, H)
            print(f"  [{split_name}] H={H} done. valid={len(ni)}", flush=True)
            precomputed[split_name][H] = (sp, si, np_, ni)

    # ── report ──
    print("\nBuilding report ...")
    report = build_report(train_s, val_s, sindy_models, nss_model, args, precomputed)
    report_path = out_dir / "ahu_v6_report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {report_path}")

    # ── HTML ──
    print("Generating HTML ...")
    plot_data = build_plot_data(val_s, args.freq_min, precomputed["validation"])
    html_str  = render_html(plot_data, args.alpha, args.nss_hidden)
    html_path = out_dir / "ahu_v6_validation.html"
    html_path.write_text(html_str, encoding="utf-8")
    print(f"Wrote {html_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
