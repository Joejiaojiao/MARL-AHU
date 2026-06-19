"""
AHU two-stage v4 piecewise-weighted delta model.

Builds on v3 but replaces the linear weight function with a two-segment
piecewise weight function:

    if |delta| < threshold:
        w = w_min                        (nearly-stable steps, very low weight)
    else:
        w = 1 + k * |delta| ** power     (changing steps, power-law boost)

Defaults:
    threshold = 0.05   (below T_rec p50; covers ~54% of T_rec samples)
    w_min     = 0.1    (keep but heavily discount near-zero-delta samples)
    power     = 2.0    (quadratic: gentler than linear for mid-delta, steeper at large)
    k         = 3.0

Subsystem inputs (same as v2/v3):
    heat_exchanger: T_out, T_ret, T_rec, pressure_hex  over [t-window ... t]
    heating_coil:   T_rec, T_coil, T_sup               over [t-window ... t]

Target: delta = X(t+1) - X(t), reconstructed as X_pred(t+1) = X(t) + delta_pred.
Gap boundaries are skipped automatically (consecutive-step check).
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


SUBSYSTEMS = {
    "heat_exchanger": {
        "target": "T_rec",
        "inputs": ["T_out", "T_ret", "T_rec", "pressure_hex"],
    },
    "heating_coil": {
        "target": "T_sup",
        "inputs": ["T_rec", "T_coil", "T_sup"],
    },
}


@dataclass
class SequenceDataset:
    timestamps: list[str]
    features: dict[str, list[list[float]]]
    current_values: dict[str, list[float]]
    targets: dict[str, list[float]]
    deltas: dict[str, list[float]]
    feature_names: dict[str, list[str]]


@dataclass
class RidgeModel:
    weights: list[float]
    means: list[float]
    stds: list[float]


@dataclass
class TreeNode:
    value: float
    feature_idx: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


@dataclass
class TreeEnsemble:
    trees: list[TreeNode]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AHU v4 piecewise-weighted delta models.")
    parser.add_argument("--split-dir", default="data_preprocessing/splits")
    parser.add_argument("--out-dir", default="model_outputs/v4")
    parser.add_argument("--freq-min", type=int, default=5)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="|delta| below this -> weight = w_min.")
    parser.add_argument("--w-min", type=float, default=0.1,
                        help="Weight for near-stable samples (|delta| < threshold).")
    parser.add_argument("--power", type=float, default=2.0,
                        help="Exponent p in w = 1 + k * |delta|^p.")
    parser.add_argument("--weight-k", type=float, default=3.0,
                        help="Scale factor k in w = 1 + k * |delta|^p.")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--trees", type=int, default=20)
    parser.add_argument("--tree-depth", type=int, default=8)
    parser.add_argument("--tree-sample", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ── data ──────────────────────────────────────────────────────────────────────

def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def to_float(row: dict[str, str], col: str) -> float:
    return float(row[col])


def time_features(ts: datetime) -> list[float]:
    minute_of_day = ts.hour * 60 + ts.minute
    day_angle = 2.0 * math.pi * minute_of_day / (24.0 * 60.0)
    year_angle = 2.0 * math.pi * ts.timetuple().tm_yday / 365.25
    return [math.sin(day_angle), math.cos(day_angle),
            math.sin(year_angle), math.cos(year_angle)]


def feature_names_for(inputs: list[str], window: int) -> list[str]:
    names: list[str] = []
    for offset in range(window, -1, -1):
        suffix = "t" if offset == 0 else f"t-{offset}"
        names.extend(f"{col}_{suffix}" for col in inputs)
    for offset in range(window, 0, -1):
        left = "t" if offset - 1 == 0 else f"t-{offset-1}"
        right = f"t-{offset}"
        names.extend(f"d_{col}_{left}_{right}" for col in inputs)
    names.extend(["hour_sin", "hour_cos", "year_sin", "year_cos"])
    return names


def build_sequence_features(
    rows: list[dict[str, str]],
    idx: int,
    inputs: list[str],
    window: int,
    ts: datetime,
) -> list[float]:
    vals: dict[str, list[float]] = {col: [] for col in inputs}
    feats: list[float] = []
    for src in range(idx - window, idx + 1):
        row = rows[src]
        for col in inputs:
            v = to_float(row, col)
            vals[col].append(v)
            feats.append(v)
    for pos in range(1, window + 1):
        for col in inputs:
            feats.append(vals[col][pos] - vals[col][pos - 1])
    feats.extend(time_features(ts))
    return feats


def build_dataset(rows: list[dict[str, str]], freq_min: int, window: int) -> SequenceDataset:
    expected_step = timedelta(minutes=freq_min)
    parsed_times = [datetime.fromisoformat(r["timestamp"]) for r in rows]
    timestamps: list[str] = []
    features: dict[str, list[list[float]]] = {name: [] for name in SUBSYSTEMS}
    current_values: dict[str, list[float]] = {cfg["target"]: [] for cfg in SUBSYSTEMS.values()}
    targets: dict[str, list[float]] = {cfg["target"]: [] for cfg in SUBSYSTEMS.values()}
    deltas: dict[str, list[float]] = {cfg["target"]: [] for cfg in SUBSYSTEMS.values()}
    names = {name: feature_names_for(cfg["inputs"], window) for name, cfg in SUBSYSTEMS.items()}

    for idx in range(window, len(rows) - 1):
        if any(
            parsed_times[p + 1] - parsed_times[p] != expected_step
            for p in range(idx - window, idx + 1)
        ):
            continue
        try:
            sub_feats = {
                name: build_sequence_features(rows, idx, cfg["inputs"], window, parsed_times[idx])
                for name, cfg in SUBSYSTEMS.items()
            }
            now_row = rows[idx]
            next_row = rows[idx + 1]
            now_targets = {cfg["target"]: to_float(now_row, cfg["target"]) for cfg in SUBSYSTEMS.values()}
            next_targets = {cfg["target"]: to_float(next_row, cfg["target"]) for cfg in SUBSYSTEMS.values()}
        except (KeyError, ValueError):
            continue

        timestamps.append(next_row["timestamp"])
        for name, fv in sub_feats.items():
            features[name].append(fv)
        for target in current_values:
            current_values[target].append(now_targets[target])
            targets[target].append(next_targets[target])
            deltas[target].append(next_targets[target] - now_targets[target])

    return SequenceDataset(timestamps, features, current_values, targets, deltas, names)


# ── piecewise weight function ─────────────────────────────────────────────────

def piecewise_weight(delta: float, threshold: float, w_min: float, k: float, power: float) -> float:
    """
    Two-segment weight:
      |delta| <  threshold  ->  w_min
      |delta| >= threshold  ->  1 + k * |delta|^power
    """
    abs_d = abs(delta)
    if abs_d < threshold:
        return w_min
    return 1.0 + k * (abs_d ** power)


def compute_weights(
    deltas: list[float],
    threshold: float,
    w_min: float,
    k: float,
    power: float,
) -> list[float]:
    return [piecewise_weight(d, threshold, w_min, k, power) for d in deltas]


# ── weighted Ridge ────────────────────────────────────────────────────────────

def fit_scaler(features: list[list[float]]) -> tuple[list[float], list[float]]:
    n_feat = len(features[0])
    means, stds = [], []
    for col in range(n_feat):
        vals = [row[col] for row in features]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
        std = math.sqrt(var)
        means.append(mean)
        stds.append(std if std > 1e-12 else 1.0)
    return means, stds


def scale_row(row: list[float], means: list[float], stds: list[float]) -> list[float]:
    return [(v - m) / s for v, m, s in zip(row, means, stds)]


def solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    aug = [matrix[i][:] + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("singular matrix")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        pv = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pv
        for r in range(n):
            if r == col:
                continue
            f = aug[r][col]
            if f == 0:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= f * aug[col][j]
    return [aug[i][n] for i in range(n)]


def train_weighted_ridge(
    features: list[list[float]],
    y: list[float],
    sample_weights: list[float],
    alpha: float,
) -> RidgeModel:
    means, stds = fit_scaler(features)
    n = len(features[0]) + 1
    xtwx = [[0.0] * n for _ in range(n)]
    xtwy = [0.0] * n
    for row, target, w in zip(features, y, sample_weights):
        x = [1.0] + scale_row(row, means, stds)
        for i in range(n):
            xtwy[i] += w * x[i] * target
            for j in range(n):
                xtwx[i][j] += w * x[i] * x[j]
    for i in range(1, n):
        xtwx[i][i] += alpha
    return RidgeModel(solve_linear(xtwx, xtwy), means, stds)


def predict_ridge(model: RidgeModel, current: float, row: list[float]) -> float:
    x = [1.0] + scale_row(row, model.means, model.stds)
    delta = sum(w * v for w, v in zip(model.weights, x))
    return current + delta


# ── weighted Tree ensemble ────────────────────────────────────────────────────

def candidate_thresholds(values: list[float], rng: random.Random, count: int = 16) -> list[float]:
    if not values:
        return []
    if len(values) <= count:
        return sorted(set(values))
    sv = sorted(values)
    thresholds: set[float] = set()
    for q in range(1, count + 1):
        pos = int(q * (len(sv) - 1) / (count + 1))
        thresholds.add(sv[pos])
    for _ in range(min(4, len(sv))):
        thresholds.add(sv[rng.randrange(len(sv))])
    return sorted(thresholds)


def build_tree(
    features: list[list[float]],
    y: list[float],
    weights: list[float],
    indexes: list[int],
    depth: int,
    max_depth: int,
    min_leaf: int,
    max_features: int,
    rng: random.Random,
) -> TreeNode:
    w_sum = sum(weights[i] for i in indexes)
    value = sum(weights[i] * y[i] for i in indexes) / w_sum if w_sum > 0 else 0.0
    if depth >= max_depth or len(indexes) < min_leaf * 2:
        return TreeNode(value=value)

    n_feat = len(features[0])
    feat_ids = rng.sample(range(n_feat), min(max_features, n_feat))

    w_sq = sum(weights[i] * y[i] * y[i] for i in indexes)
    w_y = sum(weights[i] * y[i] for i in indexes)
    parent_wsse = w_sq - w_y * w_y / w_sum

    best: tuple[float, int, float, list[int], list[int]] | None = None
    for fid in feat_ids:
        vals = [features[i][fid] for i in indexes]
        for thresh in candidate_thresholds(vals, rng):
            left: list[int] = []
            right: list[int] = []
            lw = lwy = lwsq = rw = rwy = rwsq = 0.0
            for i in indexes:
                wi, yi = weights[i], y[i]
                if features[i][fid] <= thresh:
                    left.append(i)
                    lw += wi; lwy += wi * yi; lwsq += wi * yi * yi
                else:
                    right.append(i)
                    rw += wi; rwy += wi * yi; rwsq += wi * yi * yi
            if len(left) < min_leaf or len(right) < min_leaf:
                continue
            l_wsse = lwsq - lwy * lwy / lw if lw > 0 else 0.0
            r_wsse = rwsq - rwy * rwy / rw if rw > 0 else 0.0
            gain = parent_wsse - (l_wsse + r_wsse)
            if best is None or gain > best[0]:
                best = (gain, fid, thresh, left, right)

    if best is None or best[0] <= 1e-9:
        return TreeNode(value=value)

    _, fid, thresh, left, right = best
    return TreeNode(
        value=value,
        feature_idx=fid,
        threshold=thresh,
        left=build_tree(features, y, weights, left, depth + 1, max_depth, min_leaf, max_features, rng),
        right=build_tree(features, y, weights, right, depth + 1, max_depth, min_leaf, max_features, rng),
    )


def train_weighted_tree_ensemble(
    features: list[list[float]],
    deltas: list[float],
    sample_weights: list[float],
    n_trees: int,
    max_depth: int,
    max_samples: int,
    seed: int,
) -> TreeEnsemble:
    rng = random.Random(seed)
    n = len(features)
    sample_size = min(max_samples, n)
    min_leaf = max(20, sample_size // 300)
    max_features = max(1, int(math.sqrt(len(features[0]))))
    trees: list[TreeNode] = []

    total_w = sum(sample_weights)
    cum: list[float] = []
    running = 0.0
    for w in sample_weights:
        running += w
        cum.append(running)

    for t in range(n_trees):
        indexes: list[int] = []
        for _ in range(sample_size):
            r = rng.random() * total_w
            lo, hi = 0, n - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if cum[mid] < r:
                    lo = mid + 1
                else:
                    hi = mid
            indexes.append(lo)
        trees.append(
            build_tree(features, deltas, sample_weights, indexes, 0, max_depth, min_leaf, max_features, rng)
        )
    return TreeEnsemble(trees)


def predict_tree_node(node: TreeNode, row: list[float]) -> float:
    while node.feature_idx is not None and node.left is not None and node.right is not None:
        node = node.left if row[node.feature_idx] <= node.threshold else node.right  # type: ignore[arg-type]
    return node.value


def predict_tree(model: TreeEnsemble, current: float, row: list[float]) -> float:
    delta = sum(predict_tree_node(t, row) for t in model.trees) / len(model.trees)
    return current + delta


# ── metrics ───────────────────────────────────────────────────────────────────

def metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    errors = [p - t for t, p in zip(y_true, y_pred)]
    n = len(errors)
    mae = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    mean_y = sum(y_true) / n
    sse = sum(e * e for e in errors)
    sst = sum((t - mean_y) ** 2 for t in y_true)
    return {
        "n": float(n),
        "mae": mae,
        "rmse": rmse,
        "r2": 1.0 - sse / sst if sst > 0 else float("nan"),
        "bias": sum(errors) / n,
    }


def fmt_metrics(split: str, model: str, v: dict[str, float]) -> str:
    return (
        f"{split:10s} {model:18s} n={int(v['n']):6d} "
        f"MAE={v['mae']:.4f} RMSE={v['rmse']:.4f} R2={v['r2']:.4f} bias={v['bias']:+.4f}"
    )


def closer_stats(
    true_vals: list[float],
    pred_vals: list[float],
    prev_vals: list[float],
) -> tuple[float, float, float]:
    cc = cp = tied = 0
    for t, p, prev in zip(true_vals, pred_vals, prev_vals):
        dc = abs(p - t)
        dp = abs(p - prev)
        if dc < dp:
            cc += 1
        elif dp < dc:
            cp += 1
        else:
            tied += 1
    n = len(true_vals)
    return cc / n, cp / n, tied / n


# ── output ────────────────────────────────────────────────────────────────────

def write_predictions(
    path: Path,
    ds: SequenceDataset,
    models: dict[str, dict[str, object]],
) -> list[dict[str, str]]:
    columns = ["timestamp"]
    for cfg in SUBSYSTEMS.values():
        t = cfg["target"]
        columns.extend([f"{t}_true", f"{t}_persistence", f"{t}_pw_ridge", f"{t}_pw_tree"])

    out_rows: list[dict[str, str]] = []
    for idx, ts in enumerate(ds.timestamps):
        row: dict[str, str] = {"timestamp": ts}
        for subsystem, cfg in SUBSYSTEMS.items():
            target = cfg["target"]
            current = ds.current_values[target][idx]
            feats = ds.features[subsystem][idx]
            ridge = models[subsystem]["ridge"]
            tree = models[subsystem]["tree"]
            row[f"{target}_true"] = f"{ds.targets[target][idx]:.8g}"
            row[f"{target}_persistence"] = f"{current:.8g}"
            row[f"{target}_pw_ridge"] = f"{predict_ridge(ridge, current, feats):.8g}"  # type: ignore[arg-type]
            row[f"{target}_pw_tree"] = f"{predict_tree(tree, current, feats):.8g}"  # type: ignore[arg-type]
        out_rows.append(row)

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(out_rows)
    return out_rows


def compute_plot_metrics(rows: list[dict[str, str]], target: str, series: str) -> tuple[float, float, float]:
    errors = [float(r[f"{target}_{series}"]) - float(r[f"{target}_true"]) for r in rows]
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    bias = sum(errors) / len(errors)
    return mae, rmse, bias


def render_html(
    path: Path,
    rows: list[dict[str, str]],
    window: int,
    threshold: float,
    w_min: float,
    k: float,
    power: float,
    split_label: str = "validation",
) -> str:
    series_keys = ["persistence", "pw_ridge", "pw_tree"]
    colors = {"persistence": "#9ca3af", "pw_ridge": "#2563eb", "pw_tree": "#dc2626"}
    labels = {"persistence": "Persistence", "pw_ridge": "PW Ridge", "pw_tree": "PW Tree"}

    plot_data = {
        "timestamps": [r["timestamp"] for r in rows],
        "targets": {
            target: {
                "values": {s: [float(r[f"{target}_{s}"]) for r in rows] for s in ["true"] + series_keys},
                "errors": {
                    s: [float(r[f"{target}_{s}"]) - float(r[f"{target}_true"]) for r in rows]
                    for s in series_keys
                },
            }
            for target in ["T_rec", "T_sup"]
        },
        "valueSeries": ["true"] + series_keys,
        "errorSeries": series_keys,
        "colors": {"true": "#111827", **colors},
        "labels": {"true": "True", **labels},
        "expectedStepMinutes": 5,
    }
    data_json = json.dumps(plot_data, separators=(",", ":")).replace("</", "<\\/")

    metric_rows_html = []
    for target in ["T_rec", "T_sup"]:
        for s, label in labels.items():
            mae, rmse, bias = compute_plot_metrics(rows, target, s)
            metric_rows_html.append(
                f"<tr><td>{target}</td><td>{label}</td>"
                f"<td>{mae:.4f}</td><td>{rmse:.4f}</td><td>{bias:+.4f}</td></tr>"
            )

    sections = []
    for target in ["T_rec", "T_sup"]:
        for kind, title in [("values", "value prediction"), ("errors", "prediction error")]:
            key = f"{target}_{kind}"
            sections.append(f"""
        <section class="panel">
          <div class="panel-head">
            <div>
              <div class="panel-title">{target} {title}</div>
              <div class="window-label" data-window="{key}"></div>
            </div>
            <div class="tools">
              <button type="button" data-action="zoom-in" data-target="{key}">Zoom in</button>
              <button type="button" data-action="zoom-out" data-target="{key}">Zoom out</button>
              <button type="button" data-action="reset" data-target="{key}">Reset</button>
            </div>
          </div>
          <canvas data-chart="{key}" data-target-name="{target}" data-kind="{kind}" height="360"></canvas>
          <div class="readout" data-readout="{key}"></div>
        </section>""")

    script = _canvas_script(data_json)
    first = html.escape(rows[0]["timestamp"])
    last = html.escape(rows[-1]["timestamp"])
    legend_html = "".join(
        f'<span class="legend-item"><span class="swatch" style="background:{c}"></span>{labels[s]}</span>'
        for s, c in colors.items()
    )
    weight_desc = (
        f"w = {w_min} if |Δ| &lt; {threshold}, "
        f"else w = 1 + {k} × |Δ|<sup>{power}</sup>"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AHU v4 Piecewise-Weighted 5-min Predictions</title>
  <style>
    :root {{ font-family: Arial, Helvetica, sans-serif; background: #f3f4f6; color: #111827; }}
    body {{ margin: 0; padding: 28px; }}
    main {{ max-width: 1120px; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .meta {{ color: #4b5563; font-size: 14px; line-height: 1.5; }}
    .note {{ background: #fff; border: 1px solid #d1d5db; border-radius: 8px; padding: 12px 14px; margin: 12px 0; font-size: 14px; color: #374151; }}
    .legend {{ display: flex; gap: 18px; margin: 12px 0; flex-wrap: wrap; font-size: 14px; }}
    .legend-item {{ display: inline-flex; align-items: center; gap: 7px; }}
    .swatch {{ width: 28px; height: 3px; border-radius: 999px; display: inline-block; }}
    .panel {{ background: #fff; border: 1px solid #d1d5db; border-radius: 8px; margin: 14px 0; padding: 14px 16px 12px; }}
    .panel-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 8px; }}
    .panel-title {{ font-size: 16px; font-weight: 700; }}
    .window-label {{ color: #6b7280; font-size: 12px; margin-top: 3px; }}
    .tools {{ display: flex; gap: 6px; flex-wrap: wrap; justify-content: flex-end; }}
    button {{ border: 1px solid #cbd5e1; background: #fff; color: #111827; border-radius: 6px; padding: 6px 9px; font-size: 12px; cursor: pointer; }}
    button:hover {{ background: #f3f4f6; }}
    canvas {{ width: 100%; height: 360px; display: block; border: 1px solid #e5e7eb; cursor: grab; touch-action: none; }}
    canvas:active {{ cursor: grabbing; }}
    .readout {{ min-height: 18px; margin-top: 8px; color: #374151; font-size: 13px; font-variant-numeric: tabular-nums; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d1d5db; border-radius: 8px; overflow: hidden; font-size: 14px; margin-bottom: 18px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e5e7eb; text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ background: #f9fafb; font-weight: 700; }}
  </style>
</head>
<body>
  <main>
    <h1>AHU v4 Piecewise-Weighted — 5-min Predictions ({split_label})</h1>
    <p class="meta">Source: {html.escape(str(path))}<br />
    Time range: {first} to {last}. Rows: {len(rows)}. Window: [t-{window} … t].</p>
    <div class="note">
      <b>Piecewise weight function:</b> {weight_desc}<br />
      Near-stable steps (|Δ| &lt; {threshold}) receive weight {w_min} — heavily discounted.<br />
      Changing steps receive power-law boosted weight, forcing the model to learn larger transitions.
    </div>
    <div class="legend">
      <span class="legend-item"><span class="swatch" style="background:#111827"></span>True</span>
      {legend_html}
    </div>
    <table>
      <thead><tr><th>Target</th><th>Model</th><th>MAE</th><th>RMSE</th><th>Bias</th></tr></thead>
      <tbody>{''.join(metric_rows_html)}</tbody>
    </table>
    {''.join(sections)}
  </main>
  {script}
</body>
</html>
"""


def _canvas_script(data_json: str) -> str:
    return f"""<script>
const AHU_DATA={data_json};
const charts=new Map();
function clamp(v,a,b){{return Math.max(a,Math.min(b,v));}}
function fmtTs(t){{return t.slice(0,16);}}
function minsBetween(a,b){{return(Date.parse(b.replace(' ','T'))-Date.parse(a.replace(' ','T')))/60000;}}
function isCont(pi,i){{if(pi===null||i<=0)return true;return Math.abs(minsBetween(AHU_DATA.timestamps[pi],AHU_DATA.timestamps[i])-AHU_DATA.expectedStepMinutes)<1e-6;}}
function makeChart(canvas){{const n=AHU_DATA.timestamps.length;return{{key:canvas.dataset.chart,canvas,ctx:canvas.getContext('2d'),target:canvas.dataset.targetName,kind:canvas.dataset.kind,start:0,end:n-1,hoverIndex:null,dragging:false,dragStartX:0,dragStartStart:0,dragStartEnd:n-1,cssWidth:0,cssHeight:0}};}}
function seriesList(c){{return c.kind==='values'?AHU_DATA.valueSeries:AHU_DATA.errorSeries;}}
function seriesVals(c){{return AHU_DATA.targets[c.target][c.kind];}}
function chartArea(c){{return{{left:66,top:22,width:c.cssWidth-90,height:c.cssHeight-74}};}}
function visBounds(c){{const vals=seriesVals(c);const l=Math.max(0,Math.floor(c.start));const r=Math.min(AHU_DATA.timestamps.length-1,Math.ceil(c.end));let mn=c.kind==='errors'?0:Infinity,mx=c.kind==='errors'?0:-Infinity;for(const s of seriesList(c)){{const a=vals[s];for(let i=l;i<=r;i++){{if(a[i]<mn)mn=a[i];if(a[i]>mx)mx=a[i];}}}}const mg=(mx-mn)*0.10||0.1;return{{min:mn-mg,max:mx+mg}};}}
function resizeCanvas(c){{const rect=c.canvas.getBoundingClientRect();const dpr=window.devicePixelRatio||1;const w=Math.max(320,Math.floor(rect.width));const h=Number(c.canvas.getAttribute('height'))||360;c.canvas.width=Math.floor(w*dpr);c.canvas.height=Math.floor(h*dpr);c.ctx.setTransform(dpr,0,0,dpr,0,0);c.cssWidth=w;c.cssHeight=h;}}
function xFor(c,i,a){{return a.left+((i-c.start)/(c.end-c.start||1))*a.width;}}
function iForX(c,x,a){{return c.start+((x-a.left)/a.width)*(c.end-c.start);}}
function yFor(v,a,b){{return a.top+a.height-((v-b.min)/(b.max-b.min||1))*a.height;}}
function drawLine(ctx,c,arr,a,b,color,lw,dash){{ctx.save();ctx.strokeStyle=color;ctx.lineWidth=lw;ctx.setLineDash(dash||[]);ctx.beginPath();const l=Math.max(0,Math.floor(c.start));const r=Math.min(AHU_DATA.timestamps.length-1,Math.ceil(c.end));let started=false;let pi=null;for(let i=l;i<=r;i++){{const x=xFor(c,i,a);const y=yFor(arr[i],a,b);if(!started||!isCont(pi,i)){{ctx.moveTo(x,y);started=true;}}else{{ctx.lineTo(x,y);}}pi=i;}}ctx.stroke();ctx.restore();}}
function drawChart(c){{const ctx=c.ctx;const w=c.cssWidth;const h=c.cssHeight;if(!w||!h)return;const a=chartArea(c);const b=visBounds(c);ctx.clearRect(0,0,w,h);ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);ctx.strokeStyle='#e5e7eb';ctx.lineWidth=1;ctx.fillStyle='#6b7280';ctx.font='12px Arial';ctx.textAlign='right';ctx.textBaseline='middle';for(let tick=0;tick<=4;tick++){{const v=b.min+(b.max-b.min)*tick/4;const y=yFor(v,a,b);ctx.beginPath();ctx.moveTo(a.left,y);ctx.lineTo(a.left+a.width,y);ctx.stroke();ctx.fillText(v.toFixed(c.kind==='errors'?3:2),a.left-8,y);}}if(c.kind==='errors'){{const zy=yFor(0,a,b);ctx.save();ctx.strokeStyle='#111827';ctx.setLineDash([5,5]);ctx.beginPath();ctx.moveTo(a.left,zy);ctx.lineTo(a.left+a.width,zy);ctx.stroke();ctx.restore();}}const vals=seriesVals(c);for(const s of seriesList(c))drawLine(ctx,c,vals[s],a,b,AHU_DATA.colors[s],s==='true'?1.7:1.4,[]);ctx.strokeStyle='#9ca3af';ctx.strokeRect(a.left,a.top,a.width,a.height);const li=Math.max(0,Math.round(c.start));const ri=Math.min(AHU_DATA.timestamps.length-1,Math.round(c.end));ctx.fillStyle='#6b7280';ctx.textAlign='left';ctx.textBaseline='top';ctx.fillText(fmtTs(AHU_DATA.timestamps[li]),a.left,a.top+a.height+12);ctx.textAlign='right';ctx.fillText(fmtTs(AHU_DATA.timestamps[ri]),a.left+a.width,a.top+a.height+12);if(c.hoverIndex!==null){{const i=clamp(c.hoverIndex,li,ri);const x=xFor(c,i,a);ctx.strokeStyle='#374151';ctx.beginPath();ctx.moveTo(x,a.top);ctx.lineTo(x,a.top+a.height);ctx.stroke();const parts=[fmtTs(AHU_DATA.timestamps[i])];for(const s of seriesList(c))parts.push(`${{AHU_DATA.labels[s]}}=${{vals[s][i].toFixed(c.kind==='errors'?4:3)}}`);const ro=document.querySelector(`[data-readout="${{c.key}}"]`);if(ro)ro.textContent=parts.join(' | ');}}const lbl=document.querySelector(`[data-window="${{c.key}}"]`);if(lbl)lbl.textContent=`${{fmtTs(AHU_DATA.timestamps[li])}} to ${{fmtTs(AHU_DATA.timestamps[ri])}} (${{ri-li+1}} points)`;}}
function zoom(c,ci,factor){{const n=AHU_DATA.timestamps.length;const os=c.end-c.start;const ns=clamp(os*factor,20,n-1);let s=ci-ns/2;let e=ci+ns/2;if(s<0){{e-=s;s=0;}}if(e>n-1){{s-=e-(n-1);e=n-1;}}c.start=clamp(s,0,n-1);c.end=clamp(e,0,n-1);drawChart(c);}}
function resetChart(c){{c.start=0;c.end=AHU_DATA.timestamps.length-1;drawChart(c);}}
function attachEvents(c){{c.canvas.addEventListener('wheel',(e)=>{{e.preventDefault();const rect=c.canvas.getBoundingClientRect();const a=chartArea(c);const ci=clamp(iForX(c,e.clientX-rect.left,a),0,AHU_DATA.timestamps.length-1);zoom(c,ci,e.deltaY<0?0.75:1.35);}},{{passive:false}});c.canvas.addEventListener('pointerdown',(e)=>{{c.dragging=true;c.dragStartX=e.clientX;c.dragStartStart=c.start;c.dragStartEnd=c.end;c.canvas.setPointerCapture(e.pointerId);}});c.canvas.addEventListener('pointermove',(e)=>{{const rect=c.canvas.getBoundingClientRect();const a=chartArea(c);const x=e.clientX-rect.left;if(c.dragging){{const dx=e.clientX-c.dragStartX;const span=c.dragStartEnd-c.dragStartStart;const shift=-dx/a.width*span;c.start=c.dragStartStart+shift;c.end=c.dragStartEnd+shift;if(c.start<0){{c.end-=c.start;c.start=0;}}if(c.end>AHU_DATA.timestamps.length-1){{c.start-=c.end-(AHU_DATA.timestamps.length-1);c.end=AHU_DATA.timestamps.length-1;}}}}const nh=clamp(Math.round(iForX(c,x,a)),0,AHU_DATA.timestamps.length-1);if(c.dragging||nh!==c.hoverIndex){{c.hoverIndex=nh;drawChart(c);}}}});c.canvas.addEventListener('pointerup',(e)=>{{c.dragging=false;if(c.canvas.hasPointerCapture(e.pointerId))c.canvas.releasePointerCapture(e.pointerId);}});c.canvas.addEventListener('pointerleave',()=>{{c.dragging=false;c.hoverIndex=null;const ro=document.querySelector(`[data-readout="${{c.key}}"]`);if(ro)ro.textContent='';drawChart(c);}});c.canvas.addEventListener('dblclick',()=>resetChart(c));}}
function init(){{document.querySelectorAll('canvas[data-chart]').forEach((canvas)=>{{const c=makeChart(canvas);charts.set(c.key,c);attachEvents(c);resizeCanvas(c);drawChart(c);}});document.querySelectorAll('button[data-action]').forEach((btn)=>btn.addEventListener('click',()=>{{const c=charts.get(btn.dataset.target);if(!c)return;const ci=(c.start+c.end)/2;if(btn.dataset.action==='zoom-in')zoom(c,ci,0.5);if(btn.dataset.action==='zoom-out')zoom(c,ci,2.0);if(btn.dataset.action==='reset')resetChart(c);}}));window.addEventListener('resize',()=>{{for(const c of charts.values()){{resizeCanvas(c);drawChart(c);}}}});}}
init();
</script>"""


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train = build_dataset(load_rows(split_dir / "train_5min.csv"), args.freq_min, args.window)
    validation = build_dataset(load_rows(split_dir / "validation_5min.csv"), args.freq_min, args.window)
    datasets = {"train": train, "validation": validation}

    weight_desc = (
        f"w = {args.w_min} if |delta| < {args.threshold}, "
        f"else w = 1 + {args.weight_k} * |delta|^{args.power}"
    )
    report = [
        "AHU two-stage v4 piecewise-weighted delta model report",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Split directory: {split_dir.resolve()}",
        "Prediction horizon: 5 min (one step ahead)",
        f"Sequence window: [t-{args.window} ... t]",
        f"Weight function: {weight_desc}",
        "Target: delta = X(t+1) - X(t), reconstructed as X_pred(t+1) = X(t) + delta_pred.",
        "Gap boundaries skipped (consecutive 5-min check on full window).",
        "Subsystem inputs:",
        "  heat_exchanger: T_out, T_ret, T_rec, pressure_hex",
        "  heating_coil:   T_rec, T_coil, T_sup",
        "",
    ]

    models: dict[str, dict[str, object]] = {}
    for subsystem, cfg in SUBSYSTEMS.items():
        target = cfg["target"]
        train_deltas = train.deltas[target]
        train_weights = compute_weights(
            train_deltas, args.threshold, args.w_min, args.weight_k, args.power
        )
        train_feats = train.features[subsystem]

        w_vals = sorted(train_weights)
        n_w = len(w_vals)
        print(
            f"Training {subsystem} ({target})  "
            f"weight range [{w_vals[0]:.3f}, {w_vals[-1]:.3f}]  "
            f"median={w_vals[n_w//2]:.3f}"
        )

        ridge = train_weighted_ridge(train_feats, train_deltas, train_weights, args.ridge_alpha)
        tree = train_weighted_tree_ensemble(
            train_feats, train_deltas, train_weights,
            args.trees, args.tree_depth, args.tree_sample,
            args.seed + len(models) * 100,
        )
        models[subsystem] = {"ridge": ridge, "tree": tree}

        report.append(f"Subsystem: {subsystem}  target: {target}(t+1)")
        report.append(f"Feature count: {len(train.feature_names[subsystem])}")
        report.append(f"Train samples: {len(train_feats)}")

        # closer-to analysis + metrics for each split
        for split_name, ds in datasets.items():
            true_vals = ds.targets[target]
            current_vals = ds.current_values[target]
            feats = ds.features[subsystem]
            ridge_pred = [predict_ridge(ridge, c, f) for c, f in zip(current_vals, feats)]  # type: ignore[arg-type]
            tree_pred  = [predict_tree(tree,  c, f) for c, f in zip(current_vals, feats)]  # type: ignore[arg-type]

            report.append(f"Closer-to analysis on {split_name} set:")
            report.append("  (R(t+1) closer to true X(t+1) vs previous X(t)?)")
            for model_name, preds in [("pw_ridge", ridge_pred), ("pw_tree", tree_pred)]:
                cc, cp, ti = closer_stats(true_vals, preds, current_vals)
                report.append(
                    f"  {model_name:12s}  closer_to_X(t+1)={cc:.1%}  "
                    f"closer_to_X(t)={cp:.1%}  tied={ti:.1%}"
                )
            report.append("")
            report.append(f"Metrics on {split_name} set:")
            report.append(fmt_metrics(split_name, "persistence", metrics(true_vals, current_vals)))
            report.append(fmt_metrics(split_name, "pw_ridge",    metrics(true_vals, ridge_pred)))
            report.append(fmt_metrics(split_name, "pw_tree",     metrics(true_vals, tree_pred)))
            report.append("")

    # piecewise/: report + validation csv/html for comparison with v3
    piecewise_dir = out_dir / "piecewise"
    piecewise_dir.mkdir(parents=True, exist_ok=True)
    report_path = piecewise_dir / "ahu_v4_piecewise_model_report_5min.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {report_path}")

    val_pred_path = piecewise_dir / "ahu_v4_piecewise_predictions_5min.csv"
    val_html_path = piecewise_dir / "ahu_v4_piecewise_predictions_5min.html"
    val_predictions = write_predictions(val_pred_path, validation, models)
    val_html_path.write_text(
        render_html(val_html_path, val_predictions, args.window,
                    args.threshold, args.w_min, args.weight_k, args.power,
                    split_label="validation"),
        encoding="utf-8",
    )
    print(f"Wrote {val_pred_path}")
    print(f"Wrote {val_html_path}")

    # train/ and validation/: full per-split predictions
    for split_name, ds in datasets.items():
        split_dir_out = out_dir / split_name
        split_dir_out.mkdir(parents=True, exist_ok=True)
        pred_path = split_dir_out / f"ahu_v4_piecewise_{split_name}_predictions_5min.csv"
        html_path = split_dir_out / f"ahu_v4_piecewise_{split_name}_predictions_5min.html"
        predictions = write_predictions(pred_path, ds, models)
        html_path.write_text(
            render_html(html_path, predictions, args.window,
                        args.threshold, args.w_min, args.weight_k, args.power,
                        split_label=split_name),
            encoding="utf-8",
        )
        print(f"Wrote {pred_path}")
        print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
