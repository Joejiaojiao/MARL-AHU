"""
Two-stage AHU dynamic models using only measured temperatures and pressure.

The script trains two one-step-ahead subsystem models from the existing
chronological train/validation/test split:

  heat_exchanger: predict T_rec(t+1)
  heating_coil:   predict T_sup(t+1)

Only these measured variables are used:
T_out, T_ret, T_rec, T_coil, T_sup, pressure_hex.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


SENSOR_COLUMNS = ["T_out", "T_ret", "T_rec", "T_coil", "T_sup", "pressure_hex"]
TARGETS = {
    "heat_exchanger": "T_rec",
    "heating_coil": "T_sup",
}


@dataclass
class Dataset:
    timestamps: list[str]
    features: list[list[float]]
    current_values: dict[str, list[float]]
    targets: dict[str, list[float]]
    deltas: dict[str, list[float]]
    feature_names: list[str]


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
    residual_target: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AHU heat exchanger and heating coil models.")
    parser.add_argument("--split-dir", default="data_preprocessing/splits", help="Directory containing train_5min.csv, validation_5min.csv, test_5min.csv.")
    parser.add_argument("--out-dir", default="model_outputs", help="Directory for model reports and visualizations.")
    parser.add_argument("--freq-min", type=int, default=5, help="Expected sample interval in minutes.")
    parser.add_argument(
        "--horizons",
        default="1",
        help="Comma-separated prediction horizons in time steps. With 5-min data: 3,6 = 15,30 min.",
    )
    parser.add_argument("--lags", type=int, default=3, help="Number of past rows to include as lag features.")
    parser.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge L2 regularization strength.")
    parser.add_argument("--trees", type=int, default=18, help="Number of trees in the tree ensemble.")
    parser.add_argument("--tree-depth", type=int, default=8, help="Maximum depth for each tree.")
    parser.add_argument("--tree-sample", type=int, default=18000, help="Maximum bootstrap samples per tree.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def parse_horizons(text: str) -> list[int]:
    horizons = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("--horizons must contain positive integers")
    return horizons


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def to_float(row: dict[str, str], col: str) -> float:
    return float(row[col])


def time_features(timestamp: datetime) -> list[float]:
    minute_of_day = timestamp.hour * 60 + timestamp.minute
    day_angle = 2.0 * math.pi * minute_of_day / (24.0 * 60.0)
    year_angle = 2.0 * math.pi * timestamp.timetuple().tm_yday / 365.25
    return [math.sin(day_angle), math.cos(day_angle), math.sin(year_angle), math.cos(year_angle)]


def build_feature_names(lags: int) -> list[str]:
    names: list[str] = []
    for lag in range(lags + 1):
        suffix = "t" if lag == 0 else f"t-{lag}"
        names.extend(f"{col}_{suffix}" for col in SENSOR_COLUMNS)
    for lag in range(lags):
        left = "t" if lag == 0 else f"t-{lag}"
        right = f"t-{lag + 1}"
        names.extend(f"delta_{col}_{left}_minus_{right}" for col in SENSOR_COLUMNS)
    names.extend(
        [
            "delta_T_ret_out_t",
            "delta_T_rec_out_t",
            "delta_T_coil_rec_t",
            "delta_T_sup_coil_t",
            "hour_sin",
            "hour_cos",
            "year_sin",
            "year_cos",
        ]
    )
    return names


def build_dataset(rows: list[dict[str, str]], freq_min: int, lags: int, horizon_steps: int) -> Dataset:
    expected_step = timedelta(minutes=freq_min)
    parsed_times = [datetime.fromisoformat(row["timestamp"]) for row in rows]
    feature_names = build_feature_names(lags)
    timestamps: list[str] = []
    features: list[list[float]] = []
    current_values = {target: [] for target in TARGETS.values()}
    targets = {target: [] for target in TARGETS.values()}
    deltas = {target: [] for target in TARGETS.values()}

    for idx in range(lags, len(rows) - horizon_steps):
        first_required = idx - lags
        last_required = idx + horizon_steps
        if any(
            parsed_times[pos + 1] - parsed_times[pos] != expected_step
            for pos in range(first_required, last_required)
        ):
            continue

        try:
            row_features: list[float] = []
            lagged_values: dict[str, list[float]] = {col: [] for col in SENSOR_COLUMNS}
            for lag in range(lags + 1):
                source = rows[idx - lag]
                for col in SENSOR_COLUMNS:
                    value = to_float(source, col)
                    row_features.append(value)
                    lagged_values[col].append(value)

            for lag in range(lags):
                for col in SENSOR_COLUMNS:
                    row_features.append(lagged_values[col][lag] - lagged_values[col][lag + 1])

            current = rows[idx]
            t_out = to_float(current, "T_out")
            t_ret = to_float(current, "T_ret")
            t_rec = to_float(current, "T_rec")
            t_coil = to_float(current, "T_coil")
            t_sup = to_float(current, "T_sup")
            row_features.extend(
                [
                    t_ret - t_out,
                    t_rec - t_out,
                    t_coil - t_rec,
                    t_sup - t_coil,
                    *time_features(parsed_times[idx]),
                ]
            )
            next_row = rows[idx + horizon_steps]
            next_targets = {target: to_float(next_row, target) for target in TARGETS.values()}
            now_targets = {target: to_float(current, target) for target in TARGETS.values()}
        except (KeyError, ValueError):
            continue

        timestamps.append(next_row["timestamp"])
        features.append(row_features)
        for target in TARGETS.values():
            targets[target].append(next_targets[target])
            current_values[target].append(now_targets[target])
            deltas[target].append(next_targets[target] - now_targets[target])

    return Dataset(timestamps, features, current_values, targets, deltas, feature_names)


def fit_scaler(features: list[list[float]]) -> tuple[list[float], list[float]]:
    n_features = len(features[0])
    means: list[float] = []
    stds: list[float] = []
    for col in range(n_features):
        values = [row[col] for row in features]
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
        std = math.sqrt(var)
        means.append(mean)
        stds.append(std if std > 1e-12 else 1.0)
    return means, stds


def scale_row(row: list[float], means: list[float], stds: list[float]) -> list[float]:
    return [(value - mean) / std for value, mean, std in zip(row, means, stds)]


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    augmented = [matrix[i][:] + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ValueError("singular matrix")
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        pivot_value = augmented[col][col]
        for j in range(col, n + 1):
            augmented[col][j] /= pivot_value
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0:
                continue
            for j in range(col, n + 1):
                augmented[row][j] -= factor * augmented[col][j]
    return [augmented[i][n] for i in range(n)]


def train_ridge(features: list[list[float]], y: list[float], alpha: float) -> RidgeModel:
    means, stds = fit_scaler(features)
    n = len(features[0]) + 1
    xtx = [[0.0 for _ in range(n)] for _ in range(n)]
    xty = [0.0 for _ in range(n)]
    for row, target in zip(features, y):
        x = [1.0] + scale_row(row, means, stds)
        for i in range(n):
            xty[i] += x[i] * target
            for j in range(n):
                xtx[i][j] += x[i] * x[j]
    for i in range(1, n):
        xtx[i][i] += alpha
    return RidgeModel(solve_linear_system(xtx, xty), means, stds)


def predict_ridge(model: RidgeModel, row: list[float]) -> float:
    x = [1.0] + scale_row(row, model.means, model.stds)
    return sum(weight * value for weight, value in zip(model.weights, x))


def predict_ridge_delta(model: RidgeModel, current_value: float, row: list[float]) -> float:
    return current_value + predict_ridge(model, row)


def variance_from_sums(total: float, total_sq: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return total_sq - (total * total / n)


def candidate_thresholds(values: list[float], rng: random.Random, count: int = 16) -> list[float]:
    if not values:
        return []
    if len(values) <= count:
        return sorted(set(values))
    sorted_values = sorted(values)
    thresholds = set()
    for q in range(1, count + 1):
        pos = int(q * (len(sorted_values) - 1) / (count + 1))
        thresholds.add(sorted_values[pos])
    for _ in range(min(4, len(sorted_values))):
        thresholds.add(sorted_values[rng.randrange(len(sorted_values))])
    return sorted(thresholds)


def build_tree(
    features: list[list[float]],
    y: list[float],
    indexes: list[int],
    depth: int,
    max_depth: int,
    min_leaf: int,
    max_features: int,
    rng: random.Random,
) -> TreeNode:
    y_values = [y[idx] for idx in indexes]
    value = sum(y_values) / len(y_values)
    if depth >= max_depth or len(indexes) < min_leaf * 2:
        return TreeNode(value=value)

    n_features = len(features[0])
    feature_ids = rng.sample(range(n_features), min(max_features, n_features))
    parent_sse = variance_from_sums(sum(y_values), sum(v * v for v in y_values), len(y_values))
    best: tuple[float, int, float, list[int], list[int]] | None = None

    for feature_idx in feature_ids:
        values = [features[idx][feature_idx] for idx in indexes]
        for threshold in candidate_thresholds(values, rng):
            left: list[int] = []
            right: list[int] = []
            left_sum = left_sq = right_sum = right_sq = 0.0
            for idx in indexes:
                target = y[idx]
                if features[idx][feature_idx] <= threshold:
                    left.append(idx)
                    left_sum += target
                    left_sq += target * target
                else:
                    right.append(idx)
                    right_sum += target
                    right_sq += target * target
            if len(left) < min_leaf or len(right) < min_leaf:
                continue
            sse = variance_from_sums(left_sum, left_sq, len(left)) + variance_from_sums(right_sum, right_sq, len(right))
            gain = parent_sse - sse
            if best is None or gain > best[0]:
                best = (gain, feature_idx, threshold, left, right)

    if best is None or best[0] <= 1e-9:
        return TreeNode(value=value)

    _, feature_idx, threshold, left, right = best
    return TreeNode(
        value=value,
        feature_idx=feature_idx,
        threshold=threshold,
        left=build_tree(features, y, left, depth + 1, max_depth, min_leaf, max_features, rng),
        right=build_tree(features, y, right, depth + 1, max_depth, min_leaf, max_features, rng),
    )


def train_tree_ensemble(
    features: list[list[float]],
    y: list[float],
    current: list[float],
    residual_target: str,
    n_trees: int,
    max_depth: int,
    max_samples: int,
    seed: int,
) -> TreeEnsemble:
    rng = random.Random(seed)
    residuals = [target - now for target, now in zip(y, current)]
    n = len(features)
    sample_size = min(max_samples, n)
    min_leaf = max(40, sample_size // 250)
    max_features = max(1, int(math.sqrt(len(features[0]))))
    trees: list[TreeNode] = []
    for _ in range(n_trees):
        indexes = [rng.randrange(n) for _ in range(sample_size)]
        trees.append(build_tree(features, residuals, indexes, 0, max_depth, min_leaf, max_features, rng))
    return TreeEnsemble(trees, residual_target)


def predict_tree_node(node: TreeNode, row: list[float]) -> float:
    while node.feature_idx is not None and node.threshold is not None and node.left is not None and node.right is not None:
        node = node.left if row[node.feature_idx] <= node.threshold else node.right
    return node.value


def predict_tree_delta(model: TreeEnsemble, current_value: float, row: list[float]) -> float:
    delta = sum(predict_tree_node(tree, row) for tree in model.trees) / len(model.trees)
    return current_value + delta


def metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    errors = [pred - true for true, pred in zip(y_true, y_pred)]
    n = len(errors)
    mae = sum(abs(err) for err in errors) / n
    rmse = math.sqrt(sum(err * err for err in errors) / n)
    mean_y = sum(y_true) / n
    sse = sum(err * err for err in errors)
    sst = sum((true - mean_y) ** 2 for true in y_true)
    return {
        "n": float(n),
        "mae": mae,
        "rmse": rmse,
        "r2": 1.0 - sse / sst if sst > 0 else float("nan"),
        "bias": sum(errors) / n,
    }


def evaluate(dataset: Dataset, target: str, predictor) -> dict[str, float]:
    return metrics(dataset.targets[target], [predictor(row) for row in dataset.features])


def evaluate_ridge_delta(dataset: Dataset, target: str, model: RidgeModel) -> dict[str, float]:
    return metrics(
        dataset.targets[target],
        [
            predict_ridge_delta(model, current, row)
            for current, row in zip(dataset.current_values[target], dataset.features)
        ],
    )


def evaluate_tree_delta(dataset: Dataset, target: str, model: TreeEnsemble) -> dict[str, float]:
    return metrics(
        dataset.targets[target],
        [
            predict_tree_delta(model, current, row)
            for current, row in zip(dataset.current_values[target], dataset.features)
        ],
    )


def format_metrics(split: str, model: str, values: dict[str, float]) -> str:
    return (
        f"{split:10s} {model:12s} n={int(values['n']):6d} "
        f"MAE={values['mae']:.4f} RMSE={values['rmse']:.4f} "
        f"R2={values['r2']:.4f} bias={values['bias']:.4f}"
    )


def downsample_indexes(n: int, limit: int = 1600) -> list[int]:
    if n <= limit:
        return list(range(n))
    step = n / limit
    return [int(i * step) for i in range(limit)]


def svg_series(values: list[float], indexes: list[int], vmin: float, vmax: float, width: int, height: int, color: str) -> str:
    if not indexes:
        return ""
    span = vmax - vmin if vmax > vmin else 1.0
    points: list[str] = []
    for pos, idx in enumerate(indexes):
        x = pos * width / max(1, len(indexes) - 1)
        y = height - ((values[idx] - vmin) / span * height)
        points.append(f"{x:.2f},{y:.2f}")
    return f'<polyline fill="none" stroke="{color}" stroke-width="1.4" points="{" ".join(points)}" />'


def write_html(path: Path, predictions: list[dict[str, str]], report: str) -> None:
    targets = ["T_rec", "T_coil"]
    sections: list[str] = []
    for target in targets:
        true = [float(row[f"{target}_true"]) for row in predictions]
        persist = [float(row[f"{target}_persistence"]) for row in predictions]
        ridge = [float(row[f"{target}_ridge_arx"]) for row in predictions]
        tree = [float(row[f"{target}_tree_ensemble"]) for row in predictions]
        indexes = downsample_indexes(len(true))
        all_values = [value for series in [true, persist, ridge, tree] for value in series]
        vmin = min(all_values)
        vmax = max(all_values)
        padding = max(0.2, (vmax - vmin) * 0.08)
        vmin -= padding
        vmax += padding
        svg = "\n".join(
            [
                svg_series(true, indexes, vmin, vmax, 980, 320, "#111827"),
                svg_series(persist, indexes, vmin, vmax, 980, 320, "#9ca3af"),
                svg_series(ridge, indexes, vmin, vmax, 980, 320, "#2563eb"),
                svg_series(tree, indexes, vmin, vmax, 980, 320, "#dc2626"),
            ]
        )
        sections.append(
            f"""
            <section>
              <h2>{html.escape(target)} test prediction</h2>
              <svg viewBox="0 0 980 320" role="img" aria-label="{html.escape(target)} predictions">{svg}</svg>
              <div class="legend">
                <span><b class="true"></b>true</span>
                <span><b class="persist"></b>persistence</span>
                <span><b class="ridge"></b>ridge_arx</span>
                <span><b class="tree"></b>tree_ensemble</span>
              </div>
            </section>
            """
        )

    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AHU two-stage model predictions</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
    h1 {{ font-size: 24px; margin-bottom: 6px; }}
    h2 {{ font-size: 18px; margin: 28px 0 10px; }}
    svg {{ width: 100%; max-width: 1100px; height: auto; border: 1px solid #d1d5db; background: #fff; }}
    pre {{ background: #f3f4f6; padding: 14px; overflow-x: auto; border-radius: 6px; }}
    .legend {{ display: flex; gap: 18px; margin-top: 8px; flex-wrap: wrap; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
    .legend b {{ display: inline-block; width: 22px; height: 3px; }}
    .true {{ background: #111827; }}
    .persist {{ background: #9ca3af; }}
    .ridge {{ background: #2563eb; }}
    .tree {{ background: #dc2626; }}
  </style>
</head>
<body>
  <h1>AHU Heat Exchanger and Heating Coil Models</h1>
  <p>One-step-ahead prediction from five temperatures and pressure difference only.</p>
  {''.join(sections)}
  <h2>Report</h2>
  <pre>{html.escape(report)}</pre>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_predictions(path: Path, test: Dataset, models: dict[str, dict[str, object]]) -> list[dict[str, str]]:
    columns = ["timestamp"]
    for target in TARGETS.values():
        columns.extend([f"{target}_true", f"{target}_persistence", f"{target}_ridge_arx", f"{target}_tree_ensemble"])
    rows_out: list[dict[str, str]] = []
    for idx, timestamp in enumerate(test.timestamps):
        row = {"timestamp": timestamp}
        for target in TARGETS.values():
            features = test.features[idx]
            ridge = models[target]["ridge"]
            tree = models[target]["tree"]
            row[f"{target}_true"] = f"{test.targets[target][idx]:.8g}"
            row[f"{target}_persistence"] = f"{test.current_values[target][idx]:.8g}"
            row[f"{target}_ridge_arx"] = f"{predict_ridge_delta(ridge, test.current_values[target][idx], features):.8g}"  # type: ignore[arg-type]
            row[f"{target}_tree_ensemble"] = f"{predict_tree_delta(tree, test.current_values[target][idx], features):.8g}"  # type: ignore[arg-type]
        rows_out.append(row)

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows_out)
    return rows_out


def main() -> None:
    args = parse_args()
    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = parse_horizons(args.horizons)
    raw_train = load_rows(split_dir / "train_5min.csv")
    raw_validation = load_rows(split_dir / "validation_5min.csv")
    raw_test = load_rows(split_dir / "test_5min.csv")

    header_lines = [
        "AHU two-stage dynamic model report",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Split directory: {split_dir.resolve()}",
        "Inputs: " + ", ".join(SENSOR_COLUMNS),
        "No actuator/control signals are used.",
        "Model target: predict delta = X(t+horizon) - X(t), then reconstruct X_pred = X(t) + delta_pred.",
        f"Horizons: {', '.join(str(horizon * args.freq_min) for horizon in horizons)} min",
        f"Lag rows: {args.lags}",
        f"Tree ensemble: residual/delta trees, trees={args.trees}, max_depth={args.tree_depth}, max_samples={args.tree_sample}",
        "",
    ]
    all_report_lines = header_lines[:]

    from visualize_ahu_two_stage import render_html

    for horizon_steps in horizons:
        horizon_min = horizon_steps * args.freq_min
        suffix = f"{horizon_min}min"
        train = build_dataset(raw_train, args.freq_min, args.lags, horizon_steps)
        validation = build_dataset(raw_validation, args.freq_min, args.lags, horizon_steps)
        test = build_dataset(raw_test, args.freq_min, args.lags, horizon_steps)
        datasets = {"train": train, "validation": validation, "test": test}

        report_lines = [
            f"Horizon: {horizon_min} min ({horizon_steps} steps)",
            f"Feature count: {len(train.feature_names)}",
            f"Samples: train={len(train.features)}, validation={len(validation.features)}, test={len(test.features)}",
            "",
        ]

        models: dict[str, dict[str, object]] = {}
        for subsystem, target in TARGETS.items():
            y_train = train.deltas[target]
            ridge = train_ridge(train.features, y_train, args.ridge_alpha)
            tree = train_tree_ensemble(
                train.features,
                train.targets[target],
                train.current_values[target],
                target,
                args.trees,
                args.tree_depth,
                args.tree_sample,
                args.seed + len(models) * 100 + horizon_steps,
            )
            models[target] = {"ridge": ridge, "tree": tree}

            report_lines.append(f"Subsystem: {subsystem}")
            report_lines.append(f"Target: {target}(t+{horizon_steps})")
            for split_name, dataset in datasets.items():
                persist = metrics(dataset.targets[target], dataset.current_values[target])
                ridge_metrics = evaluate_ridge_delta(dataset, target, ridge)
                tree_metrics = evaluate_tree_delta(dataset, target, tree)
                report_lines.append(format_metrics(split_name, "persistence", persist))
                report_lines.append(format_metrics(split_name, "ridge_arx", ridge_metrics))
                report_lines.append(format_metrics(split_name, "tree_ensemble", tree_metrics))
            report_lines.append("")

            print(f"{subsystem}: trained target {target} horizon {horizon_min} min")

        prediction_path = out_dir / f"ahu_two_stage_test_predictions_{suffix}.csv"
        predictions = write_predictions(prediction_path, test, models)
        horizon_report = "\n".join(report_lines)
        report_path = out_dir / f"ahu_two_stage_model_report_{suffix}.txt"
        html_path = out_dir / f"ahu_two_stage_predictions_{suffix}.html"
        report_path.write_text("\n".join(header_lines + report_lines), encoding="utf-8")
        html_path.write_text(render_html(prediction_path, predictions, predictions, horizon_min), encoding="utf-8")

        print(f"Wrote {report_path}")
        print(f"Wrote {prediction_path}")
        print(f"Wrote {html_path}")
        all_report_lines.extend(report_lines)

    combined_report_path = out_dir / "ahu_two_stage_model_report.txt"
    combined_report_path.write_text("\n".join(all_report_lines), encoding="utf-8")
    print(f"Wrote {combined_report_path}")


if __name__ == "__main__":
    main()
