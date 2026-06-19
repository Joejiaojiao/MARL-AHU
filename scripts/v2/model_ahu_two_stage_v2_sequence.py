"""
AHU two-stage v2 sequence forecasting model.

This keeps the v1 baseline untouched and trains subsystem-specific 5-minute
delta predictors from a sequence window:

  heat_exchanger inputs: T_out, T_ret, T_rec, pressure_hex over [t-window ... t]
  heating_coil inputs:   T_rec, T_coil, T_sup over [t-window ... t]

The target is delta = X(t+1) - X(t), reconstructed as X_pred(t+1) = X(t) + delta.

The implementation uses only the Python standard library because the current
environment does not provide numpy/sklearn/torch.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from model_ahu_two_stage import (
    RidgeModel,
    TreeEnsemble,
    evaluate_ridge_delta,
    evaluate_tree_delta,
    format_metrics,
    metrics,
    predict_ridge_delta,
    predict_tree_delta,
    train_ridge,
    train_tree_ensemble,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AHU v2 sequence forecasting models.")
    parser.add_argument("--split-dir", default="data_preprocessing/splits")
    parser.add_argument("--out-dir", default="model_outputs")
    parser.add_argument("--freq-min", type=int, default=5)
    parser.add_argument("--window", type=int, default=12, help="Past steps before t. 12 means [t-12 ... t].")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--trees", type=int, default=24)
    parser.add_argument("--tree-depth", type=int, default=9)
    parser.add_argument("--tree-sample", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=84)
    return parser.parse_args()


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


def feature_names(inputs: list[str], window: int) -> list[str]:
    names: list[str] = []
    for offset in range(window, -1, -1):
        suffix = "t" if offset == 0 else f"t-{offset}"
        names.extend(f"{col}_{suffix}" for col in inputs)
    for offset in range(window, 0, -1):
        left = "t" if offset - 1 == 0 else f"t-{offset - 1}"
        right = f"t-{offset}"
        names.extend(f"delta_{col}_{left}_minus_{right}" for col in inputs)
    names.extend(["hour_sin", "hour_cos", "year_sin", "year_cos"])
    return names


def build_sequence_features(
    rows: list[dict[str, str]],
    idx: int,
    inputs: list[str],
    window: int,
    timestamp: datetime,
) -> list[float]:
    values_by_col: dict[str, list[float]] = {col: [] for col in inputs}
    features: list[float] = []

    for source_idx in range(idx - window, idx + 1):
        row = rows[source_idx]
        for col in inputs:
            value = to_float(row, col)
            values_by_col[col].append(value)
            features.append(value)

    for pos in range(1, window + 1):
        for col in inputs:
            features.append(values_by_col[col][pos] - values_by_col[col][pos - 1])

    features.extend(time_features(timestamp))
    return features


def build_dataset(rows: list[dict[str, str]], freq_min: int, window: int) -> SequenceDataset:
    expected_step = timedelta(minutes=freq_min)
    parsed_times = [datetime.fromisoformat(row["timestamp"]) for row in rows]
    timestamps: list[str] = []
    features = {name: [] for name in SUBSYSTEMS}
    current_values = {cfg["target"]: [] for cfg in SUBSYSTEMS.values()}
    targets = {cfg["target"]: [] for cfg in SUBSYSTEMS.values()}
    deltas = {cfg["target"]: [] for cfg in SUBSYSTEMS.values()}
    names = {name: feature_names(cfg["inputs"], window) for name, cfg in SUBSYSTEMS.items()}

    for idx in range(window, len(rows) - 1):
        if any(parsed_times[pos + 1] - parsed_times[pos] != expected_step for pos in range(idx - window, idx + 1)):
            continue

        try:
            subsystem_features = {
                name: build_sequence_features(rows, idx, cfg["inputs"], window, parsed_times[idx])
                for name, cfg in SUBSYSTEMS.items()
            }
            next_row = rows[idx + 1]
            now_row = rows[idx]
            now_targets = {cfg["target"]: to_float(now_row, cfg["target"]) for cfg in SUBSYSTEMS.values()}
            next_targets = {cfg["target"]: to_float(next_row, cfg["target"]) for cfg in SUBSYSTEMS.values()}
        except (KeyError, ValueError):
            continue

        timestamps.append(next_row["timestamp"])
        for name, row_features in subsystem_features.items():
            features[name].append(row_features)
        for target in current_values:
            current_values[target].append(now_targets[target])
            targets[target].append(next_targets[target])
            deltas[target].append(next_targets[target] - now_targets[target])

    return SequenceDataset(timestamps, features, current_values, targets, deltas, names)


def evaluate_persistence(dataset: SequenceDataset, target: str) -> dict[str, float]:
    return metrics(dataset.targets[target], dataset.current_values[target])


def write_predictions(
    path: Path,
    test: SequenceDataset,
    models: dict[str, dict[str, object]],
) -> list[dict[str, str]]:
    columns = ["timestamp"]
    for cfg in SUBSYSTEMS.values():
        target = cfg["target"]
        columns.extend([f"{target}_true", f"{target}_sequence_linear", f"{target}_sequence_tree"])

    out_rows: list[dict[str, str]] = []
    for idx, timestamp in enumerate(test.timestamps):
        row = {"timestamp": timestamp}
        for subsystem, cfg in SUBSYSTEMS.items():
            target = cfg["target"]
            current = test.current_values[target][idx]
            features = test.features[subsystem][idx]
            ridge = models[subsystem]["ridge"]
            tree = models[subsystem]["tree"]
            row[f"{target}_true"] = f"{test.targets[target][idx]:.8g}"
            row[f"{target}_sequence_linear"] = f"{predict_ridge_delta(ridge, current, features):.8g}"  # type: ignore[arg-type]
            row[f"{target}_sequence_tree"] = f"{predict_tree_delta(tree, current, features):.8g}"  # type: ignore[arg-type]
        out_rows.append(row)

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(out_rows)
    return out_rows


def compute_metrics(rows: list[dict[str, str]], target: str, series: str) -> tuple[float, float, float]:
    errors = [float(row[f"{target}_{series}"]) - float(row[f"{target}_true"]) for row in rows]
    mae = sum(abs(err) for err in errors) / len(errors)
    rmse = math.sqrt(sum(err * err for err in errors) / len(errors))
    bias = sum(errors) / len(errors)
    return mae, rmse, bias


def render_html(path: Path, rows: list[dict[str, str]], window: int) -> str:
    plot_data = {
        "timestamps": [row["timestamp"] for row in rows],
        "targets": {
            target: {
                "values": {
                    "true": [float(row[f"{target}_true"]) for row in rows],
                    "sequence_linear": [float(row[f"{target}_sequence_linear"]) for row in rows],
                    "sequence_tree": [float(row[f"{target}_sequence_tree"]) for row in rows],
                },
                "errors": {
                    "sequence_linear": [
                        float(row[f"{target}_sequence_linear"]) - float(row[f"{target}_true"]) for row in rows
                    ],
                    "sequence_tree": [
                        float(row[f"{target}_sequence_tree"]) - float(row[f"{target}_true"]) for row in rows
                    ],
                },
            }
            for target in ["T_rec", "T_sup"]
        },
        "valueSeries": ["true", "sequence_linear", "sequence_tree"],
        "errorSeries": ["sequence_linear", "sequence_tree"],
        "colors": {"true": "#111827", "sequence_linear": "#2563eb", "sequence_tree": "#dc2626"},
        "labels": {"true": "True", "sequence_linear": "Sequence linear", "sequence_tree": "Sequence tree"},
        "expectedStepMinutes": 5,
    }
    data_json = json.dumps(plot_data, separators=(",", ":")).replace("</", "<\\/")
    metric_rows = []
    for target in ["T_rec", "T_sup"]:
        for series, label in [("sequence_linear", "Sequence linear"), ("sequence_tree", "Sequence tree")]:
            mae, rmse, bias = compute_metrics(rows, target, series)
            metric_rows.append(
                f"<tr><td>{target}</td><td>{label}</td><td>{mae:.4f}</td><td>{rmse:.4f}</td><td>{bias:.4f}</td></tr>"
            )

    sections = []
    for target in ["T_rec", "T_sup"]:
        for kind, title in [("values", "value prediction"), ("errors", "prediction error")]:
            key = f"{target}_{kind}"
            sections.append(
                f"""
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
                </section>
                """
            )

    script = canvas_script(data_json)
    first = html.escape(rows[0]["timestamp"])
    last = html.escape(rows[-1]["timestamp"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AHU Two-Stage V2 Sequence 5-Minute Prediction Results</title>
  <style>
    :root {{ font-family: Arial, Helvetica, sans-serif; background: #f3f4f6; color: #111827; }}
    body {{ margin: 0; padding: 28px; }}
    main {{ max-width: 1120px; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .meta, .note {{ color: #4b5563; line-height: 1.5; font-size: 14px; }}
    .note {{ background: #fff; border: 1px solid #d1d5db; border-radius: 8px; padding: 12px 14px; margin: 0 0 14px; }}
    .legend {{ display: flex; gap: 18px; margin: 0 0 18px; flex-wrap: wrap; font-size: 14px; }}
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
    <h1>AHU Two-Stage V2 Sequence 5-Minute Prediction Results</h1>
    <p class="meta">Source: {html.escape(str(path))}<br />Time range: {first} to {last}. Rows: {len(rows)}. Window: [t-{window} ... t].</p>
    <div class="legend">
      <span class="legend-item"><span class="swatch" style="background:#111827"></span>True</span>
      <span class="legend-item"><span class="swatch" style="background:#2563eb"></span>Sequence linear</span>
      <span class="legend-item"><span class="swatch" style="background:#dc2626"></span>Sequence tree</span>
    </div>
    <section class="note">V2 uses subsystem-specific sequence windows and predicts delta = X(t+1)-X(t), then reconstructs X_pred(t+1)=X(t)+delta_pred.</section>
    <table><thead><tr><th>Target</th><th>Model</th><th>MAE</th><th>RMSE</th><th>Bias</th></tr></thead><tbody>{''.join(metric_rows)}</tbody></table>
    {''.join(sections)}
  </main>
  {script}
</body>
</html>
"""


def canvas_script(data_json: str) -> str:
    return f"""
  <script>
    const AHU_DATA = {data_json};
    const charts = new Map();
    function clamp(value, min, max) {{ return Math.max(min, Math.min(max, value)); }}
    function formatTimestamp(text) {{ return text.slice(0, 16); }}
    function minutesBetween(left, right) {{ return (Date.parse(right.replace(" ", "T")) - Date.parse(left.replace(" ", "T"))) / 60000; }}
    function isContinuous(prevIndex, index) {{
      if (prevIndex === null || index <= 0) return true;
      return Math.abs(minutesBetween(AHU_DATA.timestamps[prevIndex], AHU_DATA.timestamps[index]) - AHU_DATA.expectedStepMinutes) < 1e-6;
    }}
    function makeChart(canvas) {{
      const key = canvas.dataset.chart;
      const n = AHU_DATA.timestamps.length;
      return {{key, canvas, ctx: canvas.getContext("2d"), target: canvas.dataset.targetName, kind: canvas.dataset.kind, start: 0, end: n - 1, hoverIndex: null, dragging: false, dragStartX: 0, dragStartStart: 0, dragStartEnd: n - 1, cssWidth: 0, cssHeight: 0}};
    }}
    function seriesList(chart) {{ return chart.kind === "values" ? AHU_DATA.valueSeries : AHU_DATA.errorSeries; }}
    function seriesValues(chart) {{ return AHU_DATA.targets[chart.target][chart.kind]; }}
    function chartArea(chart) {{ return {{left: 66, top: 22, width: chart.cssWidth - 90, height: chart.cssHeight - 74}}; }}
    function visibleBounds(chart) {{
      const values = seriesValues(chart);
      const left = Math.max(0, Math.floor(chart.start));
      const right = Math.min(AHU_DATA.timestamps.length - 1, Math.ceil(chart.end));
      let min = chart.kind === "errors" ? 0 : Infinity;
      let max = chart.kind === "errors" ? 0 : -Infinity;
      for (const series of seriesList(chart)) {{
        const arr = values[series];
        for (let i = left; i <= right; i++) {{
          const v = arr[i];
          if (v < min) min = v;
          if (v > max) max = v;
        }}
      }}
      const margin = (max - min) * 0.10 || 0.1;
      return {{min: min - margin, max: max + margin}};
    }}
    function resizeCanvas(chart) {{
      const rect = chart.canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(320, Math.floor(rect.width));
      const height = Number(chart.canvas.getAttribute("height")) || 360;
      chart.canvas.width = Math.floor(width * dpr);
      chart.canvas.height = Math.floor(height * dpr);
      chart.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      chart.cssWidth = width;
      chart.cssHeight = height;
    }}
    function xForIndex(chart, index, area) {{ return area.left + ((index - chart.start) / (chart.end - chart.start || 1)) * area.width; }}
    function indexForX(chart, x, area) {{ return chart.start + ((x - area.left) / area.width) * (chart.end - chart.start); }}
    function yForValue(value, area, bounds) {{ return area.top + area.height - ((value - bounds.min) / (bounds.max - bounds.min || 1)) * area.height; }}
    function drawLine(ctx, chart, arr, area, bounds, color, width, dash) {{
      ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []); ctx.beginPath();
      const left = Math.max(0, Math.floor(chart.start));
      const right = Math.min(AHU_DATA.timestamps.length - 1, Math.ceil(chart.end));
      let started = false; let prevIndex = null;
      for (let i = left; i <= right; i++) {{
        const x = xForIndex(chart, i, area); const y = yForValue(arr[i], area, bounds);
        if (!started || !isContinuous(prevIndex, i)) {{ ctx.moveTo(x, y); started = true; }} else {{ ctx.lineTo(x, y); }}
        prevIndex = i;
      }}
      ctx.stroke(); ctx.restore();
    }}
    function drawChart(chart) {{
      const ctx = chart.ctx; const width = chart.cssWidth; const height = chart.cssHeight; if (!width || !height) return;
      const area = chartArea(chart); const bounds = visibleBounds(chart);
      ctx.clearRect(0, 0, width, height); ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "#e5e7eb"; ctx.lineWidth = 1; ctx.fillStyle = "#6b7280"; ctx.font = "12px Arial"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
      for (let tick = 0; tick <= 4; tick++) {{
        const value = bounds.min + (bounds.max - bounds.min) * tick / 4; const y = yForValue(value, area, bounds);
        ctx.beginPath(); ctx.moveTo(area.left, y); ctx.lineTo(area.left + area.width, y); ctx.stroke(); ctx.fillText(value.toFixed(chart.kind === "errors" ? 3 : 2), area.left - 8, y);
      }}
      if (chart.kind === "errors") {{
        const zeroY = yForValue(0, area, bounds); ctx.save(); ctx.strokeStyle = "#111827"; ctx.setLineDash([5, 5]); ctx.beginPath(); ctx.moveTo(area.left, zeroY); ctx.lineTo(area.left + area.width, zeroY); ctx.stroke(); ctx.restore();
      }}
      const values = seriesValues(chart);
      for (const series of seriesList(chart)) drawLine(ctx, chart, values[series], area, bounds, AHU_DATA.colors[series], series === "true" ? 1.7 : 1.4, []);
      ctx.strokeStyle = "#9ca3af"; ctx.strokeRect(area.left, area.top, area.width, area.height);
      const leftIndex = Math.max(0, Math.round(chart.start)); const rightIndex = Math.min(AHU_DATA.timestamps.length - 1, Math.round(chart.end));
      ctx.fillStyle = "#6b7280"; ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillText(formatTimestamp(AHU_DATA.timestamps[leftIndex]), area.left, area.top + area.height + 12);
      ctx.textAlign = "right"; ctx.fillText(formatTimestamp(AHU_DATA.timestamps[rightIndex]), area.left + area.width, area.top + area.height + 12);
      if (chart.hoverIndex !== null) {{
        const i = clamp(chart.hoverIndex, leftIndex, rightIndex); const x = xForIndex(chart, i, area);
        ctx.strokeStyle = "#374151"; ctx.beginPath(); ctx.moveTo(x, area.top); ctx.lineTo(x, area.top + area.height); ctx.stroke();
        const parts = [formatTimestamp(AHU_DATA.timestamps[i])];
        for (const series of seriesList(chart)) parts.push(`${{AHU_DATA.labels[series]}}=${{values[series][i].toFixed(chart.kind === "errors" ? 4 : 3)}}`);
        const readout = document.querySelector(`[data-readout="${{chart.key}}"]`); if (readout) readout.textContent = parts.join(" | ");
      }}
      const label = document.querySelector(`[data-window="${{chart.key}}"]`); if (label) label.textContent = `${{formatTimestamp(AHU_DATA.timestamps[leftIndex])}} to ${{formatTimestamp(AHU_DATA.timestamps[rightIndex])}} (${{rightIndex - leftIndex + 1}} points)`;
    }}
    function zoom(chart, centerIndex, factor) {{
      const n = AHU_DATA.timestamps.length; const oldSpan = chart.end - chart.start; const newSpan = clamp(oldSpan * factor, 20, n - 1);
      let start = centerIndex - newSpan / 2; let end = centerIndex + newSpan / 2;
      if (start < 0) {{ end -= start; start = 0; }} if (end > n - 1) {{ start -= end - (n - 1); end = n - 1; }}
      chart.start = clamp(start, 0, n - 1); chart.end = clamp(end, 0, n - 1); drawChart(chart);
    }}
    function resetChart(chart) {{ chart.start = 0; chart.end = AHU_DATA.timestamps.length - 1; drawChart(chart); }}
    function attachEvents(chart) {{
      chart.canvas.addEventListener("wheel", (event) => {{ event.preventDefault(); const rect = chart.canvas.getBoundingClientRect(); const area = chartArea(chart); const center = clamp(indexForX(chart, event.clientX - rect.left, area), 0, AHU_DATA.timestamps.length - 1); zoom(chart, center, event.deltaY < 0 ? 0.75 : 1.35); }}, {{passive: false}});
      chart.canvas.addEventListener("pointerdown", (event) => {{ chart.dragging = true; chart.dragStartX = event.clientX; chart.dragStartStart = chart.start; chart.dragStartEnd = chart.end; chart.canvas.setPointerCapture(event.pointerId); }});
      chart.canvas.addEventListener("pointermove", (event) => {{
        const rect = chart.canvas.getBoundingClientRect(); const area = chartArea(chart); const x = event.clientX - rect.left;
        if (chart.dragging) {{
          const dx = event.clientX - chart.dragStartX; const span = chart.dragStartEnd - chart.dragStartStart; const shift = -dx / area.width * span;
          chart.start = chart.dragStartStart + shift; chart.end = chart.dragStartEnd + shift;
          if (chart.start < 0) {{ chart.end -= chart.start; chart.start = 0; }} if (chart.end > AHU_DATA.timestamps.length - 1) {{ chart.start -= chart.end - (AHU_DATA.timestamps.length - 1); chart.end = AHU_DATA.timestamps.length - 1; }}
        }}
        const nextHover = clamp(Math.round(indexForX(chart, x, area)), 0, AHU_DATA.timestamps.length - 1);
        if (chart.dragging || nextHover !== chart.hoverIndex) {{ chart.hoverIndex = nextHover; drawChart(chart); }}
      }});
      chart.canvas.addEventListener("pointerup", (event) => {{ chart.dragging = false; if (chart.canvas.hasPointerCapture(event.pointerId)) chart.canvas.releasePointerCapture(event.pointerId); }});
      chart.canvas.addEventListener("pointerleave", () => {{ chart.dragging = false; chart.hoverIndex = null; const readout = document.querySelector(`[data-readout="${{chart.key}}"]`); if (readout) readout.textContent = ""; drawChart(chart); }});
      chart.canvas.addEventListener("dblclick", () => resetChart(chart));
    }}
    function init() {{
      document.querySelectorAll("canvas[data-chart]").forEach((canvas) => {{ const chart = makeChart(canvas); charts.set(chart.key, chart); attachEvents(chart); resizeCanvas(chart); drawChart(chart); }});
      document.querySelectorAll("button[data-action]").forEach((button) => button.addEventListener("click", () => {{ const chart = charts.get(button.dataset.target); if (!chart) return; const center = (chart.start + chart.end) / 2; if (button.dataset.action === "zoom-in") zoom(chart, center, 0.5); if (button.dataset.action === "zoom-out") zoom(chart, center, 2.0); if (button.dataset.action === "reset") resetChart(chart); }}));
      window.addEventListener("resize", () => {{ for (const chart of charts.values()) {{ resizeCanvas(chart); drawChart(chart); }} }});
    }}
    init();
  </script>
"""


def main() -> None:
    args = parse_args()
    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = build_dataset(load_rows(split_dir / "train_5min.csv"), args.freq_min, args.window)
    validation = build_dataset(load_rows(split_dir / "validation_5min.csv"), args.freq_min, args.window)
    test = build_dataset(load_rows(split_dir / "test_5min.csv"), args.freq_min, args.window)
    datasets = {"train": train, "validation": validation, "test": test}

    report = [
        "AHU two-stage v2 sequence model report",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Split directory: {split_dir.resolve()}",
        "Prediction horizon: 5 min",
        f"Sequence window: [t-{args.window} ... t]",
        "Target: delta = X(t+1)-X(t), reconstructed as X_pred(t+1)=X(t)+delta_pred.",
        "Subsystem inputs:",
        "  heat_exchanger: T_out, T_ret, T_rec, pressure_hex",
        "  heating_coil: T_rec, T_coil, T_sup",
        "",
    ]

    models: dict[str, dict[str, object]] = {}
    for subsystem, cfg in SUBSYSTEMS.items():
        target = cfg["target"]
        ridge = train_ridge(train.features[subsystem], train.deltas[target], args.ridge_alpha)
        tree = train_tree_ensemble(
            train.features[subsystem],
            train.targets[target],
            train.current_values[target],
            target,
            args.trees,
            args.tree_depth,
            args.tree_sample,
            args.seed + len(models) * 100,
        )
        models[subsystem] = {"ridge": ridge, "tree": tree}
        report.append(f"Subsystem: {subsystem}")
        report.append(f"Target: {target}(t+1)")
        report.append(f"Feature count: {len(train.feature_names[subsystem])}")
        for split_name, dataset in datasets.items():
            persist = evaluate_persistence(dataset, target)
            ridge_metrics = evaluate_ridge_delta(
                type("DatasetProxy", (), {
                    "targets": {target: dataset.targets[target]},
                    "current_values": {target: dataset.current_values[target]},
                    "features": dataset.features[subsystem],
                })(),
                target,
                ridge,
            )
            tree_metrics = evaluate_tree_delta(
                type("DatasetProxy", (), {
                    "targets": {target: dataset.targets[target]},
                    "current_values": {target: dataset.current_values[target]},
                    "features": dataset.features[subsystem],
                })(),
                target,
                tree,
            )
            report.append(format_metrics(split_name, "persistence", persist))
            report.append(format_metrics(split_name, "seq_linear", ridge_metrics))
            report.append(format_metrics(split_name, "seq_tree", tree_metrics))
        report.append("")
        print(f"{subsystem}: trained v2 sequence target {target}")

    prediction_path = out_dir / "ahu_two_stage_v2_sequence_test_predictions_5min.csv"
    html_path = out_dir / "ahu_two_stage_v2_sequence_predictions_5min.html"
    report_path = out_dir / "ahu_two_stage_v2_sequence_model_report_5min.txt"
    predictions = write_predictions(prediction_path, test, models)
    report_path.write_text("\n".join(report), encoding="utf-8")
    html_path.write_text(render_html(prediction_path, predictions, args.window), encoding="utf-8")

    print(f"Wrote {report_path}")
    print(f"Wrote {prediction_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
