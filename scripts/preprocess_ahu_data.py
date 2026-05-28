"""
Preprocess KTH Live-in Lab AHU data exported as separate Excel files.

The script keeps the raw Excel files untouched and creates model-ready CSV files:

  data_processed/ahu_clean_5min.csv
  data_processed/ahu_clean_5min_running.csv
  data_processed/ahu_model_5min.csv
  data_processed/preprocess_report_5min.txt

It intentionally uses only the Python standard library because the project
environment may not have pandas/openpyxl installed.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET


SPREADSHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


@dataclass(frozen=True)
class SignalSpec:
    file_name: str
    output_name: str
    kind: str  # "measurement" or "control"
    zero_is_missing: bool = False


SIGNALS: tuple[SignalSpec, ...] = (
    SignalSpec("Outdoor air intake temperature.xlsx", "T_out", "measurement", True),
    SignalSpec("Extract air temperature.xlsx", "T_ret", "measurement", True),
    SignalSpec("Temperature after heat recovery.xlsx", "T_rec", "measurement", True),
    SignalSpec("Heating coil temperature.xlsx", "T_coil", "measurement", True),
    SignalSpec("Supply air temperature.xlsx", "T_sup", "measurement", True),
    SignalSpec("Pressure across the heat exchanger.xlsx", "pressure_hex", "measurement", False),
    SignalSpec("Heating actuator control signal.xlsx", "u_heat", "control", False),
    SignalSpec("Heat recovery control signal.xlsx", "u_heat_recovery", "control", False),
    SignalSpec("Control signal FF1.xlsx", "u_FF1", "control", False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess AHU Excel exports.")
    parser.add_argument("--raw-dir", default="AHU data", help="Directory containing raw Excel files.")
    parser.add_argument("--out-dir", default="data_processed", help="Directory for processed CSV files.")
    parser.add_argument("--freq-min", type=int, default=5, help="Resampling interval in minutes.")
    parser.add_argument(
        "--max-interp-gap-min",
        type=int,
        default=60,
        help="Maximum gap for linear interpolation of measurement signals.",
    )
    parser.add_argument(
        "--running-pressure-threshold",
        type=float,
        default=5.0,
        help="pressure_hex threshold used in the is_running flag.",
    )
    parser.add_argument(
        "--running-ff1-threshold",
        type=float,
        default=5.0,
        help="u_FF1 threshold used in the is_running flag.",
    )
    return parser.parse_args()


def read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []

    strings: list[str] = []
    with zf.open("xl/sharedStrings.xml") as fh:
        for _, elem in ET.iterparse(fh, events=("end",)):
            if elem.tag == SPREADSHEET_NS + "si":
                strings.append("".join(t.text or "" for t in elem.iter(SPREADSHEET_NS + "t")))
                elem.clear()
    return strings


def column_from_cell_ref(cell_ref: str) -> str:
    match = re.match(r"([A-Z]+)", cell_ref or "")
    return match.group(1) if match else ""


def cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    value = cell.find(SPREADSHEET_NS + "v")
    if value is None or value.text is None:
        return ""

    text = value.text
    if cell.get("t") == "s":
        idx = int(text)
        return shared_strings[idx] if 0 <= idx < len(shared_strings) else text
    return text


def parse_timestamp(text: str) -> datetime | None:
    text = text.strip()
    if not text:
        return None

    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass

    # Excel serial date fallback. Excel incorrectly treats 1900 as a leap year;
    # using 1899-12-30 matches common spreadsheet-library behavior.
    try:
        serial = float(text)
    except ValueError:
        return None
    return datetime(1899, 12, 30) + timedelta(days=serial)


def iter_xlsx_rows(path: Path) -> Iterable[tuple[datetime, float]]:
    with zipfile.ZipFile(path) as zf:
        shared_strings = read_shared_strings(zf)
        with zf.open("xl/worksheets/sheet1.xml") as fh:
            for _, row in ET.iterparse(fh, events=("end",)):
                if row.tag != SPREADSHEET_NS + "row":
                    continue

                row_num = int(row.get("r", "0"))
                if row_num < 2:
                    row.clear()
                    continue

                cells = {
                    column_from_cell_ref(cell.get("r", "")): cell_value(cell, shared_strings)
                    for cell in row.findall(SPREADSHEET_NS + "c")
                    if cell.get("r")
                }
                row.clear()

                timestamp = parse_timestamp(cells.get("A", ""))
                raw_value = cells.get("B", "").strip()
                if timestamp is None or raw_value == "":
                    continue

                try:
                    value = float(raw_value)
                except ValueError:
                    continue

                if math.isfinite(value):
                    yield timestamp, value


def floor_to_bin(timestamp: datetime, freq_min: int) -> datetime:
    discard = timedelta(
        minutes=timestamp.minute % freq_min,
        seconds=timestamp.second,
        microseconds=timestamp.microsecond,
    )
    return timestamp - discard


def aggregate_signal(
    path: Path, spec: SignalSpec, freq_min: int
) -> tuple[dict[datetime, float], dict[str, object]]:
    measurement_sums: dict[datetime, float] = {}
    measurement_counts: dict[datetime, int] = {}
    control_latest: dict[datetime, tuple[datetime, float]] = {}

    raw_count = 0
    used_count = 0
    zero_as_missing_count = 0
    min_time: datetime | None = None
    max_time: datetime | None = None
    min_value: float | None = None
    max_value: float | None = None

    for timestamp, value in iter_xlsx_rows(path):
        raw_count += 1
        min_time = timestamp if min_time is None or timestamp < min_time else min_time
        max_time = timestamp if max_time is None or timestamp > max_time else max_time
        min_value = value if min_value is None or value < min_value else min_value
        max_value = value if max_value is None or value > max_value else max_value

        if spec.zero_is_missing and value == 0:
            zero_as_missing_count += 1
            continue

        bin_time = floor_to_bin(timestamp, freq_min)
        used_count += 1

        if spec.kind == "measurement":
            measurement_sums[bin_time] = measurement_sums.get(bin_time, 0.0) + value
            measurement_counts[bin_time] = measurement_counts.get(bin_time, 0) + 1
        else:
            previous = control_latest.get(bin_time)
            if previous is None or timestamp > previous[0]:
                control_latest[bin_time] = (timestamp, value)

    if spec.kind == "measurement":
        aggregated = {
            bin_time: measurement_sums[bin_time] / measurement_counts[bin_time]
            for bin_time in measurement_sums
        }
    else:
        aggregated = {bin_time: value for bin_time, (_, value) in control_latest.items()}

    stats: dict[str, object] = {
        "raw_count": raw_count,
        "used_count": used_count,
        "bin_count": len(aggregated),
        "zero_as_missing_count": zero_as_missing_count,
        "min_time": min_time,
        "max_time": max_time,
        "min_value": min_value,
        "max_value": max_value,
    }
    return aggregated, stats


def make_timeline(start: datetime, end: datetime, freq_min: int) -> list[datetime]:
    timeline: list[datetime] = []
    current = floor_to_bin(start, freq_min)
    last = floor_to_bin(end, freq_min)
    step = timedelta(minutes=freq_min)
    while current <= last:
        timeline.append(current)
        current += step
    return timeline


def fill_control(values: list[float | None]) -> list[float | None]:
    filled: list[float | None] = []
    last: float | None = None
    for value in values:
        if value is not None:
            last = value
        filled.append(last)
    return filled


def interpolate_measurement(
    values: list[float | None], max_gap_steps: int
) -> list[float | None]:
    filled = values[:]
    known_indexes = [idx for idx, value in enumerate(values) if value is not None]
    if len(known_indexes) < 2:
        return filled

    for left, right in zip(known_indexes, known_indexes[1:]):
        gap = right - left
        if gap <= 1 or gap - 1 > max_gap_steps:
            continue

        left_value = values[left]
        right_value = values[right]
        if left_value is None or right_value is None:
            continue

        for idx in range(left + 1, right):
            ratio = (idx - left) / gap
            filled[idx] = left_value + (right_value - left_value) * ratio

    return filled


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def write_csv(path: Path, rows: list[dict[str, float | None | int]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            output = {}
            for col in columns:
                value = row.get(col)
                if col == "timestamp":
                    output[col] = value
                elif col == "is_running":
                    output[col] = int(value or 0)
                else:
                    output[col] = format_float(value if isinstance(value, float) else None)
            writer.writerow(output)


def has_all_model_inputs(row: dict[str, float | None | int]) -> bool:
    return all(row.get(spec.output_name) is not None for spec in SIGNALS)


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregated_by_signal: dict[str, dict[datetime, float]] = {}
    stats_by_signal: dict[str, dict[str, object]] = {}

    for spec in SIGNALS:
        path = raw_dir / spec.file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing required input file: {path}")

        print(f"Reading {spec.file_name} -> {spec.output_name}")
        aggregated, stats = aggregate_signal(path, spec, args.freq_min)
        aggregated_by_signal[spec.output_name] = aggregated
        stats_by_signal[spec.output_name] = stats

    common_start = max(
        stats["min_time"]
        for stats in stats_by_signal.values()
        if isinstance(stats["min_time"], datetime)
    )
    common_end = min(
        stats["max_time"]
        for stats in stats_by_signal.values()
        if isinstance(stats["max_time"], datetime)
    )
    timeline = make_timeline(common_start, common_end, args.freq_min)

    max_gap_steps = max(0, args.max_interp_gap_min // args.freq_min)
    columns = ["timestamp"] + [spec.output_name for spec in SIGNALS] + ["is_running"]

    filled_by_signal: dict[str, list[float | None]] = {}
    for spec in SIGNALS:
        series = aggregated_by_signal[spec.output_name]
        values = [series.get(timestamp) for timestamp in timeline]
        if spec.kind == "control":
            filled = fill_control(values)
        else:
            filled = interpolate_measurement(values, max_gap_steps)
        filled_by_signal[spec.output_name] = filled

    rows: list[dict[str, float | None | int]] = []
    for idx, timestamp in enumerate(timeline):
        row: dict[str, float | None | int] = {"timestamp": timestamp.isoformat(sep=" ")}
        for spec in SIGNALS:
            row[spec.output_name] = filled_by_signal[spec.output_name][idx]

        pressure = row.get("pressure_hex")
        ff1 = row.get("u_FF1")
        row["is_running"] = int(
            (isinstance(pressure, float) and pressure > args.running_pressure_threshold)
            or (isinstance(ff1, float) and ff1 > args.running_ff1_threshold)
        )
        rows.append(row)

    clean_path = out_dir / f"ahu_clean_{args.freq_min}min.csv"
    running_path = out_dir / f"ahu_clean_{args.freq_min}min_running.csv"
    model_path = out_dir / f"ahu_model_{args.freq_min}min.csv"
    report_path = out_dir / f"preprocess_report_{args.freq_min}min.txt"

    running_rows = [row for row in rows if row["is_running"]]
    model_rows = [row for row in running_rows if has_all_model_inputs(row)]

    write_csv(clean_path, rows, columns)
    write_csv(running_path, running_rows, columns)
    write_csv(model_path, model_rows, columns)

    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("AHU preprocessing report\n")
        fh.write(f"Generated at: {datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(f"Raw directory: {raw_dir.resolve()}\n")
        fh.write(f"Output directory: {out_dir.resolve()}\n")
        fh.write(f"Frequency: {args.freq_min} min\n")
        fh.write(f"Measurement interpolation max gap: {args.max_interp_gap_min} min\n")
        fh.write(f"Common time range: {timeline[0]} -> {timeline[-1]}\n")
        fh.write(f"Rows in clean dataset: {len(rows)}\n")
        fh.write(f"Rows in running dataset: {len(running_rows)}\n")
        fh.write(f"Rows in model-ready complete dataset: {len(model_rows)}\n\n")

        for spec in SIGNALS:
            stats = stats_by_signal[spec.output_name]
            missing_after_fill = sum(
                1 for value in filled_by_signal[spec.output_name] if value is None
            )
            fh.write(f"{spec.output_name} ({spec.file_name})\n")
            fh.write(f"  kind: {spec.kind}\n")
            fh.write(f"  raw_count: {stats['raw_count']}\n")
            fh.write(f"  used_count: {stats['used_count']}\n")
            fh.write(f"  bins_before_fill: {stats['bin_count']}\n")
            fh.write(f"  missing_after_fill: {missing_after_fill}\n")
            fh.write(f"  zero_as_missing_count: {stats['zero_as_missing_count']}\n")
            fh.write(f"  source_time_range: {stats['min_time']} -> {stats['max_time']}\n")
            fh.write(f"  source_value_range: {stats['min_value']} -> {stats['max_value']}\n\n")

    print(f"Wrote {clean_path}")
    print(f"Wrote {running_path}")
    print(f"Wrote {model_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
