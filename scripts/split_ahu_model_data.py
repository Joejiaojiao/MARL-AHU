"""
Split the AHU model-ready CSV into chronological train/validation/test sets.

The input data is a time series, so rows are split in timestamp order rather
than randomly to avoid future-data leakage.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split AHU model-ready data.")
    parser.add_argument(
        "--input",
        default="data_processed/ahu_model_5min.csv",
        help="Model-ready CSV file to split.",
    )
    parser.add_argument(
        "--out-dir",
        default="data_processed/splits",
        help="Directory for split CSV files and report.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Training set ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation set ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test set ratio.")
    return parser.parse_args()


def parse_timestamp(row: dict[str, str]) -> datetime:
    return datetime.fromisoformat(row["timestamp"])


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row.")
        rows = list(reader)
    rows.sort(key=parse_timestamp)
    return reader.fieldnames, rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_rows(
    rows: list[dict[str, str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0; got {ratio_sum:.6f}.")

    total = len(rows)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return rows[:train_end], rows[train_end:val_end], rows[val_end:]


def missing_count(fieldnames: list[str], rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows for field in fieldnames if row.get(field, "") == "")


def describe_split(name: str, fieldnames: list[str], rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return [
            f"{name}:",
            "  rows: 0",
            "  start: n/a",
            "  end: n/a",
            "  missing_values: 0",
        ]

    return [
        f"{name}:",
        f"  rows: {len(rows)}",
        f"  start: {rows[0]['timestamp']}",
        f"  end: {rows[-1]['timestamp']}",
        f"  missing_values: {missing_count(fieldnames, rows)}",
    ]


def write_report(
    path: Path,
    input_path: Path,
    fieldnames: list[str],
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    ratios: tuple[float, float, float],
) -> None:
    total = len(train_rows) + len(val_rows) + len(test_rows)
    lines = [
        "AHU model-ready data split report",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Input: {input_path}",
        "Method: chronological split by timestamp",
        f"Ratios: train={ratios[0]:.2f}, validation={ratios[1]:.2f}, test={ratios[2]:.2f}",
        f"Total rows: {total}",
        f"Columns: {', '.join(fieldnames)}",
        "",
        *describe_split("train", fieldnames, train_rows),
        "",
        *describe_split("validation", fieldnames, val_rows),
        "",
        *describe_split("test", fieldnames, test_rows),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames, rows = read_rows(input_path)
    train_rows, val_rows, test_rows = split_rows(
        rows,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )

    write_csv(out_dir / "train_5min.csv", fieldnames, train_rows)
    write_csv(out_dir / "validation_5min.csv", fieldnames, val_rows)
    write_csv(out_dir / "test_5min.csv", fieldnames, test_rows)
    write_report(
        out_dir / "split_report_5min.txt",
        input_path,
        fieldnames,
        train_rows,
        val_rows,
        test_rows,
        (args.train_ratio, args.val_ratio, args.test_ratio),
    )


if __name__ == "__main__":
    main()
