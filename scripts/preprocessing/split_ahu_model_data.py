"""
Split the AHU model-ready CSV into train and validation sets.

Splitting strategy: seasonal interleaving by quarter block.

Each calendar quarter is divided as follows:
  - Months 1 and 2 of the quarter -> train
  - Month 3 of the quarter        -> validation

Applied to every complete year in the dataset (2022 and 2023).
Any data outside those years (e.g. late 2021) is assigned to train.

This ensures both train and validation cover all four seasons,
while preserving temporal ordering within each split (no future leakage).
The two splits are kept as separate files; rows within each file remain
in chronological order so sequence models can detect gaps normally.

Why no test set:
  During model development the validation set is used for hyperparameter
  tuning.  A held-out test set should only be evaluated once at the end;
  it is not created here to avoid accidental over-fitting to test data.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# Within each quarter the last month goes to validation, the first two to train.
QUARTER_ASSIGNMENT: dict[int, tuple[list[int], int]] = {
    1: ([1, 2], 3),
    2: ([4, 5], 6),
    3: ([7, 8], 9),
    4: ([10, 11], 12),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split AHU model-ready data.")
    parser.add_argument(
        "--input",
        default="data_preprocessing/preprocess/ahu_model_5min.csv",
        help="Model-ready CSV file to split.",
    )
    parser.add_argument(
        "--out-dir",
        default="data_preprocessing/splits",
        help="Directory for split CSV files and report.",
    )
    return parser.parse_args()


def assign_split(ts: datetime) -> str:
    """Return 'train' or 'validation' for a given timestamp."""
    year, month = ts.year, ts.month
    if year not in (2022, 2023):
        return "train"
    for _q, (train_months, val_month) in QUARTER_ASSIGNMENT.items():
        if month == val_month:
            return "validation"
        if month in train_months:
            return "train"
    return "train"


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row.")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    rows.sort(key=lambda r: r["timestamp"])
    return fieldnames, rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def season_label(month: int) -> str:
    return {12: "Win", 1: "Win", 2: "Win",
            3: "Spr", 4: "Spr", 5: "Spr",
            6: "Sum", 7: "Sum", 8: "Sum",
            9: "Aut", 10: "Aut", 11: "Aut"}[month]


def describe_split(
    name: str,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> list[str]:
    if not rows:
        return [f"{name}: (empty)"]
    missing = sum(1 for row in rows for f in fieldnames if row.get(f, "") == "")
    by_month: dict[str, int] = defaultdict(int)
    for row in rows:
        ts = datetime.fromisoformat(row["timestamp"])
        by_month[f"{ts.year}-{ts.month:02d}({season_label(ts.month)})"] += 1
    month_summary = ", ".join(f"{k}:{v}" for k, v in sorted(by_month.items()))
    return [
        f"{name}:",
        f"  rows:           {len(rows)}",
        f"  start:          {rows[0]['timestamp']}",
        f"  end:            {rows[-1]['timestamp']}",
        f"  missing_values: {missing}",
        f"  months:         {month_summary}",
    ]


def write_report(
    path: Path,
    input_path: Path,
    fieldnames: list[str],
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
) -> None:
    total = len(train_rows) + len(val_rows)
    lines = [
        "AHU model-ready data split report",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Input: {input_path}",
        "Method: seasonal-interleaved quarter split (train + validation only)",
        "  Per quarter in 2022-2023: months 1-2 -> train, month 3 -> validation",
        "  Data outside 2022-2023 -> train",
        "  No test set: held-out evaluation deferred to project end.",
        f"Total rows: {total}",
        f"Train:      {len(train_rows)} ({len(train_rows)/total:.1%})",
        f"Validation: {len(val_rows)} ({len(val_rows)/total:.1%})",
        f"Columns: {', '.join(fieldnames)}",
        "",
        *describe_split("train", fieldnames, train_rows),
        "",
        *describe_split("validation", fieldnames, val_rows),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames, rows = read_rows(input_path)

    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    for row in rows:
        ts = datetime.fromisoformat(row["timestamp"])
        if assign_split(ts) == "validation":
            val_rows.append(row)
        else:
            train_rows.append(row)

    write_csv(out_dir / "train_5min.csv", fieldnames, train_rows)
    write_csv(out_dir / "validation_5min.csv", fieldnames, val_rows)
    write_report(out_dir / "split_report_5min.txt", input_path, fieldnames, train_rows, val_rows)

    print(f"Train:      {len(train_rows)} rows")
    print(f"Validation: {len(val_rows)} rows")
    print(f"Wrote {out_dir}/train_5min.csv")
    print(f"Wrote {out_dir}/validation_5min.csv")
    print(f"Wrote {out_dir}/split_report_5min.txt")


if __name__ == "__main__":
    main()
