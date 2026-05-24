from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_null(value: object) -> bool:
    return value is None or str(value).strip() == ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Report null coverage for Grand Slam model dataset columns.")
    parser.add_argument("--dataset", default="files/processed/grand_slam_moneyline/model_dataset/model_dataset.csv")
    parser.add_argument("--out", default="files/processed/grand_slam_moneyline/model_dataset/null_report.csv")
    parser.add_argument("--summary-out", default="files/processed/grand_slam_moneyline/model_dataset/null_summary.json")
    args = parser.parse_args()

    rows = read_csv(Path(args.dataset))
    columns = sorted({key for row in rows for key in row})
    report = []
    for column in columns:
        nulls = sum(1 for row in rows if is_null(row.get(column)))
        report.append(
            {
                "column": column,
                "rows": len(rows),
                "nulls": nulls,
                "null_pct": nulls / len(rows) if rows else 0,
                "is_sofascore": "True" if str(column).find("sofascore") >= 0 else "False",
            }
        )
    report.sort(key=lambda row: (row["is_sofascore"] != "True", -float(row["null_pct"]), row["column"]))
    write_csv(Path(args.out), report)

    sofascore_rows = [row for row in report if row["is_sofascore"] == "True"]
    summary = {
        "rows": len(rows),
        "columns": len(columns),
        "sofascore_columns": len(sofascore_rows),
        "sofascore_avg_null_pct": (
            sum(float(row["null_pct"]) for row in sofascore_rows) / len(sofascore_rows)
            if sofascore_rows
            else None
        ),
        "sofascore_min_null_pct": min((float(row["null_pct"]) for row in sofascore_rows), default=None),
        "sofascore_max_null_pct": max((float(row["null_pct"]) for row in sofascore_rows), default=None),
    }
    Path(args.summary_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
