from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


NON_GRAND_SLAM_CATEGORIES = {"masters_1000", "atp_500", "atp_250"}
DEFAULT_TEST_TOURNAMENTS = {"Barcelona", "Madrid", "Munich"}


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
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_completed_bo3(row: dict) -> bool:
    if row.get("target_gana_jugador_1") not in {"0", "1", 0, 1}:
        return False
    if row.get("categoria_torneo") not in NON_GRAND_SLAM_CATEGORIES:
        return False
    try:
        sets1 = int(float(str(row.get("sets_jugador_1", ""))))
        sets2 = int(float(str(row.get("sets_jugador_2", ""))))
    except ValueError:
        return False
    return max(sets1, sets2) == 2 and min(sets1, sets2) in {0, 1}


def should_swap(row: dict) -> bool:
    key = row.get("match_id") or f"{row.get('fecha')}|{row.get('torneo')}|{row.get('jugador_1')}|{row.get('jugador_2')}"
    digest = hashlib.md5(str(key).encode("utf-8")).hexdigest()
    return int(digest[-1], 16) % 2 == 1


def swap_completed_row(row: dict) -> dict:
    swapped = dict(row)
    for key in list(row.keys()):
        if key.startswith("jugador_1_"):
            swapped[key] = row.get(key.replace("jugador_1_", "jugador_2_", 1), "")
        elif key.startswith("jugador_2_"):
            swapped[key] = row.get(key.replace("jugador_2_", "jugador_1_", 1), "")
    simple_pairs = [
        ("jugador_1", "jugador_2"),
        ("sembrado_jugador_1", "sembrado_jugador_2"),
        ("jugador_1_tiene_sembrado", "jugador_2_tiene_sembrado"),
        ("sets_jugador_1", "sets_jugador_2"),
        ("games_jugador_1", "games_jugador_2"),
        ("victorias_previas_jugador_1_vs_jugador_2", "victorias_previas_jugador_2_vs_jugador_1"),
    ]
    for left, right in simple_pairs:
        swapped[left] = row.get(right, "")
        swapped[right] = row.get(left, "")
    for key, value in row.items():
        if key.startswith("diferencia_"):
            try:
                swapped[key] = -float(value)
            except (TypeError, ValueError):
                swapped[key] = value
    if row.get("target_gana_jugador_1") in {"0", "1", 0, 1}:
        swapped["target_gana_jugador_1"] = 1 - int(row["target_gana_jugador_1"])
    return swapped


def orient_completed_rows(rows: list[dict]) -> list[dict]:
    return [swap_completed_row(row) if should_swap(row) else row for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare non-Grand-Slam ATP BO3 train/test files.")
    parser.add_argument("--model-dataset", type=Path, default=Path("files/processed/model_dataset_2025_2026/model_dataset.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("files/processed/non_grand_slam_bo3/model_dataset"))
    parser.add_argument("--train-before", default="2026-04-13")
    parser.add_argument("--test-from", default="2026-04-13")
    parser.add_argument("--test-tournaments", nargs="*", default=sorted(DEFAULT_TEST_TOURNAMENTS))
    args = parser.parse_args()

    rows = [row for row in read_csv(args.model_dataset) if is_completed_bo3(row)]
    test_tournaments = set(args.test_tournaments)
    train_rows = [
        row
        for row in rows
        if row.get("fecha", "") < args.train_before
        and not (row.get("torneo") in test_tournaments and row.get("fecha", "") >= args.test_from)
    ]
    test_rows = [
        row
        for row in rows
        if row.get("torneo") in test_tournaments and row.get("fecha", "") >= args.test_from
    ]

    train_rows = orient_completed_rows(train_rows)
    test_rows = orient_completed_rows(test_rows)
    write_csv(args.out_dir / "bo3_matches.csv", orient_completed_rows(rows))
    write_csv(args.out_dir / "train.csv", train_rows)
    write_csv(args.out_dir / "test.csv", test_rows)
    summary = {
        "bo3_match_rows": len(rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "categories": sorted(NON_GRAND_SLAM_CATEGORIES),
        "train_before": args.train_before,
        "test_from": args.test_from,
        "test_tournaments": sorted(test_tournaments),
        "target": "moneyline_and_bo3_set_markets",
    }
    (args.out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
