from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


GRAND_SLAMS = {"Australian Open", "French Open", "Wimbledon", "US Open"}


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


def dedupe(rows: list[dict]) -> list[dict]:
    seen: set[tuple[str, ...]] = set()
    output: list[dict] = []
    for row in rows:
        key = (
            row.get("match_id", ""),
            row.get("source_url", ""),
            row.get("start", ""),
            row.get("player1_url", ""),
            row.get("player2_url", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def grand_slam_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        for row in read_csv(path):
            if row.get("category") == "grand_slam" or row.get("tournament") in GRAND_SLAMS:
                rows.append(row)
    rows = dedupe(rows)
    rows.sort(key=lambda row: (row.get("start", ""), row.get("tournament", ""), row.get("match_id", "")))
    return rows


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
    parser = argparse.ArgumentParser(description="Prepare Grand Slam BO5 moneyline train/test files.")
    parser.add_argument("--matches", nargs="+", type=Path, required=True)
    parser.add_argument("--model-dataset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("files/processed/grand_slam_moneyline/model_dataset"))
    args = parser.parse_args()

    matches = grand_slam_rows(args.matches)
    write_csv(args.out_dir / "grand_slam_matches.csv", matches)

    rows = [row for row in read_csv(args.model_dataset) if row.get("target_gana_jugador_1") in {"0", "1", 0, 1}]
    train_rows = [
        row
        for row in rows
        if row.get("categoria_torneo") == "grand_slam"
        and row.get("fecha", "") >= "2022-01-01"
        and row.get("fecha", "") < "2026-01-01"
    ]
    test_rows = [
        row
        for row in rows
        if row.get("torneo") == "Australian Open"
        and row.get("fecha", "") >= "2026-01-01"
        and row.get("fecha", "") < "2026-02-01"
    ]
    train_rows = orient_completed_rows(train_rows)
    test_rows = orient_completed_rows(test_rows)
    write_csv(args.out_dir / "train.csv", train_rows)
    write_csv(args.out_dir / "test_australian_open_2026.csv", test_rows)
    summary = {
        "grand_slam_match_rows": len(matches),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_period": "2022-01-01 to 2025-12-31",
        "test_period": "Australian Open 2026",
        "target": "moneyline",
    }
    (args.out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
