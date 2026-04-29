from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


DATASET_PATH = Path(__file__).with_name("05_build_model_dataset.py")
spec = importlib.util.spec_from_file_location("model_dataset", DATASET_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import helpers from {DATASET_PATH}")
model_dataset = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model_dataset
spec.loader.exec_module(model_dataset)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_name_key_lookup(profile_rows: list[dict]) -> dict[str, str]:
    candidates: dict[str, list[str]] = {}
    for row in profile_rows:
        name = row.get("player", "")
        key = model_dataset.player_key_from_url(row.get("player_url"))
        parts = [part for part in name.lower().split() if part]
        if not parts or not key:
            continue
        aliases = {
            parts[0],
            name.lower(),
            f"{parts[0]} {parts[-1][0]}" if len(parts) > 1 else parts[0],
        }
        for alias in aliases:
            clean_alias = " ".join(alias.replace(".", "").split())
            candidates.setdefault(clean_alias, []).append(key)
    lookup: dict[str, str] = {}
    for alias, keys in candidates.items():
        unique = sorted(set(keys))
        if len(unique) == 1:
            lookup[alias] = unique[0]
    # Known ambiguous surnames in the Madrid upcoming card.
    for row in profile_rows:
        if row.get("player") == "Cerundolo Francisco":
            lookup["cerundolo"] = model_dataset.player_key_from_url(row.get("player_url"))
    return lookup


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
    parser = argparse.ArgumentParser(description="Split model dataset into train/test and add Madrid upcoming rows.")
    parser.add_argument("--dataset", default="files/processed/model_dataset_2026/model_dataset.csv")
    parser.add_argument("--histories", default="files/processed/player_histories_2024_2026/player_matches.csv")
    parser.add_argument("--injuries", default="files/processed/player_histories_2024_2026/player_injuries.csv")
    parser.add_argument("--rankings", default="files/processed/atp_rankings/player_ranking_history.csv")
    parser.add_argument("--matches", default="files/processed/atp_2026/matches.csv")
    parser.add_argument("--upcoming", default="files/processed/atp_2026/upcoming_matches.csv")
    parser.add_argument("--profiles", default="files/processed/player_histories_2024_2026/player_profiles.csv")
    parser.add_argument("--out-dir", default="files/processed/model_dataset_2026")
    parser.add_argument("--upcoming-date", default="2026-04-28")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rows = read_csv(Path(args.dataset))
    test_tournaments = {"Barcelona", "Munich"}
    test_rows = orient_completed_rows(
        [
            row
            for row in rows
            if (
                row.get("torneo") in test_tournaments
                and row.get("fecha", "") >= "2026-04-13"
            )
            or (
                row.get("torneo") == "Madrid"
                and row.get("fecha", "") >= "2026-04-28"
                and row.get("target_gana_jugador_1") in {"0", "1", 0, 1}
            )
        ]
    )
    train_rows = orient_completed_rows(
        [row for row in rows if row.get("torneo") not in test_tournaments and row.get("fecha", "") < "2026-04-13"]
    )

    histories = read_csv(Path(args.histories))
    injuries = read_csv(Path(args.injuries))
    rankings = read_csv(Path(args.rankings))
    matches = read_csv(Path(args.matches))
    tournament_categories = {row.get("source_url", ""): row.get("category", "") for row in matches}
    history_events = model_dataset.build_history_events(histories, tournament_categories)
    name_to_key = build_name_key_lookup(read_csv(Path(args.profiles)))
    upcoming_events = model_dataset.build_upcoming_events(read_csv(Path(args.upcoming)), args.upcoming_date, name_to_key)
    upcoming_rows = model_dataset.build_rows_for_events(upcoming_events, history_events, injuries, rankings)

    write_csv(out_dir / "train.csv", train_rows)
    write_csv(out_dir / "test_barcelona_munich.csv", test_rows)
    write_csv(out_dir / "test_madrid_upcoming.csv", upcoming_rows)
    write_csv(out_dir / "test.csv", test_rows + upcoming_rows)
    summary = {
        "train_rows": len(train_rows),
        "test_completed_rows": len(test_rows),
        "test_madrid_upcoming_rows": len(upcoming_rows),
        "test_total_rows": len(test_rows) + len(upcoming_rows),
        "test_completed_tournaments": sorted(test_tournaments | {"Madrid"}),
        "injury_rows": len(injuries),
        "ranking_rows": len(rankings),
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
