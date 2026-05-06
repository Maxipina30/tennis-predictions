from __future__ import annotations

import argparse
import csv
import hashlib
import math
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
            " ".join(parts[:-1]) if len(parts) > 1 else parts[0],
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


def filter_upcoming_rows(rows: list[dict], tournament: str | None, round_name: str | None) -> list[dict]:
    filtered = rows
    if tournament:
        filtered = [row for row in filtered if row.get("tournament") == tournament]
    if round_name:
        filtered = [row for row in filtered if row.get("round") == round_name or row.get("round_name") == round_name]
    return filtered


def parse_wl(value: object) -> tuple[int, int] | None:
    text = str(value or "").strip()
    if "/" not in text:
        return None
    left, right = text.split("/", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return None


def source_lines(value: object) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed]


def parse_detail_year_record_player1(lines: list[str]) -> tuple[int, int] | None:
    try:
        start = lines.index("Year")
    except ValueError:
        return None
    for index in range(start + 1, len(lines) - 1):
        if lines[index] == "2026":
            return parse_wl(lines[index + 1])
    return None


def parse_detail_wl_records(detail: dict) -> tuple[dict[int, dict[str, tuple[int, int]]], dict[int, tuple[int, int]]]:
    lines = source_lines(detail.get("source_lines_sample"))
    if not lines:
        return {}, {}
    try:
        start = lines.index("Surface") + 3
        end = lines.index("Year", start)
    except ValueError:
        return {}, {}

    records: dict[int, dict[str, tuple[int, int]]] = {1: {}, 2: {}}
    totals: dict[int, tuple[int, int]] = {}
    aliases = {"indoors": "indoors", "clay": "clay", "grass": "grass", "hard": "hard", "not set": "not_set"}
    index = start
    while index + 2 < end:
        surface = aliases.get(lines[index].strip().lower())
        wl1 = parse_wl(lines[index + 1])
        wl2 = parse_wl(lines[index + 2])
        if surface:
            if wl1 is not None:
                records[1][surface] = wl1
            if wl2 is not None:
                records[2][surface] = wl2
        index += 3
    player1_year = parse_detail_year_record_player1(lines)
    if player1_year:
        totals[1] = player1_year
    return records, totals


def set_wl_features(row: dict, side: int, records: dict[str, tuple[int, int]], surface: str, total_record: tuple[int, int] | None = None) -> None:
    if not records:
        return
    if total_record:
        total_wins, total_losses = total_record
    else:
        total_wins = sum(wins for wins, _ in records.values())
        total_losses = sum(losses for _, losses in records.values())
    total_matches = total_wins + total_losses
    if total_matches:
        prefix = f"jugador_{side}_ano_actual"
        row[f"{prefix}_partidos_previos"] = total_matches
        row[f"{prefix}_log_partidos_previos"] = math.log1p(total_matches)
        row[f"{prefix}_porcentaje_victorias_previas"] = total_wins / total_matches

    surface_record = records.get(surface)
    if surface_record:
        wins, losses = surface_record
        matches = wins + losses
        if matches:
            prefix = f"jugador_{side}_ano_actual_superficie"
            row[f"{prefix}_partidos_previos"] = matches
            row[f"{prefix}_log_partidos_previos"] = math.log1p(matches)
            row[f"{prefix}_porcentaje_victorias_previas"] = wins / matches


def refresh_detail_wl_differences(row: dict) -> None:
    for name in [
        "ano_actual_partidos_previos",
        "ano_actual_log_partidos_previos",
        "ano_actual_porcentaje_victorias_previas",
        "ano_actual_superficie_partidos_previos",
        "ano_actual_superficie_log_partidos_previos",
        "ano_actual_superficie_porcentaje_victorias_previas",
    ]:
        a = model_dataset.to_float(row.get(f"jugador_1_{name}"))
        b = model_dataset.to_float(row.get(f"jugador_2_{name}"))
        row[f"diferencia_{name}"] = None if a is None or b is None else a - b


def apply_detail_wl_overrides(upcoming_rows: list[dict], details_path: Path) -> None:
    details = {str(row.get("match_id")): row for row in read_csv(details_path)}
    if not details:
        return
    for row in upcoming_rows:
        detail = details.get(str(row.get("match_id")))
        if not detail:
            continue
        records, totals = parse_detail_wl_records(detail)
        if not records:
            continue
        surface = str(row.get("superficie") or "").lower().replace(" ", "_")
        set_wl_features(row, 1, records.get(1, {}), surface, totals.get(1))
        set_wl_features(row, 2, records.get(2, {}), surface, totals.get(2))
        refresh_detail_wl_differences(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split model dataset into train/test and add upcoming rows.")
    parser.add_argument("--dataset", default="files/processed/model_dataset_2026/model_dataset.csv")
    parser.add_argument("--histories", default="files/processed/player_histories_2024_2026/player_matches.csv")
    parser.add_argument("--injuries", default="files/processed/player_histories_2024_2026/player_injuries.csv")
    parser.add_argument("--rankings", default="files/processed/atp_rankings/player_ranking_history.csv")
    parser.add_argument("--sofascore-stats", default="files/processed/sofascore_tennis/match_stats_2025_2026.csv")
    parser.add_argument("--matches", default="files/processed/atp_2026/matches.csv")
    parser.add_argument("--upcoming", default="files/processed/atp_2026/upcoming_matches.csv")
    parser.add_argument("--profiles", default="files/processed/player_histories_2024_2026/player_profiles.csv")
    parser.add_argument("--out-dir", default="files/processed/model_dataset_2026")
    parser.add_argument("--upcoming-date", default="2026-04-28")
    parser.add_argument("--upcoming-output", default="test_madrid_upcoming.csv")
    parser.add_argument("--upcoming-tournament", default=None)
    parser.add_argument("--upcoming-round", default=None)
    parser.add_argument("--upcoming-today-offset-days", type=int, default=0)
    parser.add_argument("--upcoming-tomorrow-offset-days", type=int, default=1)
    parser.add_argument("--upcoming-details", default=None)
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
    sofascore_rows = read_csv(Path(args.sofascore_stats))
    sofascore_stats_by_match = model_dataset.build_sofascore_stats_index(sofascore_rows)
    matches = read_csv(Path(args.matches))
    tournament_categories = {row.get("source_url", ""): row.get("category", "") for row in matches}
    history_events = model_dataset.build_history_events(histories, tournament_categories, sofascore_stats_by_match)
    name_to_key = build_name_key_lookup(read_csv(Path(args.profiles)))
    upcoming_source_rows = filter_upcoming_rows(read_csv(Path(args.upcoming)), args.upcoming_tournament, args.upcoming_round)
    upcoming_events = model_dataset.build_upcoming_events(
        upcoming_source_rows,
        args.upcoming_date,
        name_to_key,
        args.upcoming_today_offset_days,
        args.upcoming_tomorrow_offset_days,
    )
    upcoming_rows = model_dataset.build_rows_for_events(upcoming_events, history_events, injuries, rankings)
    details_path = Path(args.upcoming_details) if args.upcoming_details else Path(args.upcoming).with_name("upcoming_match_details.csv")
    apply_detail_wl_overrides(upcoming_rows, details_path)

    write_csv(out_dir / "train.csv", train_rows)
    write_csv(out_dir / "test_barcelona_munich.csv", test_rows)
    write_csv(out_dir / args.upcoming_output, upcoming_rows)
    write_csv(out_dir / "test.csv", test_rows + upcoming_rows)
    summary = {
        "train_rows": len(train_rows),
        "test_completed_rows": len(test_rows),
        "test_upcoming_rows": len(upcoming_rows),
        "test_upcoming_output": args.upcoming_output,
        "test_upcoming_tournament": args.upcoming_tournament,
        "test_upcoming_round": args.upcoming_round,
        "test_upcoming_today_offset_days": args.upcoming_today_offset_days,
        "test_upcoming_tomorrow_offset_days": args.upcoming_tomorrow_offset_days,
        "test_total_rows": len(test_rows) + len(upcoming_rows),
        "test_completed_tournaments": sorted(test_tournaments | {"Madrid"}),
        "injury_rows": len(injuries),
        "ranking_rows": len(rankings),
        "sofascore_stat_rows": len(sofascore_rows),
        "sofascore_stat_matches": len(sofascore_stats_by_match),
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
