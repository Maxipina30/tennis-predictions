from __future__ import annotations

import argparse
import csv
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
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dedupe(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    seen: set[tuple[str, ...]] = set()
    output: list[dict] = []
    for row in rows:
        key = tuple(row.get(column, "") for column in keys)
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def combine_matches(paths: list[Path], output_path: Path) -> None:
    rows: list[dict] = []
    for path in paths:
        rows.extend(read_csv(path))
    rows = dedupe(rows, ("match_id", "source_url", "start", "player1_url", "player2_url"))
    rows.sort(key=lambda row: (row.get("start", ""), row.get("match_id", "")))
    write_csv(output_path, rows)
    print(f"[combine] matches: {len(rows)} -> {output_path}")


def combine_history_dirs(paths: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles: list[dict] = []
    records: list[dict] = []
    matches: list[dict] = []
    injuries: list[dict] = []
    errors: list[dict] = []
    for path in paths:
        profiles.extend(read_csv(path / "player_profiles.csv"))
        records.extend(read_csv(path / "player_surface_records.csv"))
        matches.extend(read_csv(path / "player_matches.csv"))
        injuries.extend(read_csv(path / "player_injuries.csv"))
        errors.extend(read_csv(path / "player_history_errors.csv"))

    profiles = dedupe(profiles, ("player_url",))
    records = dedupe(records, ("player_url", "surface", "matches", "wins", "losses"))
    matches = dedupe(matches, ("player_url", "annual_year", "date_iso", "tournament", "opponent", "score_raw"))
    injuries = dedupe(injuries, ("player_url", "start_date", "end_date", "reason"))
    errors = dedupe(errors, ("player_url", "year", "error"))

    write_csv(output_dir / "player_profiles.csv", profiles)
    write_csv(output_dir / "player_surface_records.csv", records)
    write_csv(output_dir / "player_matches.csv", matches)
    write_csv(output_dir / "player_injuries.csv", injuries)
    write_csv(output_dir / "player_history_errors.csv", errors)
    print(f"[combine] profiles: {len(profiles)}")
    print(f"[combine] player_matches: {len(matches)}")
    print(f"[combine] injuries: {len(injuries)}")
    print(f"[combine] errors: {len(errors)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine scraped match and player-history sources for training.")
    parser.add_argument("--matches", nargs="+", type=Path, required=True)
    parser.add_argument("--out-matches", type=Path, required=True)
    parser.add_argument("--history-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-history-dir", type=Path, required=True)
    args = parser.parse_args()
    combine_matches(args.matches, args.out_matches)
    combine_history_dirs(args.history_dirs, args.out_history_dir)


if __name__ == "__main__":
    main()
