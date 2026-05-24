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


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine Grand Slam pipeline player histories.")
    parser.add_argument("--history-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("files/processed/grand_slam_moneyline/histories_combined"))
    args = parser.parse_args()

    profiles: list[dict] = []
    matches: list[dict] = []
    injuries: list[dict] = []
    errors: list[dict] = []
    for directory in args.history_dirs:
        profiles.extend(read_csv(directory / "player_profiles.csv"))
        matches.extend(read_csv(directory / "player_matches.csv"))
        injuries.extend(read_csv(directory / "player_injuries.csv"))
        errors.extend(read_csv(directory / "player_history_errors.csv"))

    profiles = dedupe(profiles, ("player_url",))
    matches = dedupe(matches, ("player_url", "annual_year", "date_iso", "tournament", "opponent", "score_raw"))
    injuries = dedupe(injuries, ("player_url", "start_date", "end_date", "reason"))
    errors = dedupe(errors, ("player_url", "year", "error"))

    write_csv(args.out_dir / "player_profiles.csv", profiles)
    write_csv(args.out_dir / "player_matches.csv", matches)
    write_csv(args.out_dir / "player_injuries.csv", injuries)
    write_csv(args.out_dir / "player_history_errors.csv", errors)

    print(f"profiles={len(profiles)}")
    print(f"player_matches={len(matches)}")
    print(f"injuries={len(injuries)}")
    print(f"errors={len(errors)}")


if __name__ == "__main__":
    main()
