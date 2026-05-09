from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]


def run(args: list[str]) -> None:
    print("[run]", " ".join(args))
    subprocess.run(args, cwd=BASE, check=True)


def slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def tournament_slug(tournament: str) -> str:
    return "-".join(tournament.lower().split())


def derive_url(tournament: str, year: str) -> str:
    return f"https://www.tennisexplorer.com/{tournament_slug(tournament)}/{year}/atp-men/"


def derive_label(tournament: str, round_name: str) -> str:
    return f"{tournament} {round_name}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dashboard predictions for a tournament round.")
    parser.add_argument("--tournament", required=True, help="Tournament name as it appears on TennisExplorer (e.g. Rome, Madrid).")
    parser.add_argument("--round", required=True, help="Round identifier (e.g. 1R, 2R, QF, SF, F).")
    parser.add_argument("--year", default="2026")
    parser.add_argument("--label", default=None, help="Display label. Defaults to '{tournament} {round}'.")
    parser.add_argument("--url", default=None, help="TennisExplorer URL. Defaults to https://www.tennisexplorer.com/{slug}/{year}/atp-men/.")
    parser.add_argument("--reference-date", default=date.today().isoformat())
    parser.add_argument("--raw-today-offset-days", type=int, default=0)
    parser.add_argument("--raw-tomorrow-offset-days", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--target-dir", default=None)
    parser.add_argument("--dataset-dir", default="files/processed/model_dataset_2025_2026")
    parser.add_argument("--training-dir", default="files/processed/model_training_2025_2026")
    parser.add_argument("--rankings-dir", default="files/processed/atp_rankings")
    parser.add_argument("--profiles", default="files/processed/player_histories_2024_2026_extended/player_profiles.csv")
    parser.add_argument("--histories", default="files/processed/player_histories_2024_2026_extended/player_matches.csv")
    parser.add_argument("--injuries", default="files/processed/player_histories_2024_2026_extended/player_injuries.csv")
    parser.add_argument("--matches", default="files/processed/training_2025_2026/matches.csv")
    parser.add_argument("--model-dataset", default="files/processed/model_dataset_2025_2026/model_dataset.csv")
    parser.add_argument("--sofascore-stats", default="files/processed/sofascore_tennis/match_stats_2025_2026.csv")
    parser.add_argument("--skip-tournament-scrape", action="store_true")
    parser.add_argument("--skip-ranking-refresh", action="store_true")
    args = parser.parse_args()
    if not args.label:
        args.label = derive_label(args.tournament, args.round)
    if not args.url:
        args.url = derive_url(args.tournament, args.year)

    target_slug = slug(args.label or f"{args.tournament}_{args.round}")
    target_dir = Path(args.target_dir or f"files/processed/{target_slug}")
    upcoming_path = target_dir / "upcoming_matches.csv"
    upcoming_rows_path = Path(args.dataset_dir) / f"test_{target_slug}_upcoming.csv"
    predictions_path = Path(args.training_dir) / f"{target_slug}_predictions.csv"
    rankings_path = Path(args.rankings_dir) / "player_ranking_history.csv"

    if not args.skip_tournament_scrape:
        run(
            [
                sys.executable,
                "src/01_scrape_tennisexplorer.py",
                "--url",
                args.url,
                "--out",
                str(target_dir),
                "--delay",
                str(args.delay),
                "--no-details",
                "--no-players",
            ]
        )

    if not args.skip_ranking_refresh:
        run(
            [
                sys.executable,
                "src/10_scrape_atp_rankings.py",
                "--profiles",
                args.profiles,
                "--upcoming",
                str(upcoming_path),
                "--upcoming-only",
                "--refresh-existing",
                "--out-dir",
                args.rankings_dir,
                "--delay",
                str(args.delay),
            ]
        )

    run(
        [
            sys.executable,
            "src/09_scrape_upcoming_details.py",
            "--upcoming",
            str(upcoming_path),
            "--out",
            str(target_dir / "upcoming_match_details.csv"),
            "--delay",
            str(args.delay),
        ]
    )

    run(
        [
            sys.executable,
            "src/05_build_model_dataset.py",
            "--matches",
            args.matches,
            "--histories",
            args.histories,
            "--injuries",
            args.injuries,
            "--rankings",
            str(rankings_path),
            "--sofascore-stats",
            args.sofascore_stats,
            "--out",
            args.model_dataset,
        ]
    )
    run(
        [
            sys.executable,
            "src/06_split_train_test.py",
            "--dataset",
            args.model_dataset,
            "--histories",
            args.histories,
            "--injuries",
            args.injuries,
            "--matches",
            args.matches,
            "--upcoming",
            str(upcoming_path),
            "--profiles",
            args.profiles,
            "--out-dir",
            args.dataset_dir,
            "--upcoming-date",
            args.reference_date,
            "--upcoming-output",
            upcoming_rows_path.name,
            "--upcoming-tournament",
            args.tournament,
            "--upcoming-round",
            args.round,
            "--upcoming-today-offset-days",
            str(args.raw_today_offset_days),
            "--upcoming-tomorrow-offset-days",
            str(args.raw_tomorrow_offset_days),
            "--rankings",
            str(rankings_path),
            "--sofascore-stats",
            args.sofascore_stats,
        ]
    )
    run(
        [
            sys.executable,
            "src/07_train_compare_models.py",
            "--train",
            str(Path(args.dataset_dir) / "train.csv"),
            "--test",
            str(Path(args.dataset_dir) / "test_barcelona_munich.csv"),
            "--upcoming",
            str(upcoming_rows_path),
            "--prediction-output",
            predictions_path.name,
            "--out-dir",
            args.training_dir,
        ]
    )

    config = {
        "label": args.label,
        "tournament": args.tournament,
        "round": args.round,
        "predictions_file": str(predictions_path).replace("\\", "/"),
        "upcoming_file": str(upcoming_path).replace("\\", "/"),
        "local_reference_date": args.reference_date,
        "raw_today_offset_days": args.raw_today_offset_days,
        "raw_tomorrow_offset_days": args.raw_tomorrow_offset_days,
        "rankings_file": str(rankings_path).replace("\\", "/"),
    }
    config_path = BASE / "files/processed/dashboard_target.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[dashboard] wrote {config_path}")


if __name__ == "__main__":
    main()
