from __future__ import annotations

import argparse
from pathlib import Path

from tennis_features import build_features_from_csvs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build no-leakage pre-match features from scraped matches.")
    parser.add_argument("--matches", default="files/processed/atp_2026/matches.csv")
    parser.add_argument("--details", default="files/processed/atp_2026/match_details.csv")
    parser.add_argument("--out", default="files/processed/atp_2026/features_no_leakage.csv")
    args = parser.parse_args()
    details_path = Path(args.details) if args.details else None
    build_features_from_csvs(Path(args.matches), Path(args.out), details_path)


if __name__ == "__main__":
    main()
