from __future__ import annotations

import argparse
from pathlib import Path

from tennis_features import add_match_detail_snapshots, build_no_leakage_features, read_csv, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Build no-leakage features from multiple season match CSVs.")
    parser.add_argument("--matches", nargs="+", required=True, help="One or more matches.csv files.")
    parser.add_argument("--details", nargs="*", default=[], help="Optional match_details.csv files.")
    parser.add_argument("--out", required=True, help="Output feature CSV.")
    args = parser.parse_args()

    all_matches: list[dict] = []
    for path in args.matches:
        all_matches.extend(read_csv(Path(path)))
    rows = build_no_leakage_features(all_matches)
    if args.details:
        all_details: list[dict] = []
        for path in args.details:
            all_details.extend(read_csv(Path(path)))
        rows = add_match_detail_snapshots(rows, all_details)
    write_csv(Path(args.out), rows)


if __name__ == "__main__":
    main()
