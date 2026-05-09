from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import sys
from pathlib import Path


SCRAPER_PATH = Path(__file__).with_name("01_scrape_tennisexplorer.py")
spec = importlib.util.spec_from_file_location("tennisexplorer_scraper", SCRAPER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import scraper helpers from {SCRAPER_PATH}")
tennisexplorer_scraper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tennisexplorer_scraper
spec.loader.exec_module(tennisexplorer_scraper)

TennisExplorerClient = tennisexplorer_scraper.TennisExplorerClient
parse_match_detail = tennisexplorer_scraper.parse_match_detail
write_csv = tennisexplorer_scraper.write_csv


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def normalized_name(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def name_matches(short_name: str | None, detail_name: str | None) -> bool:
    short = normalized_name(short_name)
    detail = normalized_name(detail_name)
    return bool(short and detail and (short in detail or detail in short))


def to_float(value: object) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def detail_title_players(title: str | None) -> tuple[str, str] | None:
    if not title or " - " not in title:
        return None
    left, right = title.split(" - ", 1)
    return left.strip(), right.strip()


def swap_pairs(detail: dict, pairs: list[tuple[str, str]]) -> None:
    for left, right in pairs:
        detail[left], detail[right] = detail.get(right), detail.get(left)


def swap_detail_sides(detail: dict) -> None:
    pairs = [
        ("homeaway_avg_odds1", "homeaway_avg_odds2"),
        ("h2h_player1_wins", "h2h_player2_wins"),
        ("set_odds_player1_wins_2_0", "set_odds_player2_wins_2_0"),
    ]
    swap_pairs(detail, pairs)


def swap_set_odds_sides(detail: dict) -> None:
    swap_pairs(detail, [("set_odds_player1_wins_set", "set_odds_player2_wins_set")])


def align_detail_to_upcoming_order(detail: dict, upcoming: dict) -> bool:
    players = detail_title_players(detail.get("title"))
    if not players:
        return False
    detail_player1, detail_player2 = players
    upcoming_player1 = upcoming.get("player1")
    upcoming_player2 = upcoming.get("player2")
    already_aligned = name_matches(upcoming_player1, detail_player1) and name_matches(upcoming_player2, detail_player2)
    reversed_order = name_matches(upcoming_player1, detail_player2) and name_matches(upcoming_player2, detail_player1)
    if not already_aligned and reversed_order:
        swap_detail_sides(detail)
        return True
    return False


def choose_wins_set_odds(candidates: list[float | None], moneyline: float | None) -> float | None:
    usable = [candidate for candidate in candidates if candidate is not None and candidate > 1]
    if moneyline is not None:
        below_moneyline = [candidate for candidate in usable if candidate < moneyline]
        if below_moneyline:
            return max(below_moneyline)
    return None


def set_odds_candidates_for_table_side(detail: dict, side: int) -> list[float | None]:
    return [
        to_float(detail.get(f"set_odds_table_player{side}_minus_1_5")),
        to_float(detail.get(f"set_odds_table_player{side}_plus_1_5")),
    ]


def table_side_for_upcoming_player(detail: dict, player: str | None) -> int | None:
    if name_matches(player, detail.get("set_odds_handicap_player1")):
        return 1
    if name_matches(player, detail.get("set_odds_handicap_player2")):
        return 2
    return None


def align_set_odds_to_upcoming_order(detail: dict, upcoming: dict, fallback_swapped: bool) -> None:
    handicap_player1 = detail.get("set_odds_handicap_player1")
    handicap_player2 = detail.get("set_odds_handicap_player2")
    upcoming_player1 = upcoming.get("player1")
    upcoming_player2 = upcoming.get("player2")
    odds1 = to_float(upcoming.get("odds1_avg"))
    odds2 = to_float(upcoming.get("odds2_avg"))

    already_aligned = name_matches(upcoming_player1, handicap_player1) and name_matches(upcoming_player2, handicap_player2)
    reversed_order = name_matches(upcoming_player1, handicap_player2) and name_matches(upcoming_player2, handicap_player1)
    side1 = table_side_for_upcoming_player(detail, upcoming_player1)
    side2 = table_side_for_upcoming_player(detail, upcoming_player2)
    player1_wins_set = choose_wins_set_odds(set_odds_candidates_for_table_side(detail, side1), odds1) if side1 else None
    player2_wins_set = choose_wins_set_odds(set_odds_candidates_for_table_side(detail, side2), odds2) if side2 else None

    if player1_wins_set is not None:
        detail["set_odds_player1_wins_set"] = player1_wins_set
    if player2_wins_set is not None:
        detail["set_odds_player2_wins_set"] = player2_wins_set
    if player1_wins_set is not None or player2_wins_set is not None:
        detail["set_odds_validation"] = "selected_below_moneyline"
    elif reversed_order:
        swap_set_odds_sides(detail)
        detail["set_odds_validation"] = "fallback_reversed_order"
    elif not already_aligned and fallback_swapped:
        swap_set_odds_sides(detail)
        detail["set_odds_validation"] = "fallback_title_order"
    else:
        detail["set_odds_validation"] = "fallback_parser_order"


def validate_market_odds(detail: dict, upcoming: dict) -> None:
    """Drop set/2-0 odds that violate the math vs the moneyline.

    P(wins set) >= P(wins match) -> set odds must be <= ML odds.
    P(wins 2-0) <= P(wins match) -> 2-0 odds must be >= ML odds.
    When the bookmaker does not offer the matching market for an extreme
    favorite the parser can pick up an unrelated handicap line; nulling
    the value is safer than letting the dashboard recommend a wrong cuota.
    """
    odds1 = to_float(upcoming.get("odds1_avg"))
    odds2 = to_float(upcoming.get("odds2_avg"))
    nulled = []
    invariants = [
        ("set_odds_player1_wins_set", odds1, lambda v, ml: v > ml),
        ("set_odds_player2_wins_set", odds2, lambda v, ml: v > ml),
        ("set_odds_player1_wins_2_0", odds1, lambda v, ml: v < ml),
        ("set_odds_player2_wins_2_0", odds2, lambda v, ml: v < ml),
    ]
    for field, ml, violates in invariants:
        value = to_float(detail.get(field))
        if value is None or ml is None:
            continue
        if violates(value, ml):
            detail[field] = None
            nulled.append(field)
    detail["market_odds_check"] = "ok" if not nulled else "nulled:" + ",".join(nulled)


def scrape_upcoming_details(upcoming_path: Path, output_path: Path, delay: float) -> None:
    client = TennisExplorerClient(delay_seconds=delay)
    rows = []
    for match in read_csv(upcoming_path):
        match_url = match.get("match_url")
        if not match_url:
            continue
        print(f"[detail] {match.get('player1')} vs {match.get('player2')} {match_url}")
        html = client.get_html(match_url)
        detail, _ = parse_match_detail(html, match_url)
        title_order_was_swapped = align_detail_to_upcoming_order(detail, match)
        align_set_odds_to_upcoming_order(detail, match, title_order_was_swapped)
        validate_market_odds(detail, match)
        match_id_match = re.search(r"id=(\d+)", match_url)
        detail["match_id"] = match_id_match.group(1) if match_id_match else match.get("match_id")
        detail["player1"] = match.get("player1")
        detail["player2"] = match.get("player2")
        rows.append(detail)
        write_csv(output_path, rows)
    nulled_rows = [row for row in rows if str(row.get("market_odds_check", "")).startswith("nulled:")]
    for row in nulled_rows:
        print(f"[check] {row.get('player1')} vs {row.get('player2')}: {row.get('market_odds_check')}")
    print(f"[check] market odds sanity ok ({len(rows) - len(nulled_rows)}/{len(rows)} rows untouched, {len(nulled_rows)} had invalid markets nulled)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape match-detail odds for upcoming matches.")
    parser.add_argument("--upcoming", default="files/processed/atp_2026/upcoming_matches.csv")
    parser.add_argument("--out", default="files/processed/atp_2026/upcoming_match_details.csv")
    parser.add_argument("--delay", type=float, default=0.8)
    args = parser.parse_args()
    scrape_upcoming_details(Path(args.upcoming), Path(args.out), args.delay)


if __name__ == "__main__":
    main()
