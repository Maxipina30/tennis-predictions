from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path


SCRAPER_PATH = Path(__file__).with_name("01_scrape_tennisexplorer.py")
spec = importlib.util.spec_from_file_location("tennisexplorer_scraper", SCRAPER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import scraper helpers from {SCRAPER_PATH}")
tennisexplorer_scraper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tennisexplorer_scraper
spec.loader.exec_module(tennisexplorer_scraper)

TennisExplorerClient = tennisexplorer_scraper.TennisExplorerClient
clean_text = tennisexplorer_scraper.clean_text
write_csv = tennisexplorer_scraper.write_csv


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def normalize(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def parse_set_score(value: str) -> tuple[list[dict], list[dict]]:
    score1: list[dict] = []
    score2: list[dict] = []
    for part in value.strip("() ").split(","):
        part = part.strip()
        match = re.match(r"^(\d+)(?:\((\d+)\))?-(\d+)(?:\((\d+)\))?$", part)
        if not match:
            continue
        g1, tb1, g2, tb2 = match.groups()
        score1.append({"games": int(g1), "tiebreak": int(tb1) if tb1 else None, "raw": g1})
        score2.append({"games": int(g2), "tiebreak": int(tb2) if tb2 else None, "raw": g2})
    while len(score1) < 5:
        score1.append({"games": None, "tiebreak": None, "raw": ""})
        score2.append({"games": None, "tiebreak": None, "raw": ""})
    return score1, score2


def profile_lookup(profiles_path: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for row in read_csv(profiles_path):
        name = row.get("player") or ""
        url = row.get("player_url") or ""
        if not name or not url:
            continue
        lookup[normalize(name)] = url
        parts = normalize(name).split()
        if parts:
            lookup.setdefault(parts[0], url)
    return lookup


def soup_and_text_lines(html: str):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    return soup, [clean_text(line) for line in soup.get_text("\n").split("\n") if clean_text(line)]


def parse_completed_detail(html: str, match_url: str, profiles: dict[str, str]) -> dict | None:
    soup, lines = soup_and_text_lines(html)
    title = clean_text(soup.find("h1").get_text(" ", strip=True)) if soup.find("h1") else ""
    title_index = lines.index(title) if title in lines else None
    if title_index is None or title_index + 8 >= len(lines):
        return None
    player1, player2 = [part.strip() for part in title.split(" - ", 1)]
    date = lines[title_index + 1]
    tournament = lines[title_index + 3]
    round_surface = lines[title_index + 4]
    player1_full = lines[title_index + 5]
    sets_line = lines[title_index + 6]
    score_line = lines[title_index + 7]
    player2_full = lines[title_index + 8]
    sets_match = re.match(r"^(\d+)\s*:\s*(\d+)$", sets_line)
    if not sets_match or not score_line.startswith("("):
        return None
    sets1, sets2 = map(int, sets_match.groups())
    score1, score2 = parse_set_score(score_line)
    games1 = sum(item["games"] or 0 for item in score1)
    games2 = sum(item["games"] or 0 for item in score2)
    round_code = ""
    round_name = ""
    if "round of 16" in round_surface:
        round_code = "R16"
        round_name = "round of 16"
    elif "quarter" in round_surface:
        round_code = "QF"
        round_name = "quarterfinal"
    elif "semi" in round_surface:
        round_code = "SF"
        round_name = "semifinal"
    elif "final" in round_surface:
        round_code = "F"
        round_name = "final"
    surface = "clay" if "clay" in round_surface else "hard" if "hard" in round_surface else "grass" if "grass" in round_surface else "unknown"
    match_id_match = re.search(r"id=(\d+)", match_url)
    match_id = match_id_match.group(1) if match_id_match else ""
    start = datetime.strptime(date, "%d.%m.%Y").strftime("%Y-%m-%dT00:00")
    return {
        "calendar_start_date": "22.04.2026",
        "category": "masters_1000",
        "country": "Spain",
        "game_diff": games1 - games2,
        "games1": games1,
        "games2": games2,
        "gender": "men",
        "match_id": match_id,
        "match_url": match_url,
        "odds1_avg": "",
        "odds2_avg": "",
        "player1": player1,
        "player1_url": profiles.get(normalize(player1_full), profiles.get(normalize(player1), "")),
        "player2": player2,
        "player2_url": profiles.get(normalize(player2_full), profiles.get(normalize(player2), "")),
        "prize_money": "8,235,540 EUR",
        "round": round_code,
        "round_name": round_name,
        "score1_json": json.dumps(score1, ensure_ascii=True),
        "score2_json": json.dumps(score2, ensure_ascii=True),
        "seed1": "",
        "seed2": "",
        "set_diff": sets1 - sets2,
        "sets1": sets1,
        "sets2": sets2,
        "source_url": "https://www.tennisexplorer.com/madrid/2026/atp-men/",
        "start": start,
        "surface": surface,
        "total_games": games1 + games2,
        "tour": "atp-men",
        "tournament": tournament,
        "winner": player1 if sets1 > sets2 else player2,
        "year": "2026",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Add completed tournament matches from TennisExplorer match-detail pages.")
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    profiles = profile_lookup(args.profiles)
    client = TennisExplorerClient(delay_seconds=args.delay)
    existing = read_csv(args.matches)
    existing_ids = {row.get("match_id") for row in existing}
    rows = []
    for detail in read_csv(args.details):
        match_id = detail.get("match_id")
        if not match_id or match_id in existing_ids:
            continue
        match_url = f"https://www.tennisexplorer.com/match-detail/?id={match_id}"
        html = client.get_html(match_url)
        row = parse_completed_detail(html, match_url, profiles)
        if row and row.get("player1_url") and row.get("player2_url"):
            print(f"[completed] {row['player1']} vs {row['player2']} {row['sets1']}-{row['sets2']}")
            rows.append(row)
    merged = existing + rows
    write_csv(args.matches, merged)
    print(f"[write] added {len(rows)} completed matches; total {len(merged)}")


if __name__ == "__main__":
    main()
