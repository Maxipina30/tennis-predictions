from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlencode, urlparse, urlunparse

from bs4 import BeautifulSoup

SCRAPER_PATH = Path(__file__).with_name("01_scrape_tennisexplorer.py")
spec = importlib.util.spec_from_file_location("tennisexplorer_scraper", SCRAPER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import scraper helpers from {SCRAPER_PATH}")
tennisexplorer_scraper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tennisexplorer_scraper
spec.loader.exec_module(tennisexplorer_scraper)

TennisExplorerClient = tennisexplorer_scraper.TennisExplorerClient
clean_text = tennisexplorer_scraper.clean_text
parse_player_profile = tennisexplorer_scraper.parse_player_profile
write_csv = tennisexplorer_scraper.write_csv


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def annual_url(player_url: str, year: int) -> str:
    parsed = urlparse(player_url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", urlencode({"annual": year}), ""))


def unique_player_urls(matches_path: Path, existing_profiles_path: Path | None = None) -> list[str]:
    urls: set[str] = set()
    for row in read_csv(matches_path):
        for key in ("player1_url", "player2_url"):
            if row.get(key):
                urls.add(row[key])
    if existing_profiles_path and existing_profiles_path.exists():
        existing = {row.get("player_url") for row in read_csv(existing_profiles_path)}
        urls = {url for url in urls if url not in existing}
    return sorted(urls)


def parse_player_name(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.find("h1")
    if not heading:
        return ""
    return clean_text(heading.get_text(" ", strip=True)).replace("- profile", "").strip()


def scrape_histories(
    matches_path: Path,
    output_dir: Path,
    years: list[int],
    delay: float,
    max_players: int | None,
    existing_profiles_path: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    client = TennisExplorerClient(delay_seconds=delay)
    player_urls = unique_player_urls(matches_path, existing_profiles_path)
    if max_players:
        player_urls = player_urls[:max_players]

    profiles_by_url: dict[str, dict] = {}
    surface_records: list[dict] = []
    player_matches: list[dict] = []
    player_injuries: list[dict] = []
    seen_injuries: set[tuple[str, str, str, str]] = set()
    errors: list[dict] = []

    for player_index, player_url in enumerate(player_urls, start=1):
        print(f"[player {player_index}/{len(player_urls)}] {player_url}")
        for year in years:
            url = annual_url(player_url, year)
            try:
                html = client.get_html(url)
                profile, records, matches, injuries = parse_player_profile(html, url)
                profile["player_url"] = player_url
                profiles_by_url[player_url] = profile
                for record in records:
                    record["player_url"] = player_url
                    surface_records.append(record)
                for match in matches:
                    match["player_url"] = player_url
                    match["annual_year"] = year
                    match["date_iso"] = parse_match_date(match.get("date"), year)
                    player_matches.append(match)
                for injury in injuries:
                    key = (
                        player_url,
                        injury.get("start_date") or injury.get("start_raw") or "",
                        injury.get("end_date") or injury.get("end_raw") or "",
                        injury.get("reason") or "",
                    )
                    if key in seen_injuries:
                        continue
                    seen_injuries.add(key)
                    injury["player_url"] = player_url
                    player_injuries.append(injury)
            except Exception as exc:
                print(f"[history skipped] {url}: {exc}")
                errors.append({"player_url": player_url, "year": year, "error": str(exc)})

        write_csv(output_dir / "player_profiles.csv", profiles_by_url.values())
        write_csv(output_dir / "player_surface_records.csv", surface_records)
        write_csv(output_dir / "player_matches.csv", player_matches)
        write_csv(output_dir / "player_injuries.csv", player_injuries)
        write_csv(output_dir / "player_history_errors.csv", errors)

    summary = {
        "matches_source": str(matches_path),
        "years": years,
        "players": len(player_urls),
        "player_matches": len(player_matches),
        "player_injuries": len(player_injuries),
        "errors": len(errors),
    }
    (output_dir / "player_history_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_match_date(value: str | None, year: int) -> str | None:
    if not value:
        return None
    match = re.match(r"^(\d{2})\.(\d{2})\.$", value.strip())
    if not match:
        return None
    day, month = map(int, match.groups())
    return f"{year:04d}-{month:02d}-{day:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape annual TennisExplorer histories for players in matches.csv.")
    parser.add_argument("--matches", default="files/processed/atp_2026/matches.csv")
    parser.add_argument("--out", default="files/processed/player_histories_2025_2026")
    parser.add_argument("--years", nargs="+", type=int, default=[2026, 2025])
    parser.add_argument("--delay", type=float, default=0.8)
    parser.add_argument("--max-players", type=int, default=None)
    parser.add_argument(
        "--skip-existing-profiles",
        default=None,
        help="Optional player_profiles.csv path; players already present there are not scraped again.",
    )
    args = parser.parse_args()
    scrape_histories(
        matches_path=Path(args.matches),
        output_dir=Path(args.out),
        years=args.years,
        delay=args.delay,
        max_players=args.max_players,
        existing_profiles_path=Path(args.skip_existing_profiles) if args.skip_existing_profiles else None,
    )


if __name__ == "__main__":
    main()
