from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag


BASE_URL = "https://www.tennisexplorer.com"
DEFAULT_TOURNAMENT_URL = "https://www.tennisexplorer.com/houston/2026/atp-men/"
SURFACES = ("clay", "hard", "indoors", "grass", "not_set")
DEFAULT_LEVELS = ("grand_slam", "masters_1000", "atp_500", "atp_250")
LOWER_LEVEL_MARKERS = ("chall", "challenger", "itf", "utr", "exh", "exhibition", "futures")
GRAND_SLAMS = {"australian open", "roland garros", "french open", "wimbledon", "us open"}
MASTERS_1000 = {
    "indian wells",
    "miami",
    "monte carlo",
    "madrid",
    "rome",
    "canada masters",
    "toronto",
    "montreal",
    "cincinnati",
    "shanghai",
    "paris masters",
    "paris",
}
ATP_500 = {
    "acapulco",
    "barcelona",
    "basel",
    "beijing",
    "dallas",
    "doha",
    "dubai",
    "halle",
    "hamburg",
    "munich",
    "queen's club",
    "queens club",
    "rio de janeiro",
    "rotterdam",
    "tokyo",
    "vienna",
    "washington",
}


@dataclass
class ScrapeConfig:
    tournament_url: str
    output_dir: Path
    delay_seconds: float
    fetch_details: bool
    fetch_players: bool
    max_details: int | None
    max_players: int | None


@dataclass
class BatchScrapeConfig:
    year: int
    output_dir: Path
    levels: tuple[str, ...]
    delay_seconds: float
    fetch_details: bool
    fetch_players: bool
    max_tournaments: int | None
    max_details_per_tournament: int | None
    max_players: int | None
    completed_only: bool


class TennisExplorerClient:
    def __init__(self, delay_seconds: float = 1.0, retries: int = 4) -> None:
        self.delay_seconds = delay_seconds
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    def get_html(self, url: str) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                time.sleep(self.delay_seconds * attempt)
                response = self.session.get(url, timeout=45)
                response.raise_for_status()
                response.encoding = response.encoding or "utf-8"
                return response.text
            except requests.RequestException as exc:
                last_error = exc
                print(f"[retry {attempt}/{self.retries}] {url}: {exc}", file=sys.stderr)
                self.session.close()
                self.session = requests.Session()
                self.session.headers.update(
                    {
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/124.0 Safari/537.36"
                        ),
                        "Accept-Language": "en-US,en;q=0.9",
                    }
                )
        raise RuntimeError(f"Failed to fetch {url}") from last_error


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    value = value.replace("\xa0", " ")
    return re.sub(r"\s+", " ", value).strip()


def text_lines(soup: BeautifulSoup) -> list[str]:
    return [clean_text(line) for line in soup.get_text("\n").splitlines() if clean_text(line)]


def to_float(value: str) -> float | None:
    value = clean_text(value)
    if not value or value == "-":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def to_int(value: str) -> int | None:
    value = clean_text(value)
    if not value or value == "-":
        return None
    match = re.search(r"-?\d+", value)
    return int(match.group(0)) if match else None


def parse_wl(value: str) -> tuple[int | None, int | None]:
    value = clean_text(value)
    if value in {"", "-"}:
        return None, None
    match = re.match(r"^(\d+)\s*/\s*(\d+)$", value)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def parse_player_seed(raw_name: str) -> tuple[str, int | None]:
    raw_name = clean_text(raw_name)
    seed = None
    seed_match = re.search(r"\((\d+)\)$", raw_name)
    if seed_match:
        seed = int(seed_match.group(1))
        raw_name = raw_name[: seed_match.start()].strip()
    return raw_name, seed


def split_matchup(raw_matchup: str) -> tuple[str, int | None, str, int | None]:
    left, separator, right = clean_text(raw_matchup).partition(" - ")
    if not separator:
        return clean_text(raw_matchup), None, "", None
    player1, seed1 = parse_player_seed(left)
    player2, seed2 = parse_player_seed(right)
    return player1, seed1, player2, seed2


def parse_score_cell(cell: Tag) -> dict[str, int | None | str]:
    direct = "".join(str(node) for node in cell.children if not isinstance(node, Tag))
    games = to_int(clean_text(direct))
    sup = cell.find("sup")
    tiebreak = to_int(sup.get_text(" ", strip=True)) if sup else None
    return {"games": games, "tiebreak": tiebreak, "raw": clean_text(cell.get_text(" ", strip=True))}


def parse_tournament_metadata(soup: BeautifulSoup, url: str) -> dict[str, str | int | None]:
    title = clean_text(soup.find("h1").get_text(" ", strip=True)) if soup.find("h1") else ""
    tournament = title
    year = None
    country = None
    title_match = re.match(r"(.+?)\s+(\d{4})\s+\((.+?)\)", title)
    if title_match:
        tournament = title_match.group(1)
        year = int(title_match.group(2))
        country = title_match.group(3)

    details_box = ""
    for box in soup.select(".boxBasic"):
        text = clean_text(box.get_text(" ", strip=True))
        if "clay" in text.lower() or "hard" in text.lower() or "grass" in text.lower():
            details_box = text.strip("()")
            break

    prize_money = None
    surface = None
    gender = None
    detail_match = re.match(r"(.+),\s*(clay|hard|indoors|grass|not set),\s*(men|women)", details_box, re.I)
    if detail_match:
        prize_money = clean_text(detail_match.group(1))
        surface = clean_text(detail_match.group(2)).lower()
        gender = clean_text(detail_match.group(3)).lower()

    parsed = urlparse(url)
    slug_parts = [part for part in parsed.path.split("/") if part]
    tour = slug_parts[2] if len(slug_parts) >= 3 else None

    return {
        "tournament": tournament,
        "year": year,
        "country": country,
        "prize_money": prize_money,
        "surface": surface,
        "gender": gender,
        "tour": tour,
        "source_url": url,
    }


def normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower().replace("-", " ")).strip()


def classify_tournament(name: str) -> str | None:
    normalized = normalize_name(name)
    if any(marker in normalized for marker in LOWER_LEVEL_MARKERS):
        return None
    if normalized in GRAND_SLAMS:
        return "grand_slam"
    if normalized in MASTERS_1000:
        return "masters_1000"
    if normalized in ATP_500:
        return "atp_500"
    if normalized in {"davis cup", "united cup", "laver cup", "next gen atp finals", "atp finals"}:
        return None
    return "atp_250"


def calendar_url(year: int) -> str:
    return f"{BASE_URL}/calendar/atp-men/{year}/"


def discover_calendar_tournaments(html: str, year: int, levels: tuple[str, ...], completed_only: bool) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []
    seen: set[str] = set()
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if not re.match(rf"^/[^/]+/{year}/atp-men/$", href):
            continue
        name = clean_text(link.get_text(" ", strip=True))
        category = classify_tournament(name)
        if category is None or category not in levels:
            continue
        tournament_url = urljoin(BASE_URL, href)
        if tournament_url in seen:
            continue
        seen.add(tournament_url)

        row = link.find_parent("tr")
        row_text = clean_text(row.get_text(" ", strip=True)) if row else ""
        cells = [clean_text(cell.get_text(" ", strip=True)) for cell in row.find_all("td")] if row else []
        singles_winner = cells[5] if len(cells) >= 6 else ""
        if completed_only and singles_winner in {"", "-"}:
            continue
        start_date = None
        date_match = re.search(r"(\d{2}\.\d{2}\.)\s+%s" % year, row_text)
        if date_match:
            start_date = f"{date_match.group(1)}{year}"
        draw_size = None
        if len(cells) >= 5:
            draw_size = to_int(cells[4])

        rows.append(
            {
                "year": year,
                "tournament": name,
                "category": category,
                "start_date": start_date,
                "draw_size": draw_size,
                "calendar_row": row_text,
                "source_url": tournament_url,
            }
        )
    return rows


def parse_match_datetime(date_time_text: str, year: int | None) -> str | None:
    date_time_text = clean_text(date_time_text)
    match = re.search(r"(\d{2})\.(\d{2})\.\s+(\d{2}):(\d{2})", date_time_text)
    if not match or not year:
        return None
    day, month, hour, minute = map(int, match.groups())
    return datetime(year, month, day, hour, minute).isoformat(timespec="minutes")


def parse_tournament_results(html: str, url: str) -> tuple[dict, list[dict]]:
    soup = BeautifulSoup(html, "html.parser")
    metadata = parse_tournament_metadata(soup, url)
    table = soup.select_one("#tournamentTabs-1-data table.result")
    if not table:
        return metadata, []

    rows: list[dict] = []
    for first_row in table.select("tbody tr[id]"):
        row_id = first_row.get("id", "")
        if row_id.endswith("b"):
            continue
        second_row = table.select_one(f"tr#{row_id}b")
        if not second_row:
            continue

        first_cells = first_row.find_all("td")
        second_cells = second_row.find_all("td")
        if len(first_cells) < 11 or len(second_cells) < 7:
            continue

        date_time_text = clean_text(first_cells[0].get_text(" ", strip=True))
        round_code = clean_text(first_cells[1].get_text(" ", strip=True))
        round_name = clean_text(first_cells[1].get("title", ""))

        player1_link = first_cells[2].find("a")
        player2_link = second_cells[0].find("a")
        player1, seed1 = parse_player_seed(first_cells[2].get_text(" ", strip=True))
        player2, seed2 = parse_player_seed(second_cells[0].get_text(" ", strip=True))

        sets1 = to_int(first_cells[3].get_text(" ", strip=True))
        sets2 = to_int(second_cells[1].get_text(" ", strip=True))
        score1 = [parse_score_cell(cell) for cell in first_cells[4:9]]
        score2 = [parse_score_cell(cell) for cell in second_cells[2:7]]
        games1 = sum(cell["games"] or 0 for cell in score1)
        games2 = sum(cell["games"] or 0 for cell in score2)

        info_link = first_row.find("a", href=re.compile(r"/match-detail/\?id="))
        match_url = urljoin(BASE_URL, info_link["href"]) if info_link else None
        match_id = None
        if match_url:
            match_id_match = re.search(r"id=(\d+)", match_url)
            match_id = match_id_match.group(1) if match_id_match else None

        odds_cells = first_row.select("td.course")
        odds1 = to_float(odds_cells[0].get_text(" ", strip=True)) if len(odds_cells) > 0 else None
        odds2 = to_float(odds_cells[1].get_text(" ", strip=True)) if len(odds_cells) > 1 else None

        winner = None
        if sets1 is not None and sets2 is not None:
            winner = player1 if sets1 > sets2 else player2

        rows.append(
            {
                **metadata,
                "category": classify_tournament(str(metadata.get("tournament") or "")),
                "match_id": match_id,
                "match_url": match_url,
                "start": parse_match_datetime(date_time_text, metadata["year"]),
                "round": round_code,
                "round_name": round_name,
                "player1": player1,
                "player2": player2,
                "player1_url": urljoin(BASE_URL, player1_link["href"]) if player1_link else None,
                "player2_url": urljoin(BASE_URL, player2_link["href"]) if player2_link else None,
                "seed1": seed1,
                "seed2": seed2,
                "sets1": sets1,
                "sets2": sets2,
                "score1_json": json.dumps(score1, ensure_ascii=True),
                "score2_json": json.dumps(score2, ensure_ascii=True),
                "games1": games1,
                "games2": games2,
                "total_games": games1 + games2,
                "game_diff": games1 - games2,
                "set_diff": (sets1 - sets2) if sets1 is not None and sets2 is not None else None,
                "winner": winner,
                "odds1_avg": odds1,
                "odds2_avg": odds2,
            }
        )
    return metadata, rows


def parse_next_matches(html: str, url: str) -> tuple[dict, list[dict]]:
    soup = BeautifulSoup(html, "html.parser")
    metadata = parse_tournament_metadata(soup, url)
    table = soup.select_one("#tournamentTabs-1-data table.result")
    if not table:
        return metadata, []

    rows: list[dict] = []
    header_text = clean_text(table.get_text(" ", strip=True)).lower()
    if "next matches" not in clean_text(soup.get_text(" ", strip=True)).lower():
        return metadata, rows

    for row in table.select("tbody tr"):
        cells = row.find_all("td")
        if len(cells) < 7:
            continue
        matchup_link = cells[2].find("a", href=True)
        if not matchup_link:
            continue
        player1, seed1, player2, seed2 = split_matchup(matchup_link.get_text(" ", strip=True))
        if not player1 or not player2:
            continue
        match_url = urljoin(BASE_URL, matchup_link["href"])
        match_id_match = re.search(r"id=(\d+)", match_url)
        h2h_text = clean_text(cells[4].get_text(" ", strip=True))
        h2h1 = h2h2 = None
        h2h_match = re.match(r"^(\d+)-(\d+)$", h2h_text)
        if h2h_match:
            h2h1 = int(h2h_match.group(1))
            h2h2 = int(h2h_match.group(2))
        rows.append(
            {
                **metadata,
                "category": classify_tournament(str(metadata.get("tournament") or "")),
                "match_id": match_id_match.group(1) if match_id_match else None,
                "match_url": match_url,
                "start_raw": clean_text(cells[0].get_text(" ", strip=True)),
                "round": clean_text(cells[1].get_text(" ", strip=True)),
                "round_name": clean_text(cells[1].get_text(" ", strip=True)),
                "player1": player1,
                "player2": player2,
                "seed1": seed1,
                "seed2": seed2,
                "h2h_player1_wins": h2h1,
                "h2h_player2_wins": h2h2,
                "odds1_avg": to_float(cells[5].get_text(" ", strip=True)),
                "odds2_avg": to_float(cells[6].get_text(" ", strip=True)),
                "status": "upcoming",
            }
        )
    return metadata, rows


def parse_surface_snapshot(lines: list[str]) -> dict[str, str | None]:
    snapshot: dict[str, str | None] = {}
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith("Surface "))
    except StopIteration:
        return snapshot

    for line in lines[start + 1 :]:
        if line.startswith("Year Summary"):
            break
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "Not" and len(parts) >= 2 and parts[1] == "set":
            surface = "not_set"
            values = parts[2:]
        else:
            surface = parts[0].lower()
            values = parts[1:]
        if surface not in SURFACES or len(values) < 2:
            continue
        snapshot[f"{surface}_player1_wl"] = values[0]
        snapshot[f"{surface}_player2_wl"] = values[1]
    return snapshot


def parse_record_blocks(lines: list[str], owner: str) -> list[dict]:
    records: list[dict] = []
    block_number = 0
    i = 0
    while i < len(lines):
        if lines[i] != "Year Summary Clay Hard Indoors Grass Not set":
            i += 1
            continue
        block_number += 1
        draw_type = {1: "singles", 2: "doubles", 3: "mixed_doubles"}.get(block_number, f"block_{block_number}")
        i += 1
        while i < len(lines):
            line = lines[i]
            if line == "Year Summary Clay Hard Indoors Grass Not set" or line.startswith("Titles -") or line.startswith("Played matches"):
                break
            parts = line.split()
            if not parts or (parts[0] != "Summary:" and not re.match(r"^\d{4}$", parts[0])):
                i += 1
                continue
            year = "summary" if parts[0] == "Summary:" else parts[0]
            values = parts[1:]
            values += ["-"] * (6 - len(values))
            row = {"player": owner, "draw_type": draw_type, "year": year}
            for key, value in zip(("summary", "clay", "hard", "indoors", "grass", "not_set"), values[:6]):
                wins, losses = parse_wl(value)
                row[f"{key}_wl"] = value
                row[f"{key}_wins"] = wins
                row[f"{key}_losses"] = losses
            records.append(row)
            i += 1
    return records


def parse_player_profile(html: str, url: str) -> tuple[dict, list[dict], list[dict]]:
    soup = BeautifulSoup(html, "html.parser")
    lines = text_lines(soup)
    title = clean_text(soup.find("h1").get_text(" ", strip=True)) if soup.find("h1") else ""
    player = title.replace("- profile", "").strip()

    profile = {"player": player, "player_url": url}
    patterns = {
        "country": r"^Country:\s+(.+)$",
        "height_weight": r"^Height / Weight:\s+(.+)$",
        "age_birthdate": r"^Age:\s+(.+)$",
        "rank_singles": r"^Current/Highest rank - singles:\s+(.+)$",
        "rank_doubles": r"^Current/Highest rank - doubles:\s+(.+)$",
        "sex": r"^Sex:\s+(.+)$",
        "plays": r"^Plays:\s+(.+)$",
    }
    for line in lines:
        for key, pattern in patterns.items():
            match = re.match(pattern, line)
            if match:
                profile[key] = match.group(1)

    records = parse_record_blocks(lines, player)
    played_matches = parse_player_played_matches_table(soup, player)
    return profile, records, played_matches


def parse_player_played_matches_table(soup: BeautifulSoup, player: str) -> list[dict]:
    matches: list[dict] = []
    table = soup.select_one("div[id^='matches-'][id$='-1-data'] table.result.balance")
    if not table:
        return matches

    current_tournament = None
    current_tournament_url = None
    for row in table.select("tr"):
        if "head" in row.get("class", []):
            link = row.select_one("td.t-name a[href]")
            current_tournament = clean_text(link.get_text(" ", strip=True)) if link else None
            current_tournament_url = urljoin(BASE_URL, link["href"]) if link else None
            continue

        cells = row.find_all("td")
        if len(cells) < 7:
            continue
        date = clean_text(cells[0].get_text(" ", strip=True))
        surface_tag = cells[1].find(attrs={"title": True})
        surface = clean_text(surface_tag.get("title")) if surface_tag else None
        matchup_cell = cells[2]
        player_links = matchup_cell.find_all("a", href=True)
        if len(player_links) < 2:
            continue
        player1 = clean_text(player_links[0].get_text(" ", strip=True))
        player2 = clean_text(player_links[1].get_text(" ", strip=True))
        player1_url = urljoin(BASE_URL, player_links[0]["href"])
        player2_url = urljoin(BASE_URL, player_links[1]["href"])
        owner_side = 1 if player_links[0].find("strong") else 2 if player_links[1].find("strong") else None
        round_cell = cells[3]
        score_link = cells[4].find("a", href=True)
        score_raw = clean_text(cells[4].get_text(" ", strip=True))
        match_url = urljoin(BASE_URL, score_link["href"]) if score_link else None
        match_id = None
        if match_url:
            match_id_match = re.search(r"id=(\d+)", match_url)
            match_id = match_id_match.group(1) if match_id_match else None
        odds1 = to_float(cells[5].get_text(" ", strip=True))
        odds2 = to_float(cells[6].get_text(" ", strip=True))
        matches.append(
            {
                "player": player,
                "tournament": current_tournament,
                "tournament_url": current_tournament_url,
                "date": date,
                "surface": surface,
                "player1": player1,
                "player2": player2,
                "player1_url": player1_url,
                "player2_url": player2_url,
                "owner_side": owner_side,
                "round": clean_text(round_cell.get_text(" ", strip=True)),
                "round_name": clean_text(round_cell.get("title", "")),
                "score_raw": score_raw,
                "match_id": match_id,
                "match_url": match_url,
                "odds1": odds1,
                "odds2": odds2,
                "source": "player_profile",
            }
        )
    return matches


def parse_player_played_matches(lines: list[str], player: str) -> list[dict]:
    matches: list[dict] = []
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith("Played matches -"))
    except StopIteration:
        return matches

    current_tournament = None
    current_year = None
    year_match = re.search(r"Played matches - (\d{4})", lines[start])
    if year_match:
        current_year = int(year_match.group(1))

    date_re = re.compile(r"^\d{2}\.\d{2}\.")
    for line in lines[start + 1 :]:
        if line in {"Injuries", "Player's profile", "Tournament search"} or line.startswith("This week's tournaments"):
            break
        if line.endswith("Round Result H A") and not date_re.match(line):
            current_tournament = line.replace("Round Result H A", "").strip()
            continue
        if not date_re.match(line):
            continue

        odds = re.findall(r"\b\d+\.\d+\b", line)
        odds1 = float(odds[-2]) if len(odds) >= 2 else None
        odds2 = float(odds[-1]) if len(odds) >= 2 else None
        line_without_odds = re.sub(r"\s+\d+\.\d+\s+\d+\.\d+$", "", line)
        score_match = re.search(r"(\d{1,2}[^\w]+\d{1,2}.*)$", line_without_odds)
        score = clean_text(score_match.group(1)) if score_match else None
        prefix = line_without_odds[: score_match.start()].strip() if score_match else line_without_odds
        parts = prefix.split()
        date = parts[0] if parts else None
        round_code = parts[-1] if len(parts) > 1 else None
        matchup = " ".join(parts[1:-1]) if len(parts) > 2 else ""
        matches.append(
            {
                "player": player,
                "year": current_year,
                "tournament": current_tournament,
                "date": date,
                "round": round_code,
                "matchup_raw": matchup,
                "score_raw": score,
                "odds1": odds1,
                "odds2": odds2,
                "source": "player_profile",
            }
        )
    return matches


def parse_match_detail(html: str, url: str) -> tuple[dict, list[dict]]:
    soup = BeautifulSoup(html, "html.parser")
    lines = text_lines(soup)
    title = clean_text(soup.find("h1").get_text(" ", strip=True)) if soup.find("h1") else ""

    detail = {"match_url": url, "title": title}
    if len(lines) > 0:
        detail["source_lines_sample"] = json.dumps(lines[:120], ensure_ascii=True)

    info_line = next((line for line in lines if re.match(r"^\d{2}\.\d{2}\.\d{4},", line)), "")
    info_parts = [clean_text(part) for part in info_line.split(",")]
    if len(info_parts) >= 5:
        detail["start_date"] = info_parts[0]
        detail["start_time"] = info_parts[1]
        detail["tournament"] = info_parts[2]
        detail["round_name"] = info_parts[3]
        detail["surface"] = info_parts[4].lower()

    h2h_line = next((line for line in lines if line.startswith("Head-to-head:")), "")
    h2h_match = re.search(r"Head-to-head:\s*(\d+)\s*-\s*(\d+)", h2h_line)
    if h2h_match:
        detail["h2h_player1_wins"] = int(h2h_match.group(1))
        detail["h2h_player2_wins"] = int(h2h_match.group(2))

    detail.update(parse_surface_snapshot(lines))
    detail.update(parse_homeaway_average_odds(lines))
    bookmaker_rows = parse_homeaway_bookmakers(lines, url)
    return detail, bookmaker_rows


def parse_homeaway_average_odds(lines: list[str]) -> dict[str, float | None]:
    result = {"homeaway_avg_odds1": None, "homeaway_avg_odds2": None}
    for i, line in enumerate(lines):
        if line == "Average odds" and i + 2 < len(lines):
            result["homeaway_avg_odds1"] = to_float(lines[i + 1])
            result["homeaway_avg_odds2"] = to_float(lines[i + 2])
            break
    return result


def parse_homeaway_bookmakers(lines: list[str], match_url: str) -> list[dict]:
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith("Home/Away"))
        end = next(i for i, line in enumerate(lines[start + 1 :], start + 1) if line.startswith("Over/Under"))
    except StopIteration:
        return []

    rows: list[dict] = []
    current_bookmaker = None
    pending_odds: list[float] = []
    skip = {"Opening odds", "Average odds"}
    for line in lines[start + 1 : end]:
        if line in skip or re.match(r"^\d{2}\.\d{2}\.", line):
            continue
        odd = to_float(line)
        if odd is not None:
            pending_odds.append(odd)
            if current_bookmaker and len(pending_odds) >= 2:
                rows.append(
                    {
                        "match_url": match_url,
                        "bookmaker": current_bookmaker,
                        "odds1": pending_odds[0],
                        "odds2": pending_odds[1],
                    }
                )
                current_bookmaker = None
                pending_odds = []
            continue
        if line:
            current_bookmaker = line
            pending_odds = []
    return rows


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scrape_tournament_data(
    client: TennisExplorerClient,
    tournament_url: str,
    fetch_details: bool,
    fetch_players: bool,
    max_details: int | None = None,
    max_players: int | None = None,
) -> dict[str, list[dict] | dict]:
    tournament_html = client.get_html(tournament_url)
    metadata, tournament_rows = parse_tournament_results(tournament_html, tournament_url)
    _, upcoming_rows = parse_next_matches(tournament_html, tournament_url)

    detail_rows: list[dict] = []
    odds_rows: list[dict] = []
    if fetch_details:
        detail_urls = [row["match_url"] for row in tournament_rows if row.get("match_url")]
        if max_details:
            detail_urls = detail_urls[:max_details]
        for detail_url in detail_urls:
            try:
                detail_html = client.get_html(detail_url)
                detail, bookmakers = parse_match_detail(detail_html, detail_url)
                match_id_match = re.search(r"id=(\d+)", detail_url)
                detail["match_id"] = match_id_match.group(1) if match_id_match else None
                detail_rows.append(detail)
                odds_rows.extend(bookmakers)
            except Exception as exc:
                print(f"[detail skipped] {detail_url}: {exc}", file=sys.stderr)

    profiles: list[dict] = []
    records: list[dict] = []
    played_matches: list[dict] = []
    if fetch_players:
        player_urls = sorted(
            {
                row.get("player1_url")
                for row in tournament_rows
                if row.get("player1_url")
            }
            | {
                row.get("player2_url")
                for row in tournament_rows
                if row.get("player2_url")
            }
        )
        if max_players:
            player_urls = player_urls[:max_players]
        for player_url in player_urls:
            player_html = client.get_html(player_url)
            profile, player_records, player_matches = parse_player_profile(player_html, player_url)
            profiles.append(profile)
            records.extend(player_records)
            played_matches.extend(player_matches)

    return {
        "metadata": metadata,
        "matches": tournament_rows,
        "upcoming_matches": upcoming_rows,
        "details": detail_rows,
        "bookmaker_odds": odds_rows,
        "player_profiles": profiles,
        "player_surface_records": records,
        "player_played_matches": played_matches,
    }


def scrape(config: ScrapeConfig) -> None:
    client = TennisExplorerClient(delay_seconds=config.delay_seconds)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    scraped = scrape_tournament_data(
        client,
        config.tournament_url,
        fetch_details=config.fetch_details,
        fetch_players=config.fetch_players,
        max_details=config.max_details,
        max_players=config.max_players,
    )
    metadata = scraped["metadata"]
    tournament_rows = scraped["matches"]
    (config.output_dir / "tournament_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    write_csv(config.output_dir / "tournament_matches.csv", tournament_rows)
    write_csv(config.output_dir / "upcoming_matches.csv", scraped["upcoming_matches"])
    write_csv(config.output_dir / "match_details.csv", scraped["details"])
    write_csv(config.output_dir / "match_homeaway_bookmaker_odds.csv", scraped["bookmaker_odds"])
    write_csv(config.output_dir / "player_profiles.csv", scraped["player_profiles"])
    write_csv(config.output_dir / "player_surface_records.csv", scraped["player_surface_records"])
    write_csv(config.output_dir / "player_played_matches.csv", scraped["player_played_matches"])


def scrape_batch(config: BatchScrapeConfig) -> None:
    client = TennisExplorerClient(delay_seconds=config.delay_seconds)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    calendar_html = client.get_html(calendar_url(config.year))
    tournaments = discover_calendar_tournaments(
        calendar_html,
        config.year,
        levels=config.levels,
        completed_only=config.completed_only,
    )
    if config.max_tournaments:
        tournaments = tournaments[: config.max_tournaments]
    write_csv(config.output_dir / "tournament_calendar.csv", tournaments)
    print(f"Discovered {len(tournaments)} ATP tournaments for {config.year}.")

    all_metadata: list[dict] = []
    all_matches: list[dict] = []
    all_upcoming: list[dict] = []
    all_details: list[dict] = []
    all_odds: list[dict] = []
    all_profiles: dict[str, dict] = {}
    all_records: list[dict] = []
    all_played_matches: list[dict] = []
    player_urls: set[str] = set()
    errors: list[dict] = []

    for index, tournament in enumerate(tournaments, start=1):
        source_url = tournament["source_url"]
        print(f"[{index}/{len(tournaments)}] Scraping {tournament['tournament']} ({tournament['category']})")
        try:
            scraped = scrape_tournament_data(
                client,
                source_url,
                fetch_details=config.fetch_details,
                fetch_players=False,
                max_details=config.max_details_per_tournament,
                max_players=config.max_players,
            )
        except Exception as exc:
            print(f"[tournament skipped] {source_url}: {exc}", file=sys.stderr)
            errors.append(
                {
                    "stage": "tournament",
                    "tournament": tournament.get("tournament"),
                    "source_url": source_url,
                    "error": str(exc),
                }
            )
            write_csv(config.output_dir / "scrape_errors.csv", errors)
            continue
        metadata = dict(scraped["metadata"])
        metadata["category"] = tournament["category"]
        metadata["calendar_start_date"] = tournament.get("start_date")
        all_metadata.append(metadata)

        for match in scraped["matches"]:
            match["category"] = tournament["category"]
            match["calendar_start_date"] = tournament.get("start_date")
            all_matches.append(match)
            if match.get("player1_url"):
                player_urls.add(str(match["player1_url"]))
            if match.get("player2_url"):
                player_urls.add(str(match["player2_url"]))
        for upcoming in scraped["upcoming_matches"]:
            upcoming["category"] = tournament["category"]
            upcoming["calendar_start_date"] = tournament.get("start_date")
            all_upcoming.append(upcoming)
        all_details.extend(scraped["details"])
        all_odds.extend(scraped["bookmaker_odds"])
        write_csv(config.output_dir / "tournament_metadata.csv", all_metadata)
        write_csv(config.output_dir / "matches.csv", all_matches)
        write_csv(config.output_dir / "upcoming_matches.csv", all_upcoming)
        write_csv(config.output_dir / "match_details.csv", all_details)
        write_csv(config.output_dir / "match_homeaway_bookmaker_odds.csv", all_odds)
        write_csv(config.output_dir / "scrape_errors.csv", errors)

    if config.fetch_players:
        urls_to_fetch = sorted(player_urls)
        if config.max_players:
            urls_to_fetch = urls_to_fetch[: config.max_players]
        print(f"Scraping {len(urls_to_fetch)} unique player profiles.")
        for index, player_url in enumerate(urls_to_fetch, start=1):
            print(f"[player {index}/{len(urls_to_fetch)}] {player_url}")
            player_html = client.get_html(player_url)
            profile, player_records, player_matches = parse_player_profile(player_html, player_url)
            all_profiles[profile["player_url"]] = profile
            all_records.extend(player_records)
            all_played_matches.extend(player_matches)

    write_csv(config.output_dir / "tournament_metadata.csv", all_metadata)
    write_csv(config.output_dir / "matches.csv", all_matches)
    write_csv(config.output_dir / "upcoming_matches.csv", all_upcoming)
    write_csv(config.output_dir / "match_details.csv", all_details)
    write_csv(config.output_dir / "match_homeaway_bookmaker_odds.csv", all_odds)
    write_csv(config.output_dir / "player_profiles.csv", all_profiles.values())
    write_csv(config.output_dir / "player_surface_records.csv", all_records)
    write_csv(config.output_dir / "player_played_matches.csv", all_played_matches)
    write_csv(config.output_dir / "scrape_errors.csv", errors)

    summary = {
        "year": config.year,
        "levels": list(config.levels),
        "completed_only": config.completed_only,
        "tournaments": len(tournaments),
        "matches": len(all_matches),
        "upcoming_matches": len(all_upcoming),
        "details": len(all_details),
        "players": len(all_profiles),
    }
    (config.output_dir / "scrape_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape TennisExplorer tournament, match and player data.")
    parser.add_argument("--batch-atp", action="store_true", help="Scrape ATP main tournaments from the season calendar.")
    parser.add_argument("--year", type=int, default=2026, help="Season year used with --batch-atp.")
    parser.add_argument(
        "--levels",
        nargs="+",
        default=list(DEFAULT_LEVELS),
        choices=list(DEFAULT_LEVELS),
        help="Tournament levels to keep with --batch-atp.",
    )
    parser.add_argument("--url", default=DEFAULT_TOURNAMENT_URL, help="Tournament URL to scrape.")
    parser.add_argument("--out", default=None, help="Output folder.")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds.")
    parser.add_argument("--no-details", action="store_true", help="Skip match detail pages.")
    parser.add_argument("--no-players", action="store_true", help="Skip player profile pages.")
    parser.add_argument("--completed-only", action="store_true", help="Skip calendar rows that do not have a winner yet.")
    parser.add_argument("--max-tournaments", type=int, default=None, help="Limit tournaments for testing.")
    parser.add_argument("--max-details", type=int, default=None, help="Limit match detail pages for testing.")
    parser.add_argument(
        "--max-details-per-tournament",
        type=int,
        default=None,
        help="Limit match detail pages per tournament in batch mode.",
    )
    parser.add_argument("--max-players", type=int, default=None, help="Limit player profile pages for testing.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.batch_atp:
        scrape_batch(
            BatchScrapeConfig(
                year=args.year,
                output_dir=Path(args.out or f"files/processed/atp_{args.year}"),
                levels=tuple(args.levels),
                delay_seconds=args.delay,
                fetch_details=not args.no_details,
                fetch_players=not args.no_players,
                max_tournaments=args.max_tournaments,
                max_details_per_tournament=args.max_details_per_tournament,
                max_players=args.max_players,
                completed_only=args.completed_only,
            )
        )
        return

    scrape(
        ScrapeConfig(
            tournament_url=args.url,
            output_dir=Path(args.out or "files/processed/tennisexplorer_houston_2026"),
            delay_seconds=args.delay,
            fetch_details=not args.no_details,
            fetch_players=not args.no_players,
            max_details=args.max_details,
            max_players=args.max_players,
        )
    )


if __name__ == "__main__":
    main()
