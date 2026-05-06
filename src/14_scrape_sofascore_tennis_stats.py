from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from playwright.async_api import async_playwright


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parents[1]
API_URL = "https://api.sofascore.com/api/v1"
SITE_URL = "https://www.sofascore.com"

STAT_KEYS = {
    "aces": "aces",
    "ace": "aces",
    "doubleFaults": "double_faults",
    "double faults": "double_faults",
    "1st serve": "first_serve_pct",
    "firstServe": "first_serve_pct",
    "first serve": "first_serve_pct",
    "1st serve points won": "first_serve_points_won_pct",
    "firstServePointsWon": "first_serve_points_won_pct",
    "first serve points won": "first_serve_points_won_pct",
    "2nd serve points won": "second_serve_points_won_pct",
    "secondServePointsWon": "second_serve_points_won_pct",
    "second serve points won": "second_serve_points_won_pct",
    "break points converted": "break_points_converted",
    "breakPointsConverted": "break_points_converted",
    "break points saved": "break_points_saved",
    "breakPointsSaved": "break_points_saved",
    "break points": "break_points",
    "breakPoints": "break_points",
    "service games won": "service_games_won",
    "serviceGamesWon": "service_games_won",
    "return games won": "return_games_won",
    "returnGamesWon": "return_games_won",
    "service points won": "service_points_won_pct",
    "servicePointsWon": "service_points_won_pct",
    "receiving points won": "receiving_points_won_pct",
    "receivingPointsWon": "receiving_points_won_pct",
    "total points won": "total_points_won_pct",
    "totalPointsWon": "total_points_won_pct",
    "1st serve return points won": "first_serve_return_points_won_pct",
    "firstServeReturnPointsWon": "first_serve_return_points_won_pct",
    "2nd serve return points won": "second_serve_return_points_won_pct",
    "secondServeReturnPointsWon": "second_serve_return_points_won_pct",
    "max points in a row": "max_points_in_a_row",
    "maxPointsInARow": "max_points_in_a_row",
}
SET_SUM_STATS = {
    "break_points_converted",
    "break_points_saved",
}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def event_datetime(event: dict, timezone: str) -> datetime:
    return datetime.fromtimestamp(event["startTimestamp"], ZoneInfo(timezone))


def normalize_text(value: str | None) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(char for char in value if not unicodedata.combining(char)).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def name_tokens(value: str | None) -> set[str]:
    stop = {"jr", "sr"}
    return {token for token in normalize_text(value).split() if len(token) > 1 and token not in stop}


def names_match(left: str | None, right: str | None) -> bool:
    left_tokens = name_tokens(left)
    right_tokens = name_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    return bool(left_tokens & right_tokens)


def sofascore_player_name(team: dict) -> str:
    return team.get("name") or team.get("shortName") or team.get("slug") or ""


def match_event(row: dict, event: dict) -> str | None:
    home = sofascore_player_name(event.get("homeTeam") or {})
    away = sofascore_player_name(event.get("awayTeam") or {})
    p1 = row.get("player1")
    p2 = row.get("player2")
    if names_match(p1, home) and names_match(p2, away):
        return "normal"
    if names_match(p1, away) and names_match(p2, home):
        return "reversed"
    return None


def stat_key(item: dict) -> str | None:
    candidates = [
        item.get("key"),
        item.get("name"),
        item.get("title"),
        item.get("statisticsType"),
    ]
    for candidate in candidates:
        if candidate in STAT_KEYS:
            return STAT_KEYS[candidate]
        normalized = normalize_text(str(candidate or ""))
        if normalized in STAT_KEYS:
            return STAT_KEYS[normalized]
    return None


def number_from_value(value: object) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def parse_fraction(value: object) -> tuple[float | None, float | None, float | None]:
    if value in (None, ""):
        return None, None, None
    text = str(value)
    fraction = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", text)
    pct = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    made = total = None
    if fraction:
        made = float(fraction.group(1))
        total = float(fraction.group(2))
    pct_value = float(pct.group(1)) if pct else None
    if pct_value is None and made is not None and total:
        pct_value = round(made / total * 100, 1)
    return made, total, pct_value


def add_stat(target: dict, key: str, value: object) -> None:
    made, total, pct_value = parse_fraction(value)
    if key.endswith("_pct"):
        target[key] = pct_value if pct_value is not None else number_from_value(value)
        base = key.removesuffix("_pct")
        if made is not None:
            target[f"{base}_won"] = made
        if total is not None:
            target[f"{base}_total"] = total
        return
    target[key] = number_from_value(value)
    if made is not None:
        target[f"{key}_won"] = made
    if total is not None:
        target[f"{key}_total"] = total


def stat_values_for_sum(value: object) -> tuple[float | None, float | None]:
    made, total, _ = parse_fraction(value)
    if made is not None:
        return made, total
    return number_from_value(value), None


def add_set_sum(target: dict, key: str, value: object) -> None:
    made, total = stat_values_for_sum(value)
    if made is None:
        return
    target[key] = target.get(key, 0.0) + made
    if total is not None:
        target[f"{key}_total"] = target.get(f"{key}_total", 0.0) + total


def parse_statistics(payload: dict | None) -> dict[str, dict]:
    by_side = {"home": {}, "away": {}}
    if not isinstance(payload, dict):
        return by_side
    periods = payload.get("statistics") or []
    all_period = next((period for period in periods if period.get("period") == "ALL"), None)
    if all_period is None and periods:
        all_period = periods[0]
    for group in (all_period or {}).get("groups", []):
        for item in group.get("statisticsItems", []):
            key = stat_key(item)
            if not key:
                continue
            add_stat(by_side["home"], key, item.get("homeValue"))
            add_stat(by_side["away"], key, item.get("awayValue"))
    set_sums = {"home": {}, "away": {}}
    for period in periods:
        if period.get("period") == "ALL":
            continue
        for group in period.get("groups", []):
            for item in group.get("statisticsItems", []):
                key = stat_key(item)
                if key not in SET_SUM_STATS:
                    continue
                add_set_sum(set_sums["home"], key, item.get("homeValue"))
                add_set_sum(set_sums["away"], key, item.get("awayValue"))
    for side in by_side:
        for key, value in set_sums[side].items():
            by_side[side][key] = value
    return by_side


async def api_get_json(context, url: str, referer: str, required: bool = False) -> dict | None:
    response = await context.request.get(
        url,
        headers={"Accept": "application/json,text/plain,*/*", "Referer": referer},
        timeout=60_000,
    )
    text = await response.text()
    if response.status == 404 and not required:
        return None
    if response.status != 200:
        if required:
            raise RuntimeError(f"HTTP {response.status} en {url}: {text[:300]}")
        print(f"ADVERTENCIA HTTP {response.status}: {url}")
        return None
    return json.loads(text)


async def scheduled_events_for_date(context, date: str, referer: str, cache_dir: Path, refresh: bool) -> list[dict]:
    cache_path = cache_dir / f"scheduled_{date}.json"
    if cache_path.exists() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        payload = await api_get_json(context, f"{API_URL}/sport/tennis/scheduled-events/{date}", referer, required=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload.get("events", []) if isinstance(payload, dict) else []


def row_for_match(match: dict, event: dict, orientation: str, stats: dict, timezone: str) -> dict:
    event_dt = event_datetime(event, timezone)
    p1_side = "home" if orientation == "normal" else "away"
    p2_side = "away" if orientation == "normal" else "home"
    row = {
        "match_id": match.get("match_id"),
        "date": event_dt.date().isoformat(),
        "time": event_dt.strftime("%H:%M"),
        "tournament": match.get("tournament"),
        "round": match.get("round"),
        "player1": match.get("player1"),
        "player2": match.get("player2"),
        "player1_url": match.get("player1_url"),
        "player2_url": match.get("player2_url"),
        "sofascore_event_id": event.get("id"),
        "sofascore_custom_id": event.get("customId"),
        "sofascore_slug": event.get("slug"),
        "sofascore_url": f"{SITE_URL}/{event.get('slug', '')}/{event.get('customId', '')}",
        "match_orientation": orientation,
    }
    for prefix, side in (("player1", p1_side), ("player2", p2_side)):
        for key, value in stats[side].items():
            row[f"{prefix}_{key}"] = value
    return row


async def scrape(args: argparse.Namespace) -> None:
    matches_path = Path(args.matches)
    out_path = Path(args.out)
    cache_dir = Path(args.cache_dir)
    if not matches_path.is_absolute():
        matches_path = BASE_DIR / matches_path
    if not out_path.is_absolute():
        out_path = BASE_DIR / out_path
    if not cache_dir.is_absolute():
        cache_dir = BASE_DIR / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    matches = [row for row in read_csv(matches_path) if parse_date(row.get("start"))]
    if args.from_date:
        matches = [row for row in matches if str(row.get("start", ""))[:10] >= args.from_date]
    if args.to_date:
        matches = [row for row in matches if str(row.get("start", ""))[:10] <= args.to_date]
    if args.limit:
        matches = matches[: args.limit]
    dates = sorted({str(row.get("start"))[:10] for row in matches})

    rows: list[dict] = []
    misses: list[dict] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=args.headless)
        context = await browser.new_context(
            locale="en-US",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        referer = f"{SITE_URL}/tennis"
        page = await context.new_page()
        await page.goto(referer, wait_until="domcontentloaded", timeout=60_000)

        events_by_date: dict[str, list[dict]] = {}
        for date in dates:
            events_by_date[date] = await scheduled_events_for_date(context, date, referer, cache_dir, args.refresh_cache)
            if args.delay:
                await asyncio.sleep(args.delay)

        total = len(matches)
        for index, match in enumerate(matches, start=1):
            date = str(match.get("start"))[:10]
            found = None
            orientation = None
            for event in events_by_date.get(date, []):
                orientation = match_event(match, event)
                if orientation:
                    found = event
                    break
            if not found or not orientation:
                misses.append({"match_id": match.get("match_id"), "date": date, "player1": match.get("player1"), "player2": match.get("player2")})
                continue
            stats_payload = await api_get_json(context, f"{API_URL}/event/{found['id']}/statistics", referer)
            stats = parse_statistics(stats_payload)
            rows.append(row_for_match(match, found, orientation, stats, args.timezone))
            if index % 25 == 0 or index == total:
                print(f"Procesados {index}/{total} partidos; encontrados={len(rows)}; sin_match={len(misses)}")
            if args.delay:
                await asyncio.sleep(args.delay)
        await browser.close()

    write_csv(out_path, rows)
    if misses:
        write_csv(out_path.with_name(out_path.stem + "_misses.csv"), misses)
    print(f"Guardadas stats SofaScore en {out_path} ({len(rows)} filas)")
    if misses:
        print(f"Partidos sin match SofaScore: {len(misses)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape tennis match stats from SofaScore API.")
    parser.add_argument("--matches", default="files/processed/atp_2026/matches.csv")
    parser.add_argument("--out", default="files/processed/sofascore_tennis/match_stats.csv")
    parser.add_argument("--cache-dir", default="files/processed/sofascore_tennis/cache")
    parser.add_argument("--from-date", default=None)
    parser.add_argument("--to-date", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timezone", default="America/Sao_Paulo")
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    asyncio.run(scrape(args))


if __name__ == "__main__":
    main()
