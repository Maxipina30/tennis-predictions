from __future__ import annotations

import argparse
import csv
import math
import time
from bisect import bisect_right
from datetime import date, datetime
from pathlib import Path

import cloudscraper


ATP_RANKING_HISTORY_URL = "https://www.atptour.com/es/-/www/rank/history/{player_id}"
ATP_PLAYER_SEARCH_URL = "https://www.atptour.com/es/-/www/players/find/byname/{query}/es"
POC_PLAYERS = ["Sinner", "Norrie", "Zverev"]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    value = value.split("T", 1)[0]
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def resolve_player_id(scraper: cloudscraper.CloudScraper, player: str) -> str | None:
    response = scraper.get(ATP_PLAYER_SEARCH_URL.format(query=player.replace(" ", "%20")), timeout=30)
    response.raise_for_status()
    candidates = response.json()
    normalized = player.replace("-", " ").lower()
    active_candidates = [candidate for candidate in candidates if candidate.get("Active") == "A"]
    for candidate in active_candidates + candidates:
        full_name = f"{candidate.get('FirstName', '')} {candidate.get('LastName', '')}".strip().lower()
        last_name = str(candidate.get("LastName", "")).lower()
        if normalized == full_name or normalized == last_name or normalized in full_name:
            return str(candidate["PlayerId"]).lower()
    return str(candidates[0]["PlayerId"]).lower() if candidates else None


def scrape_rank_history(scraper: cloudscraper.CloudScraper, player: str, player_id: str, delay: float) -> list[dict]:
    url = ATP_RANKING_HISTORY_URL.format(player_id=player_id)
    response = scraper.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    rows = []
    for item in payload.get("History", []):
        rank_date = parse_date(item.get("RankDate"))
        if rank_date is None:
            continue
        rows.append(
            {
                "player": player,
                "player_id": player_id,
                "rank_date": rank_date.isoformat(),
                "singles_rank": item.get("SglRollRank") or "",
                "singles_points": item.get("SglRollPoints") or "",
                "race_rank": item.get("SglRaceRank") or "",
                "race_points": item.get("SglRacePoints") or "",
            }
        )
    time.sleep(delay)
    return rows


def rank_as_of(history: list[dict], player: str, match_date: date) -> dict[str, int | float | str | None]:
    player_rows = [row for row in history if row["player"] == player and row.get("singles_rank")]
    dated = sorted((parse_date(row["rank_date"]), row) for row in player_rows)
    dates = [item[0] for item in dated]
    index = bisect_right(dates, match_date) - 1
    if index < 0:
        return {"rank": None, "points": None, "rank_date": ""}
    row = dated[index][1]
    rank = int(row["singles_rank"])
    points = int(row["singles_points"]) if row.get("singles_points") else None
    return {
        "rank": rank,
        "points": points,
        "rank_date": row["rank_date"],
        "log_rank": math.log(rank),
    }


def build_match_features(history: list[dict], upcoming_path: Path) -> list[dict]:
    matches = read_csv(upcoming_path)
    players = set(POC_PLAYERS)
    rows = []
    for match in matches:
        player1 = match.get("player1", "")
        player2 = match.get("player2", "")
        if player1 not in players and player2 not in players:
            continue
        match_date = parse_date(match.get("start") or match.get("date") or "2026-04-28")
        if match_date is None:
            continue
        rank1 = rank_as_of(history, player1, match_date)
        rank2 = rank_as_of(history, player2, match_date)
        rows.append(
            {
                "match_id": match.get("match_id", ""),
                "fecha": match_date.isoformat(),
                "jugador_1": player1,
                "jugador_2": player2,
                "ranking_jugador_1": rank1["rank"],
                "ranking_jugador_2": rank2["rank"],
                "ranking_fecha_jugador_1": rank1["rank_date"],
                "ranking_fecha_jugador_2": rank2["rank_date"],
                "puntos_ranking_jugador_1": rank1["points"],
                "puntos_ranking_jugador_2": rank2["points"],
                "diferencia_ranking": (rank2["rank"] - rank1["rank"]) if rank1["rank"] and rank2["rank"] else "",
                "diferencia_log_ranking": (rank2["log_rank"] - rank1["log_rank"]) if rank1["rank"] and rank2["rank"] else "",
                "diferencia_puntos_ranking": (rank1["points"] - rank2["points"]) if rank1["points"] and rank2["points"] else "",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="POC for ATP ranking history scraping.")
    parser.add_argument("--out-dir", default="files/processed/atp_rankings_poc")
    parser.add_argument("--upcoming", default="files/processed/atp_2026/upcoming_matches.csv")
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    scraper = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "windows", "mobile": False})
    id_rows = []
    history_rows: list[dict] = []
    for player in POC_PLAYERS:
        player_id = resolve_player_id(scraper, player)
        id_rows.append({"player": player, "player_id": player_id or ""})
        if not player_id:
            print(f"[ranking] {player}: player_id not found")
            continue
        rows = scrape_rank_history(scraper, player, player_id, args.delay)
        print(f"[ranking] {player} ({player_id}): {len(rows)} rows")
        history_rows.extend(rows)

    write_csv(out_dir / "player_ids_poc.csv", id_rows)
    history_path = out_dir / "player_ranking_history.csv"
    features_path = out_dir / "ranking_features_poc.csv"
    write_csv(history_path, history_rows)
    feature_rows = build_match_features(history_rows, Path(args.upcoming))
    write_csv(features_path, feature_rows)
    print(f"[write] {history_path} ({len(history_rows)} rows)")
    print(f"[write] {features_path} ({len(feature_rows)} rows)")


if __name__ == "__main__":
    main()
