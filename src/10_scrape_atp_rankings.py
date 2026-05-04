from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from datetime import date, datetime
from pathlib import Path
from urllib.parse import quote

import cloudscraper


ATP_RANKING_HISTORY_URL = "https://www.atptour.com/es/-/www/rank/history/{player_id}"
ATP_PLAYER_SEARCH_URL = "https://www.atptour.com/es/-/www/players/find/byname/{query}/es"
MANUAL_ATP_IDS = {
    "basile-54647": ("b0vx", "Pierluigi Basile"),
    "bautista-agut": ("bd06", "Roberto Bautista Agut"),
    "bondioli": ("b0pe", "Federico Bondioli"),
    "carballes-baena": ("cf59", "Roberto Carballes Baena"),
    "carboni-ecd4c": ("c0ow", "Lorenzo Carboni"),
    "carreno-busta": ("cd85", "Pablo Carreno Busta"),
    "de-minaur": ("dh58", "Alex de Minaur"),
    "garin": ("gd64", "Cristian Garin"),
    "ramos-vinolas": ("r772", "Albert Ramos-Vinolas"),
    "travaglia": ("ta12", "Stefano Travaglia"),
    "vasami-a02e2": ("v0j1", "Jacopo Vasami"),
    "zayid": ("ac39", "Mubarak Shannan Zayid"),
}


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


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    value = value.split("T", 1)[0]
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def player_key_from_url(url: str | None) -> str:
    if not url:
        return ""
    parts = [part for part in url.rstrip("/").split("/") if part]
    return parts[-1] if parts else ""


def normalize_name(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def all_tokens_match(query: str, full_name: str) -> bool:
    query_tokens = set(normalize_name(query).split())
    name_tokens = set(normalize_name(full_name).split())
    return bool(query_tokens) and query_tokens.issubset(name_tokens)


def query_variants(player_name: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", player_name).strip()
    if not cleaned:
        return []
    tokens = cleaned.split()
    variants = [cleaned]
    if len(tokens) >= 2:
        variants.append(" ".join([tokens[-1], *tokens[:-1]]))
        variants.append(tokens[0])
    return list(dict.fromkeys(variants))


def load_players(profiles_path: Path) -> list[dict]:
    players = []
    for row in read_csv(profiles_path):
        player_url = row.get("player_url") or ""
        player_name = row.get("player") or ""
        if not player_url or not player_name:
            continue
        players.append(
            {
                "player_url": player_url,
                "player_key": player_key_from_url(player_url),
                "player_name": player_name,
            }
        )
    return sorted(players, key=lambda row: row["player_key"])


def resolve_player_id(scraper: cloudscraper.CloudScraper, player_name: str) -> tuple[str, str]:
    variants = query_variants(player_name)
    for query_index, query in enumerate(variants):
        response = scraper.get(ATP_PLAYER_SEARCH_URL.format(query=quote(query)), timeout=30)
        response.raise_for_status()
        candidates = response.json()
        active_candidates = [candidate for candidate in candidates if candidate.get("Active") == "A"]
        for candidate in active_candidates + candidates:
            full_name = f"{candidate.get('FirstName', '')} {candidate.get('LastName', '')}".strip()
            if all_tokens_match(player_name, full_name) or all_tokens_match(query, full_name):
                return str(candidate["PlayerId"]).lower(), full_name
        if candidates and query_index == 0 and len(normalize_name(player_name).split()) <= 1:
            candidate = candidates[0]
            full_name = f"{candidate.get('FirstName', '')} {candidate.get('LastName', '')}".strip()
            return str(candidate["PlayerId"]).lower(), full_name
    return "", ""


def scrape_rank_history(
    scraper: cloudscraper.CloudScraper,
    player: dict,
    atp_player_id: str,
    atp_player_name: str,
) -> list[dict]:
    response = scraper.get(ATP_RANKING_HISTORY_URL.format(player_id=atp_player_id), timeout=30)
    response.raise_for_status()
    payload = response.json()
    rows: list[dict] = []
    for item in payload.get("History", []):
        rank_date = parse_date(item.get("RankDate"))
        if rank_date is None:
            continue
        rows.append(
            {
                "player_url": player["player_url"],
                "player_key": player["player_key"],
                "player_name": player["player_name"],
                "atp_player_id": atp_player_id,
                "atp_player_name": atp_player_name,
                "rank_date": rank_date.isoformat(),
                "singles_rank": item.get("SglRollRank") or "",
                "singles_points": item.get("SglRollPoints") or "",
                "race_rank": item.get("SglRaceRank") or "",
                "race_points": item.get("SglRacePoints") or "",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape ATP singles ranking history for TennisExplorer players.")
    parser.add_argument("--profiles", type=Path, default=Path("files/processed/player_histories_2024_2026_extended/player_profiles.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("files/processed/atp_rankings"))
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--max-players", type=int, default=None)
    args = parser.parse_args()

    players = load_players(args.profiles)
    if args.max_players:
        players = players[: args.max_players]

    existing_history = read_csv(args.out_dir / "player_ranking_history.csv")
    existing_ids = read_csv(args.out_dir / "player_ids.csv")
    history_by_key = {row.get("player_key") for row in existing_history if row.get("player_key")}
    id_rows_by_key = {row.get("player_key"): row for row in existing_ids if row.get("player_key")}
    history_rows = list(existing_history)
    id_rows = list(existing_ids)

    scraper = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "windows", "mobile": False})
    for index, player in enumerate(players, start=1):
        if player["player_key"] in history_by_key:
            print(f"[ranking {index}/{len(players)}] cached {player['player_name']}")
            continue
        print(f"[ranking {index}/{len(players)}] {player['player_name']}")
        atp_player_id = id_rows_by_key.get(player["player_key"], {}).get("atp_player_id", "")
        atp_player_name = id_rows_by_key.get(player["player_key"], {}).get("atp_player_name", "")
        if not atp_player_id and player["player_key"] in MANUAL_ATP_IDS:
            atp_player_id, atp_player_name = MANUAL_ATP_IDS[player["player_key"]]
        status = "ok"
        error = ""
        try:
            if not atp_player_id:
                atp_player_id, atp_player_name = resolve_player_id(scraper, player["player_name"])
                time.sleep(args.delay)
            if not atp_player_id:
                status = "not_found"
                rows = []
            else:
                rows = scrape_rank_history(scraper, player, atp_player_id, atp_player_name)
                time.sleep(args.delay)
                history_rows.extend(rows)
                if rows:
                    history_by_key.add(player["player_key"])
        except Exception as exc:
            status = "error"
            error = str(exc)
            rows = []
        id_row = {
            **player,
            "atp_player_id": atp_player_id,
            "atp_player_name": atp_player_name,
            "status": status,
            "ranking_rows": len(rows),
            "error": error,
        }
        id_rows_by_key[player["player_key"]] = id_row
        id_rows = [row for row in id_rows if row.get("player_key") != player["player_key"]]
        id_rows.append(id_row)
        write_csv(args.out_dir / "player_ids.csv", id_rows)
        write_csv(args.out_dir / "player_ranking_history.csv", history_rows)

    summary = {
        "players": len(players),
        "resolved": sum(1 for row in id_rows if row.get("atp_player_id")),
        "history_players": len({row.get("player_key") for row in history_rows if row.get("player_key")}),
        "history_rows": len(history_rows),
    }
    (args.out_dir / "ranking_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
