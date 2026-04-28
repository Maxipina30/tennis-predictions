from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROUND_ORDER = {
    "1R": 1,
    "2R": 2,
    "3R": 3,
    "R16": 4,
    "QF": 5,
    "SF": 6,
    "F": 7,
}
LEVEL_WEIGHTS = {
    "grand_slam": 2000.0,
    "masters_1000": 1000.0,
    "atp_500": 500.0,
    "atp_250": 250.0,
    "challenger": 100.0,
    "itf": 25.0,
    "unknown": 100.0,
}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


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


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def to_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def to_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        return None


def rate(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return num / den


def player_key_from_url(url: str | None) -> str:
    if not url:
        return ""
    parts = [part for part in url.rstrip("/").split("/") if part]
    return parts[-1] if parts else ""


def normalize_surface(value: str | None) -> str:
    value = (value or "unknown").lower().replace(" ", "_")
    if value in {"clay", "hard", "grass", "indoors"}:
        return value
    return "unknown"


def infer_category(tournament_url: str | None, category_lookup: dict[str, str]) -> str:
    if tournament_url and tournament_url in category_lookup:
        return category_lookup[tournament_url]
    text = (tournament_url or "").lower()
    if "challenger" in text:
        return "challenger"
    if "itf" in text or "futures" in text:
        return "itf"
    return "unknown"


def parse_score(score: str | None) -> dict:
    if not score:
        return {"sets1": None, "sets2": None, "games1": 0, "games2": 0, "tb1": 0, "tb2": 0}
    sets1 = sets2 = games1 = games2 = tb1 = tb2 = 0
    for raw_set in score.split(","):
        part = raw_set.strip()
        match = re.match(r"^(\d+)(?:\((\d+)\))?-(\d+)(?:\((\d+)\))?", part)
        if not match:
            continue
        g1 = int(match.group(1))
        t1 = match.group(2)
        g2 = int(match.group(3))
        t2 = match.group(4)
        games1 += g1
        games2 += g2
        if g1 > g2:
            sets1 += 1
        elif g2 > g1:
            sets2 += 1
        if (g1, g2) in {(7, 6), (6, 7)}:
            if g1 > g2:
                tb1 += 1
            else:
                tb2 += 1
        elif t1 is not None or t2 is not None:
            if g1 > g2:
                tb1 += 1
            else:
                tb2 += 1
    return {"sets1": sets1, "sets2": sets2, "games1": games1, "games2": games2, "tb1": tb1, "tb2": tb2}


def score_from_tournament_row(row: dict) -> dict:
    return {
        "sets1": to_int(row.get("sets1")),
        "sets2": to_int(row.get("sets2")),
        "games1": to_int(row.get("games1")) or 0,
        "games2": to_int(row.get("games2")) or 0,
        "tb1": count_tiebreaks_from_json(row.get("score1_json"), row.get("score2_json"), 1),
        "tb2": count_tiebreaks_from_json(row.get("score1_json"), row.get("score2_json"), 2),
    }


def count_tiebreaks_from_json(score1_json: str | None, score2_json: str | None, side: int) -> int:
    try:
        score1 = json.loads(score1_json or "[]")
        score2 = json.loads(score2_json or "[]")
    except json.JSONDecodeError:
        return 0
    count = 0
    for s1, s2 in zip(score1, score2):
        g1 = s1.get("games")
        g2 = s2.get("games")
        if (g1, g2) == (7, 6) and side == 1:
            count += 1
        elif (g1, g2) == (6, 7) and side == 2:
            count += 1
    return count


@dataclass
class PlayerState:
    matches: int = 0
    wins: int = 0
    games_for: int = 0
    games_against: int = 0
    sets_for: int = 0
    sets_against: int = 0
    weighted_matches: float = 0.0
    weighted_wins: float = 0.0
    tiebreaks_played: int = 0
    tiebreaks_won: int = 0
    straight_set_wins: int = 0
    straight_set_losses: int = 0
    by_year: dict[int, "PlayerState"] = field(default_factory=dict)
    by_surface: dict[str, "PlayerState"] = field(default_factory=dict)
    by_year_surface: dict[tuple[int, str], "PlayerState"] = field(default_factory=dict)
    recent: deque[dict] = field(default_factory=lambda: deque(maxlen=20))
    last_match_date: datetime | None = None
    tournament_rounds: dict[tuple[int, str], int] = field(default_factory=dict)

    def child_year(self, year: int) -> "PlayerState":
        return self.by_year.setdefault(year, PlayerState())

    def child_surface(self, surface: str) -> "PlayerState":
        return self.by_surface.setdefault(surface, PlayerState())

    def child_year_surface(self, year: int, surface: str) -> "PlayerState":
        return self.by_year_surface.setdefault((year, surface), PlayerState())

    def snapshot_core(self, prefix: str) -> dict:
        recent_5 = list(self.recent)[-5:]
        recent_10 = list(self.recent)[-10:]
        return {
            f"{prefix}_partidos_previos": self.matches,
            f"{prefix}_porcentaje_victorias_previas": rate(self.wins, self.matches),
            f"{prefix}_porcentaje_victorias_ponderado_nivel": rate(self.weighted_wins, self.weighted_matches),
            f"{prefix}_diferencia_promedio_games": rate(self.games_for - self.games_against, self.matches),
            f"{prefix}_diferencia_promedio_sets": rate(self.sets_for - self.sets_against, self.matches),
            f"{prefix}_porcentaje_tiebreaks_ganados": rate(self.tiebreaks_won, self.tiebreaks_played),
            f"{prefix}_tiebreaks_previos": self.tiebreaks_played,
            f"{prefix}_porcentaje_victorias_sets_corridos": rate(self.straight_set_wins, self.matches),
            f"{prefix}_porcentaje_derrotas_sets_corridos": rate(self.straight_set_losses, self.matches),
            f"{prefix}_porcentaje_victorias_ultimos_5": rate(sum(r["won"] for r in recent_5), len(recent_5)),
            f"{prefix}_porcentaje_victorias_ultimos_10": rate(sum(r["won"] for r in recent_10), len(recent_10)),
            f"{prefix}_diferencia_games_ultimos_10": (
                sum(r["game_diff"] for r in recent_10) / len(recent_10) if recent_10 else None
            ),
        }

    def snapshot(self, prefix: str, match_date: datetime, year: int, surface: str, tournament: str) -> dict:
        row = self.snapshot_core(prefix)
        year_state = self.by_year.get(year, PlayerState())
        surface_state = self.by_surface.get(surface, PlayerState())
        year_surface_state = self.by_year_surface.get((year, surface), PlayerState())
        row.update(year_state.snapshot_core(f"{prefix}_ano_actual"))
        row.update(surface_state.snapshot_core(f"{prefix}_superficie"))
        row.update(year_surface_state.snapshot_core(f"{prefix}_ano_actual_superficie"))

        if self.last_match_date:
            row[f"{prefix}_dias_descanso"] = max((match_date - self.last_match_date).days, 0)
        else:
            row[f"{prefix}_dias_descanso"] = None

        recent_7 = [r for r in self.recent if (match_date - r["date"]).days <= 7]
        row[f"{prefix}_partidos_ultimos_7_dias"] = len(recent_7)
        row[f"{prefix}_games_ultimos_7_dias"] = sum(r["games"] for r in recent_7)
        row[f"{prefix}_sets_ultimos_7_dias"] = sum(r["sets"] for r in recent_7)

        prior_round = self.tournament_rounds.get((year - 1, tournament))
        row[f"{prefix}_ronda_mismo_torneo_ano_anterior"] = prior_round
        row[f"{prefix}_defendia_titulo"] = int(prior_round == ROUND_ORDER["F"]) if prior_round is not None else None
        return row

    def update(self, event: dict, side: int) -> None:
        date = event["date"]
        year = date.year
        surface = event["surface"]
        tournament = event["tournament"]
        category = event["category"]
        weight = LEVEL_WEIGHTS.get(category, LEVEL_WEIGHTS["unknown"])
        won = event["winner_side"] == side
        games_for = event["games1"] if side == 1 else event["games2"]
        games_against = event["games2"] if side == 1 else event["games1"]
        sets_for = event["sets1"] if side == 1 else event["sets2"]
        sets_against = event["sets2"] if side == 1 else event["sets1"]
        tb_won = event["tb1"] if side == 1 else event["tb2"]
        tb_lost = event["tb2"] if side == 1 else event["tb1"]

        self.matches += 1
        self.wins += int(won)
        self.games_for += games_for
        self.games_against += games_against
        self.sets_for += sets_for
        self.sets_against += sets_against
        self.weighted_matches += weight
        self.weighted_wins += weight * int(won)
        self.tiebreaks_played += tb_won + tb_lost
        self.tiebreaks_won += tb_won
        self.straight_set_wins += int(won and sets_against == 0)
        self.straight_set_losses += int((not won) and sets_for == 0)
        self.recent.append({"date": date, "won": int(won), "game_diff": games_for - games_against, "games": games_for + games_against, "sets": sets_for + sets_against})
        self.last_match_date = date
        round_value = ROUND_ORDER.get(event.get("round") or "", 0)
        key = (year, tournament)
        self.tournament_rounds[key] = max(self.tournament_rounds.get(key, 0), round_value)

        for child in (self.child_year(year), self.child_surface(surface), self.child_year_surface(year, surface)):
            child.update_without_children(event, side, weight, won, games_for, games_against, sets_for, sets_against, tb_won, tb_lost)

    def update_without_children(self, event: dict, side: int, weight: float, won: bool, games_for: int, games_against: int, sets_for: int, sets_against: int, tb_won: int, tb_lost: int) -> None:
        self.matches += 1
        self.wins += int(won)
        self.games_for += games_for
        self.games_against += games_against
        self.sets_for += sets_for
        self.sets_against += sets_against
        self.weighted_matches += weight
        self.weighted_wins += weight * int(won)
        self.tiebreaks_played += tb_won + tb_lost
        self.tiebreaks_won += tb_won
        self.straight_set_wins += int(won and sets_against == 0)
        self.straight_set_losses += int((not won) and sets_for == 0)


def build_history_events(player_matches: list[dict], tournament_categories: dict[str, str]) -> list[dict]:
    events_by_id: dict[str, dict] = {}
    for row in player_matches:
        date = parse_date(row.get("date_iso"))
        if not date:
            continue
        score = parse_score(row.get("score_raw"))
        if score["sets1"] is None or score["sets2"] is None or score["sets1"] == score["sets2"]:
            continue
        match_id = row.get("match_id") or f"{row.get('player1_url')}|{row.get('player2_url')}|{row.get('date_iso')}|{row.get('score_raw')}"
        winner_side = 1 if score["sets1"] > score["sets2"] else 2
        event = {
            "match_id": match_id,
            "date": date,
            "tournament": row.get("tournament") or "",
            "tournament_url": row.get("tournament_url") or "",
            "category": infer_category(row.get("tournament_url"), tournament_categories),
            "surface": normalize_surface(row.get("surface")),
            "round": row.get("round") or "",
            "player1_key": player_key_from_url(row.get("player1_url")),
            "player2_key": player_key_from_url(row.get("player2_url")),
            "sets1": score["sets1"],
            "sets2": score["sets2"],
            "games1": score["games1"],
            "games2": score["games2"],
            "tb1": score["tb1"],
            "tb2": score["tb2"],
            "winner_side": winner_side,
        }
        events_by_id[match_id] = event
    return sorted(events_by_id.values(), key=lambda event: (event["date"], event["match_id"]))


def build_target_events(matches: list[dict]) -> list[dict]:
    events: list[dict] = []
    for row in matches:
        date = parse_date(row.get("start"))
        if not date:
            continue
        score = score_from_tournament_row(row)
        if score["sets1"] is None or score["sets2"] is None or score["sets1"] == score["sets2"]:
            continue
        events.append(
            {
                "match_id": row.get("match_id"),
                "date": date,
                "tournament": row.get("tournament") or "",
                "category": row.get("category") or infer_category(row.get("source_url"), {}),
                "surface": normalize_surface(row.get("surface")),
                "round": row.get("round") or "",
                "player1": row.get("player1"),
                "player2": row.get("player2"),
                "player1_key": player_key_from_url(row.get("player1_url")),
                "player2_key": player_key_from_url(row.get("player2_url")),
                "seed1": to_int(row.get("seed1")),
                "seed2": to_int(row.get("seed2")),
                "sets1": score["sets1"],
                "sets2": score["sets2"],
                "games1": score["games1"],
                "games2": score["games2"],
                "tb1": score["tb1"],
                "tb2": score["tb2"],
                "winner_side": 1 if score["sets1"] > score["sets2"] else 2,
            }
        )
    return sorted(events, key=lambda event: (event["date"], event["match_id"] or ""))


def build_upcoming_events(upcoming: list[dict], fallback_date: str, name_to_key: dict[str, str] | None = None) -> list[dict]:
    fallback = parse_date(fallback_date) or datetime.today()
    events: list[dict] = []
    for row in upcoming:
        start_raw = row.get("start_raw") or ""
        event_date = fallback
        time_match = re.search(r"(\d{1,2}):(\d{2})", start_raw)
        if time_match:
            event_date = event_date.replace(hour=int(time_match.group(1)), minute=int(time_match.group(2)))
        events.append(
            {
                "match_id": row.get("match_id"),
                "date": event_date,
                "tournament": row.get("tournament") or "",
                "category": row.get("category") or infer_category(row.get("source_url"), {}),
                "surface": normalize_surface(row.get("surface")),
                "round": row.get("round") or "",
                "player1": row.get("player1"),
                "player2": row.get("player2"),
                "player1_key": lookup_player_key(row.get("player1"), name_to_key),
                "player2_key": lookup_player_key(row.get("player2"), name_to_key),
                "seed1": to_int(row.get("seed1")),
                "seed2": to_int(row.get("seed2")),
                "sets1": None,
                "sets2": None,
                "games1": None,
                "games2": None,
                "tb1": None,
                "tb2": None,
                "winner_side": None,
            }
        )
    return sorted(events, key=lambda event: (event["date"], event["match_id"] or ""))


def lookup_player_key(value: str | None, name_to_key: dict[str, str] | None) -> str:
    if not value:
        return ""
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    if name_to_key and normalized in name_to_key:
        return name_to_key[normalized]
    return player_key_from_name(value)


def player_key_from_name(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def diff_columns(row: dict, left: str, right: str, names: list[str]) -> None:
    for name in names:
        a = to_float(row.get(f"{left}_{name}"))
        b = to_float(row.get(f"{right}_{name}"))
        row[f"diferencia_{name}"] = None if a is None or b is None else a - b


def build_rows_for_events(target_events: list[dict], history_events: list[dict]) -> list[dict]:
    states: dict[str, PlayerState] = defaultdict(PlayerState)
    h2h: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    history_index = 0
    rows: list[dict] = []
    for event in target_events:
        # Player-profile histories only have day-level dates. To avoid leakage, never
        # let matches from the same calendar day feed the pre-match snapshot.
        while history_index < len(history_events) and history_events[history_index]["date"].date() < event["date"].date():
            past = history_events[history_index]
            if past["player1_key"]:
                states[past["player1_key"]].update(past, 1)
            if past["player2_key"]:
                states[past["player2_key"]].update(past, 2)
            pair_key = tuple(sorted((past["player1_key"], past["player2_key"])))
            winner_key = past["player1_key"] if past["winner_side"] == 1 else past["player2_key"]
            h2h[pair_key][winner_key] += 1
            history_index += 1

        p1 = event["player1_key"]
        p2 = event["player2_key"]
        pair_key = tuple(sorted((p1, p2)))
        prior_h2h = h2h[pair_key]
        row = {
            "match_id": event["match_id"],
            "fecha": event["date"].date().isoformat(),
            "torneo": event["tournament"],
            "categoria_torneo": event["category"],
            "superficie": event["surface"],
            "ronda": event["round"],
            "jugador_1": event["player1"],
            "jugador_2": event["player2"],
            "sembrado_jugador_1": event["seed1"],
            "sembrado_jugador_2": event["seed2"],
            "diferencia_sembrado": seed_diff(event["seed1"], event["seed2"]),
            "partidos_previos_entre_ellos": prior_h2h[p1] + prior_h2h[p2],
            "victorias_previas_jugador_1_vs_jugador_2": prior_h2h[p1],
            "victorias_previas_jugador_2_vs_jugador_1": prior_h2h[p2],
            "diferencia_h2h_previo": prior_h2h[p1] - prior_h2h[p2],
            "target_gana_jugador_1": None if event["winner_side"] is None else int(event["winner_side"] == 1),
            "sets_jugador_1": event["sets1"],
            "sets_jugador_2": event["sets2"],
            "games_jugador_1": event["games1"],
            "games_jugador_2": event["games2"],
        }
        row.update(states[p1].snapshot("jugador_1", event["date"], event["date"].year, event["surface"], event["tournament"]))
        row.update(states[p2].snapshot("jugador_2", event["date"], event["date"].year, event["surface"], event["tournament"]))
        diff_columns(
            row,
            "jugador_1",
            "jugador_2",
            [
                "porcentaje_victorias_previas",
                "porcentaje_victorias_ponderado_nivel",
                "diferencia_promedio_games",
                "diferencia_promedio_sets",
                "porcentaje_tiebreaks_ganados",
                "porcentaje_victorias_ultimos_10",
                "diferencia_games_ultimos_10",
                "porcentaje_victorias_sets_corridos",
                "porcentaje_derrotas_sets_corridos",
                "dias_descanso",
                "games_ultimos_7_dias",
                "ronda_mismo_torneo_ano_anterior",
                "superficie_porcentaje_victorias_previas",
                "ano_actual_porcentaje_victorias_previas",
                "ano_actual_superficie_porcentaje_victorias_previas",
            ],
        )
        rows.append(row)
    return rows


def build_dataset(matches_path: Path, histories_path: Path, output_path: Path) -> None:
    matches = read_csv(matches_path)
    player_matches = read_csv(histories_path)
    tournament_categories = {row.get("source_url", ""): row.get("category", "") for row in matches}
    history_events = build_history_events(player_matches, tournament_categories)
    target_events = build_target_events(matches)

    rows = build_rows_for_events(target_events, history_events)

    write_csv(output_path, rows)
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "matches": len(matches),
                "history_rows": len(player_matches),
                "history_events": len(history_events),
                "model_rows": len(rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def seed_diff(seed1: int | None, seed2: int | None) -> int | None:
    if seed1 is None and seed2 is None:
        return None
    value1 = seed1 if seed1 is not None else 999
    value2 = seed2 if seed2 is not None else 999
    return value2 - value1


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chronological no-leakage model dataset.")
    parser.add_argument("--matches", default="files/processed/atp_2026/matches.csv")
    parser.add_argument("--histories", default="files/processed/player_histories_2024_2026/player_matches.csv")
    parser.add_argument("--out", default="files/processed/model_dataset_2026/model_dataset.csv")
    args = parser.parse_args()
    build_dataset(Path(args.matches), Path(args.histories), Path(args.out))


if __name__ == "__main__":
    main()
