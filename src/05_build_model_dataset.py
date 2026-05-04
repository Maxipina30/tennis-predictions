from __future__ import annotations

import argparse
import csv
import json
import math
import re
from bisect import bisect_right
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
UNSEEDED_SEED_VALUE = 64
MISSING_RANK_VALUE = 10000
MISSING_POINTS_VALUE = 0


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


def days_between(start: datetime | None, end: datetime | None) -> int | None:
    if not start or not end:
        return None
    return max((end.date() - start.date()).days, 0)


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
    by_category: dict[str, "PlayerState"] = field(default_factory=dict)
    by_year_surface: dict[tuple[int, str], "PlayerState"] = field(default_factory=dict)
    recent: deque[dict] = field(default_factory=lambda: deque(maxlen=20))
    last_match_date: datetime | None = None
    tournament_rounds: dict[tuple[int, str], int] = field(default_factory=dict)

    def child_year(self, year: int) -> "PlayerState":
        return self.by_year.setdefault(year, PlayerState())

    def child_surface(self, surface: str) -> "PlayerState":
        return self.by_surface.setdefault(surface, PlayerState())

    def child_category(self, category: str) -> "PlayerState":
        return self.by_category.setdefault(category, PlayerState())

    def child_year_surface(self, year: int, surface: str) -> "PlayerState":
        return self.by_year_surface.setdefault((year, surface), PlayerState())

    def snapshot_core(self, prefix: str) -> dict:
        recent_5 = list(self.recent)[-5:]
        recent_10 = list(self.recent)[-10:]
        return {
            f"{prefix}_partidos_previos": self.matches,
            f"{prefix}_log_partidos_previos": math.log1p(self.matches),
            f"{prefix}_porcentaje_victorias_previas": rate(self.wins, self.matches),
            f"{prefix}_porcentaje_victorias_ponderado_nivel": rate(self.weighted_wins, self.weighted_matches),
            f"{prefix}_margen_promedio_games": rate(self.games_for - self.games_against, self.matches),
            f"{prefix}_margen_promedio_sets": rate(self.sets_for - self.sets_against, self.matches),
            f"{prefix}_porcentaje_tiebreaks_ganados": rate(self.tiebreaks_won, self.tiebreaks_played),
            f"{prefix}_tiebreaks_previos": self.tiebreaks_played,
            f"{prefix}_porcentaje_victorias_sets_corridos": rate(self.straight_set_wins, self.matches),
            f"{prefix}_porcentaje_derrotas_sets_corridos": rate(self.straight_set_losses, self.matches),
            f"{prefix}_porcentaje_victorias_ultimos_5": rate(sum(r["won"] for r in recent_5), len(recent_5)),
            f"{prefix}_porcentaje_victorias_ultimos_10": rate(sum(r["won"] for r in recent_10), len(recent_10)),
            f"{prefix}_margen_games_ultimos_10": (
                sum(r["game_diff"] for r in recent_10) / len(recent_10) if recent_10 else None
            ),
        }

    def snapshot(self, prefix: str, match_date: datetime, year: int, surface: str, category: str, tournament: str) -> dict:
        row = self.snapshot_core(prefix)
        year_state = self.by_year.get(year, PlayerState())
        surface_state = self.by_surface.get(surface, PlayerState())
        category_state = self.by_category.get(category, PlayerState())
        year_surface_state = self.by_year_surface.get((year, surface), PlayerState())
        row.update(year_state.snapshot_core(f"{prefix}_ano_actual"))
        row.update(surface_state.snapshot_core(f"{prefix}_superficie"))
        row.update(category_state.snapshot_core(f"{prefix}_categoria_torneo"))
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
        row[f"{prefix}_jugo_mismo_torneo_ano_anterior"] = int(prior_round is not None)
        row[f"{prefix}_ronda_mismo_torneo_ano_anterior"] = prior_round or 0
        row[f"{prefix}_defendia_titulo"] = int(prior_round == ROUND_ORDER["F"])
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

        for child in (self.child_year(year), self.child_surface(surface), self.child_category(category), self.child_year_surface(year, surface)):
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


def build_injury_index(injury_rows: list[dict]) -> dict[str, list[dict]]:
    injuries_by_player: dict[str, list[dict]] = defaultdict(list)
    for row in injury_rows:
        player_key = player_key_from_url(row.get("player_url"))
        start = parse_date(row.get("start_date"))
        if not player_key or not start:
            continue
        end = parse_date(row.get("end_date"))
        duration = to_int(row.get("duration_days"))
        if duration is None:
            duration = days_between(start, end)
        injuries_by_player[player_key].append(
            {
                "start": start,
                "end": end,
                "duration_days": duration,
                "reason": row.get("reason") or "",
            }
        )
    for injuries in injuries_by_player.values():
        injuries.sort(key=lambda injury: (injury["start"], injury["reason"]))
    return injuries_by_player


def injury_snapshot(injuries: list[dict], prefix: str, match_date: datetime) -> dict:
    prior = [injury for injury in injuries if injury["start"].date() < match_date.date()]
    current_year = [injury for injury in prior if injury["start"].year == match_date.year]
    completed_current_year = [injury for injury in current_year if injury.get("duration_days") is not None]
    active = [
        injury
        for injury in prior
        if injury["start"].date() < match_date.date()
        and (injury.get("end") is None or injury["end"].date() >= match_date.date())
    ]
    last_injury = prior[-1] if prior else None
    last_reference_date = None
    if last_injury:
        last_reference_date = last_injury.get("end") or last_injury["start"]

    return {
        f"{prefix}_lesiones_ano_actual": len(current_year),
        f"{prefix}_dias_lesionado_ano_actual": sum(injury["duration_days"] for injury in completed_current_year),
        f"{prefix}_lesion_abierta": int(bool(active)),
        f"{prefix}_dias_desde_ultima_lesion": (
            max((match_date.date() - last_reference_date.date()).days, 0) if last_reference_date else None
        ),
    }


def build_ranking_index(ranking_rows: list[dict]) -> dict[str, list[tuple[datetime, dict]]]:
    rankings_by_player: dict[str, list[tuple[datetime, dict]]] = defaultdict(list)
    for row in ranking_rows:
        player_key = row.get("player_key") or player_key_from_url(row.get("player_url"))
        rank_date = parse_date(row.get("rank_date"))
        if not player_key or not rank_date or not row.get("singles_rank"):
            continue
        rankings_by_player[player_key].append((rank_date, row))
    for rankings in rankings_by_player.values():
        rankings.sort(key=lambda item: item[0])
    return rankings_by_player


def ranking_as_of(rankings: list[tuple[datetime, dict]], match_date: datetime) -> dict:
    if not rankings:
        return {
            "rank": MISSING_RANK_VALUE,
            "points": MISSING_POINTS_VALUE,
            "race_rank": MISSING_RANK_VALUE,
            "race_points": MISSING_POINTS_VALUE,
            "rank_date": "",
            "log_rank": math.log(MISSING_RANK_VALUE),
            "has_ranking": 0,
        }
    dates = [item[0].date() for item in rankings]
    index = bisect_right(dates, match_date.date()) - 1
    if index < 0:
        return {
            "rank": MISSING_RANK_VALUE,
            "points": MISSING_POINTS_VALUE,
            "race_rank": MISSING_RANK_VALUE,
            "race_points": MISSING_POINTS_VALUE,
            "rank_date": "",
            "log_rank": math.log(MISSING_RANK_VALUE),
            "has_ranking": 0,
        }
    rank_date, row = rankings[index]
    rank = to_int(row.get("singles_rank"))
    if rank is None:
        rank = MISSING_RANK_VALUE
    return {
        "rank": rank,
        "points": to_int(row.get("singles_points")) or MISSING_POINTS_VALUE,
        "race_rank": to_int(row.get("race_rank")) or MISSING_RANK_VALUE,
        "race_points": to_int(row.get("race_points")) or MISSING_POINTS_VALUE,
        "rank_date": rank_date.date().isoformat(),
        "log_rank": math.log(rank) if rank and rank > 0 else None,
        "has_ranking": 1,
    }


def ranking_snapshot(
    rankings_by_player: dict[str, list[tuple[datetime, dict]]],
    player_key: str,
    prefix: str,
    match_date: datetime,
) -> dict:
    ranking = ranking_as_of(rankings_by_player.get(player_key, []), match_date)
    return {
        f"{prefix}_ranking": ranking["rank"],
        f"{prefix}_ranking_log": ranking["log_rank"],
        f"{prefix}_puntos_ranking": ranking["points"],
        f"{prefix}_ranking_race": ranking["race_rank"],
        f"{prefix}_puntos_race": ranking["race_points"],
        f"{prefix}_ranking_fecha": ranking["rank_date"],
        f"{prefix}_tiene_ranking": ranking["has_ranking"],
    }


def build_rows_for_events(
    target_events: list[dict],
    history_events: list[dict],
    injury_rows: list[dict] | None = None,
    ranking_rows: list[dict] | None = None,
) -> list[dict]:
    states: dict[str, PlayerState] = defaultdict(PlayerState)
    injuries_by_player = build_injury_index(injury_rows or [])
    rankings_by_player = build_ranking_index(ranking_rows or [])
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
            "jugador_1_tiene_sembrado": int(event["seed1"] is not None),
            "jugador_2_tiene_sembrado": int(event["seed2"] is not None),
            "diferencia_tiene_sembrado": int(event["seed1"] is not None) - int(event["seed2"] is not None),
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
        row.update(states[p1].snapshot("jugador_1", event["date"], event["date"].year, event["surface"], event["category"], event["tournament"]))
        row.update(states[p2].snapshot("jugador_2", event["date"], event["date"].year, event["surface"], event["category"], event["tournament"]))
        row.update(injury_snapshot(injuries_by_player.get(p1, []), "jugador_1", event["date"]))
        row.update(injury_snapshot(injuries_by_player.get(p2, []), "jugador_2", event["date"]))
        row.update(ranking_snapshot(rankings_by_player, p1, "jugador_1", event["date"]))
        row.update(ranking_snapshot(rankings_by_player, p2, "jugador_2", event["date"]))
        diff_columns(
            row,
            "jugador_1",
            "jugador_2",
            [
                "ranking",
                "ranking_log",
                "puntos_ranking",
                "ranking_race",
                "puntos_race",
                "tiene_ranking",
            ],
        )
        diff_columns(
            row,
            "jugador_1",
            "jugador_2",
            [
                "porcentaje_victorias_previas",
                "partidos_previos",
                "log_partidos_previos",
                "porcentaje_victorias_ponderado_nivel",
                "margen_promedio_games",
                "margen_promedio_sets",
                "porcentaje_tiebreaks_ganados",
                "tiebreaks_previos",
                "porcentaje_victorias_ultimos_10",
                "margen_games_ultimos_10",
                "porcentaje_victorias_sets_corridos",
                "porcentaje_derrotas_sets_corridos",
                "dias_descanso",
                "games_ultimos_7_dias",
                "jugo_mismo_torneo_ano_anterior",
                "ronda_mismo_torneo_ano_anterior",
                "superficie_porcentaje_victorias_previas",
                "superficie_partidos_previos",
                "superficie_log_partidos_previos",
                "categoria_torneo_porcentaje_victorias_previas",
                "categoria_torneo_partidos_previos",
                "categoria_torneo_log_partidos_previos",
                "categoria_torneo_margen_promedio_games",
                "categoria_torneo_margen_promedio_sets",
                "categoria_torneo_porcentaje_victorias_ultimos_10",
                "ano_actual_porcentaje_victorias_previas",
                "ano_actual_partidos_previos",
                "ano_actual_log_partidos_previos",
                "ano_actual_superficie_porcentaje_victorias_previas",
                "ano_actual_superficie_partidos_previos",
                "ano_actual_superficie_log_partidos_previos",
                "lesiones_ano_actual",
                "dias_lesionado_ano_actual",
                "lesion_abierta",
                "dias_desde_ultima_lesion",
            ],
        )
        rows.append(row)
    return rows


def build_dataset(
    matches_path: Path,
    histories_path: Path,
    output_path: Path,
    injuries_path: Path | None = None,
    rankings_path: Path | None = None,
) -> None:
    matches = read_csv(matches_path)
    player_matches = read_csv(histories_path)
    injuries = read_csv(injuries_path) if injuries_path else []
    rankings = read_csv(rankings_path) if rankings_path else []
    tournament_categories = {row.get("source_url", ""): row.get("category", "") for row in matches}
    history_events = build_history_events(player_matches, tournament_categories)
    target_events = build_target_events(matches)

    rows = build_rows_for_events(target_events, history_events, injuries, rankings)

    write_csv(output_path, rows)
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "matches": len(matches),
                "history_rows": len(player_matches),
                "history_events": len(history_events),
                "injury_rows": len(injuries),
                "ranking_rows": len(rankings),
                "model_rows": len(rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def seed_diff(seed1: int | None, seed2: int | None) -> int | None:
    if seed1 is None and seed2 is None:
        return None
    value1 = min(seed1, UNSEEDED_SEED_VALUE) if seed1 is not None else UNSEEDED_SEED_VALUE
    value2 = min(seed2, UNSEEDED_SEED_VALUE) if seed2 is not None else UNSEEDED_SEED_VALUE
    return value2 - value1


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chronological no-leakage model dataset.")
    parser.add_argument("--matches", default="files/processed/atp_2026/matches.csv")
    parser.add_argument("--histories", default="files/processed/player_histories_2024_2026/player_matches.csv")
    parser.add_argument("--injuries", default="files/processed/player_histories_2024_2026/player_injuries.csv")
    parser.add_argument("--rankings", default="files/processed/atp_rankings/player_ranking_history.csv")
    parser.add_argument("--out", default="files/processed/model_dataset_2026/model_dataset.csv")
    args = parser.parse_args()
    injuries_path = Path(args.injuries)
    rankings_path = Path(args.rankings)
    build_dataset(
        Path(args.matches),
        Path(args.histories),
        Path(args.out),
        injuries_path if injuries_path.exists() else None,
        rankings_path if rankings_path.exists() else None,
    )


if __name__ == "__main__":
    main()
