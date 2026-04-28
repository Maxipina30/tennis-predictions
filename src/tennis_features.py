from __future__ import annotations

import csv
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable


SURFACES = ("clay", "hard", "indoors", "grass", "not_set")


def parse_start(value: str | None) -> datetime:
    if not value:
        return datetime.max
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.max


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


def as_int(value: object) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(str(value)))
    except ValueError:
        return 0


def as_float(value: object) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(str(value))
    except ValueError:
        return None


def safe_rate(wins: int, total: int) -> float | None:
    if total <= 0:
        return None
    return wins / total


@dataclass
class PlayerState:
    matches: int = 0
    wins: int = 0
    losses: int = 0
    games_for: int = 0
    games_against: int = 0
    sets_for: int = 0
    sets_against: int = 0
    straight_set_wins: int = 0
    straight_set_losses: int = 0
    surface_matches: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    surface_wins: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    recent_results: deque[int] = field(default_factory=lambda: deque(maxlen=10))
    recent_game_diffs: deque[int] = field(default_factory=lambda: deque(maxlen=10))

    def snapshot(self, prefix: str, surface: str) -> dict:
        recent_total = len(self.recent_results)
        recent_wins = sum(self.recent_results)
        surface_matches = self.surface_matches[surface]
        surface_wins = self.surface_wins[surface]
        return {
            f"{prefix}_prior_matches": self.matches,
            f"{prefix}_prior_wins": self.wins,
            f"{prefix}_prior_losses": self.losses,
            f"{prefix}_prior_win_rate": safe_rate(self.wins, self.matches),
            f"{prefix}_prior_surface_matches": surface_matches,
            f"{prefix}_prior_surface_wins": surface_wins,
            f"{prefix}_prior_surface_win_rate": safe_rate(surface_wins, surface_matches),
            f"{prefix}_prior_recent_matches": recent_total,
            f"{prefix}_prior_recent_win_rate": safe_rate(recent_wins, recent_total),
            f"{prefix}_prior_avg_games_for": safe_rate(self.games_for, self.matches),
            f"{prefix}_prior_avg_games_against": safe_rate(self.games_against, self.matches),
            f"{prefix}_prior_avg_game_diff": safe_rate(self.games_for - self.games_against, self.matches),
            f"{prefix}_prior_avg_set_diff": safe_rate(self.sets_for - self.sets_against, self.matches),
            f"{prefix}_prior_recent_avg_game_diff": (
                sum(self.recent_game_diffs) / len(self.recent_game_diffs)
                if self.recent_game_diffs
                else None
            ),
            f"{prefix}_prior_straight_set_win_rate": safe_rate(self.straight_set_wins, self.matches),
            f"{prefix}_prior_straight_set_loss_rate": safe_rate(self.straight_set_losses, self.matches),
        }

    def update(self, won: bool, surface: str, games_for: int, games_against: int, sets_for: int, sets_against: int) -> None:
        self.matches += 1
        self.wins += int(won)
        self.losses += int(not won)
        self.games_for += games_for
        self.games_against += games_against
        self.sets_for += sets_for
        self.sets_against += sets_against
        self.surface_matches[surface] += 1
        self.surface_wins[surface] += int(won)
        self.recent_results.append(int(won))
        self.recent_game_diffs.append(games_for - games_against)
        if won and sets_against == 0:
            self.straight_set_wins += 1
        if not won and sets_for == 0:
            self.straight_set_losses += 1


def normalize_surface(value: str | None) -> str:
    value = (value or "not_set").lower().replace(" ", "_")
    return value if value in SURFACES else "not_set"


def add_prefix_diff(row: dict, left_prefix: str, right_prefix: str, keys: list[str]) -> None:
    for key in keys:
        left = as_float(row.get(f"{left_prefix}_{key}"))
        right = as_float(row.get(f"{right_prefix}_{key}"))
        row[f"diff_{key}"] = None if left is None or right is None else left - right


def build_no_leakage_features(matches: list[dict]) -> list[dict]:
    states: dict[str, PlayerState] = defaultdict(PlayerState)
    h2h: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    feature_rows: list[dict] = []

    sorted_matches = sorted(matches, key=lambda row: (parse_start(row.get("start")), row.get("match_id") or ""))
    for match in sorted_matches:
        player1 = match.get("player1", "")
        player2 = match.get("player2", "")
        if not player1 or not player2:
            continue

        surface = normalize_surface(match.get("surface"))
        sets1 = as_int(match.get("sets1"))
        sets2 = as_int(match.get("sets2"))
        games1 = as_int(match.get("games1"))
        games2 = as_int(match.get("games2"))
        if sets1 == sets2:
            continue

        pair_key = tuple(sorted((player1, player2)))
        prior_h2h = h2h[pair_key]
        row = {
            "match_id": match.get("match_id"),
            "start": match.get("start"),
            "year": match.get("year"),
            "tournament": match.get("tournament"),
            "category": match.get("category"),
            "surface": surface,
            "round": match.get("round"),
            "round_name": match.get("round_name"),
            "player1": player1,
            "player2": player2,
            "seed1": match.get("seed1"),
            "seed2": match.get("seed2"),
            "odds1_avg": match.get("odds1_avg"),
            "odds2_avg": match.get("odds2_avg"),
            "target_player1_win": int(sets1 > sets2),
            "sets1": sets1,
            "sets2": sets2,
            "games1": games1,
            "games2": games2,
            "total_games": as_int(match.get("total_games")),
            "game_diff": as_int(match.get("game_diff")),
            "prior_h2h_player1_wins": prior_h2h[player1],
            "prior_h2h_player2_wins": prior_h2h[player2],
            "prior_h2h_matches": prior_h2h[player1] + prior_h2h[player2],
        }
        row.update(states[player1].snapshot("player1", surface))
        row.update(states[player2].snapshot("player2", surface))
        add_prefix_diff(
            row,
            "player1",
            "player2",
            [
                "prior_matches",
                "prior_win_rate",
                "prior_surface_win_rate",
                "prior_recent_win_rate",
                "prior_avg_game_diff",
                "prior_avg_set_diff",
                "prior_recent_avg_game_diff",
            ],
        )
        feature_rows.append(row)

        player1_won = sets1 > sets2
        states[player1].update(player1_won, surface, games1, games2, sets1, sets2)
        states[player2].update(not player1_won, surface, games2, games1, sets2, sets1)
        h2h[pair_key][player1 if player1_won else player2] += 1

    return feature_rows


def build_features_from_csv(input_path: Path, output_path: Path) -> None:
    matches = read_csv(input_path)
    rows = build_no_leakage_features(matches)
    write_csv(output_path, rows)
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "matches_in": len(matches),
        "feature_rows": len(rows),
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def add_match_detail_snapshots(feature_rows: list[dict], details: list[dict]) -> list[dict]:
    by_match_id = {row.get("match_id"): row for row in details if row.get("match_id")}
    skip = {"match_id", "match_url", "source_lines_sample", "title"}
    enriched: list[dict] = []
    for row in feature_rows:
        merged = dict(row)
        detail = by_match_id.get(row.get("match_id"))
        if detail:
            for key, value in detail.items():
                if key not in skip:
                    merged[f"te_detail_{key}"] = value
        enriched.append(merged)
    return enriched


def build_features_from_csvs(input_path: Path, output_path: Path, details_path: Path | None = None) -> None:
    matches = read_csv(input_path)
    rows = build_no_leakage_features(matches)
    if details_path:
        rows = add_match_detail_snapshots(rows, read_csv(details_path))
    write_csv(output_path, rows)
    summary = {
        "input": str(input_path),
        "details": str(details_path) if details_path else None,
        "output": str(output_path),
        "matches_in": len(matches),
        "feature_rows": len(rows),
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
