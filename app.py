from __future__ import annotations

import re
from itertools import combinations
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st


BASE = Path(__file__).parent
MODEL_DATA = BASE / "files" / "processed" / "grand_slam_moneyline" / "model_dataset"
TRAINING = BASE / "files" / "processed" / "grand_slam_moneyline" / "model_training"
TEMPORAL = BASE / "files" / "processed" / "temporal_validation_2026"
ATP = BASE / "files" / "processed" / "atp_2026"
ATP_2025 = BASE / "files" / "processed" / "atp_2025"
DASHBOARD_TARGET = BASE / "files" / "processed" / "dashboard_target.json"

MODEL_NAME = "regresion_logistica"
DETAIL_ODDS_FILES = [
    ATP / "upcoming_match_details.csv",
    ATP / "match_details.csv",
]
MODEL_LABEL = "Grand Slam BO5"
DEFAULT_MIN_MATCH_PROB = 0.65
MIN_SET_PROB = 0.72
MIN_SWEEP_PROB = 0.18
MIN_HANDICAP_PROB = 0.45
PARLAY_MIN_ODDS = 1.10
PARLAY_MAX_ODDS = 1.50
PARLAY_MIN_TOTAL_ODDS = 1.60
PARLAY_MAX_TOTAL_ODDS = 2.00
MONEYLINE_STAKE_UNITS = 1.0
PARLAY_STAKE_UNITS = 5.0
DEFAULT_TARGET_CONFIG = {
    "label": "Madrid upcoming",
    "tournament": "Madrid",
    "round": None,
    "predictions_file": "files/processed/model_training_2025_2026/madrid_predictions.csv",
    "upcoming_file": "files/processed/atp_2026/upcoming_matches.csv",
    "local_reference_date": None,
    "raw_today_offset_days": 0,
    "raw_tomorrow_offset_days": 1,
    "model_data_dir": "files/processed/model_dataset_2025_2026",
    "training_dir": "files/processed/model_training_2025_2026",
    "model_label": "ATP BO3",
    "market_profile": "non_grand_slam_bo3",
    "excluded_match_ids": [],
}


st.set_page_config(page_title="Tennis Value Dashboard", layout="wide")


@st.cache_data
def read_csv_cached(path: Path, modified_at: float) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_csv(path: Path) -> pd.DataFrame:
    modified_at = path.stat().st_mtime if path.exists() else 0
    return read_csv_cached(path, modified_at)


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def resolve_path(value: str | None) -> Path:
    if not value:
        return Path()
    path = Path(value)
    return path if path.is_absolute() else BASE / path


def load_target_config() -> dict:
    config = {**DEFAULT_TARGET_CONFIG, **read_json(DASHBOARD_TARGET)}
    config["predictions_path"] = resolve_path(config.get("predictions_file"))
    config["upcoming_path"] = resolve_path(config.get("upcoming_file"))
    config["model_data_path"] = resolve_path(config.get("model_data_dir"))
    config["training_path"] = resolve_path(config.get("training_dir"))
    return config


def filter_prediction_target(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    if df.empty:
        return df
    if "match_id" in df.columns and config.get("excluded_match_ids"):
        excluded = {str(match_id) for match_id in config.get("excluded_match_ids", [])}
        df = df[~df["match_id"].astype(str).isin(excluded)]
    tournament = config.get("tournament")
    round_name = config.get("round")
    filtered = df
    if tournament:
        for col in ("torneo", "tournament"):
            if col in filtered.columns:
                filtered = filtered[filtered[col] == tournament]
                break
    if round_name:
        for col in ("ronda", "round", "round_name"):
            if col in filtered.columns:
                filtered = filtered[filtered[col] == round_name]
                break
    return filtered.copy()


def parse_date_value(value: object) -> date:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return date.today()
    return parsed.date()


def schedule_reference_date(config: dict) -> date:
    return parse_date_value(config.get("local_reference_date"))


def parse_schedule_datetime(start_raw: object, year: object, config: dict) -> datetime:
    text = str(start_raw or "").strip().lower()
    base = schedule_reference_date(config)
    match_date = base
    explicit_date = re.search(r"(\d{1,2})\.(\d{1,2})\.", text)
    if explicit_date:
        match_year = int(year) if str(year or "").isdigit() else base.year
        match_date = date(match_year, int(explicit_date.group(2)), int(explicit_date.group(1)))
    elif "tomorrow" in text:
        match_date = base + timedelta(days=int(config.get("raw_tomorrow_offset_days") or 1))
    elif "today" in text or re.match(r"^\d{1,2}:\d{2}", text):
        match_date = base + timedelta(days=int(config.get("raw_today_offset_days") or 0))

    time_match = re.search(r"(\d{1,2}):(\d{2})", text)
    hour = int(time_match.group(1)) if time_match else 0
    minute = int(time_match.group(2)) if time_match else 0
    return datetime.combine(match_date, datetime.min.time()).replace(hour=hour, minute=minute)


def add_schedule_columns(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    if df.empty or "start_raw" not in df.columns:
        return df
    out = df.copy()
    years = out["year"] if "year" in out.columns else pd.Series([None] * len(out), index=out.index)
    schedule = [
        parse_schedule_datetime(start_raw, year, config)
        for start_raw, year in zip(out["start_raw"], years)
    ]
    out["fecha_hora_local"] = pd.to_datetime(schedule)
    out["fecha_local"] = out["fecha_hora_local"].dt.strftime("%Y-%m-%d %H:%M")
    return out


def tournament_column(df: pd.DataFrame) -> str | None:
    for col in ("Torneo", "torneo", "tournament"):
        if col in df.columns:
            return col
    return None


def tournament_filter(df: pd.DataFrame, key: str) -> tuple[pd.DataFrame, str]:
    col = tournament_column(df)
    if df.empty or col is None:
        return df, "Todos"
    tournaments = sorted(str(value) for value in df[col].dropna().unique())
    selected = st.selectbox("Torneo", ["Todos", *tournaments], key=key)
    if selected == "Todos":
        return df, selected
    return df[df[col].astype(str) == selected], selected


def day_filter(df: pd.DataFrame, key: str) -> tuple[pd.DataFrame, str]:
    if df.empty or "fecha_hora_local" not in df.columns:
        return df, "Todos"
    available_days = sorted(pd.to_datetime(df["fecha_hora_local"], errors="coerce").dt.date.dropna().unique())
    day_options = ["Todos", *[day.isoformat() for day in available_days]]
    selected = st.selectbox("Dia", day_options, key=key)
    if selected == "Todos":
        return df, selected
    selected_day = date.fromisoformat(selected)
    return df[pd.to_datetime(df["fecha_hora_local"], errors="coerce").dt.date == selected_day], selected


def load_prediction_frame(target: dict) -> pd.DataFrame:
    preds = filter_prediction_target(read_csv(target["predictions_path"]), target)
    upcoming = filter_prediction_target(read_csv(target["upcoming_path"]), target)
    if preds.empty:
        return preds

    preds["match_id"] = preds["match_id"].astype(str)
    if not upcoming.empty:
        upcoming["match_id"] = upcoming["match_id"].astype(str)
        keep = [
            "match_id",
            "start_raw",
            "odds1_avg",
            "odds2_avg",
            "h2h_player1_wins",
            "h2h_player2_wins",
            "match_url",
            "year",
        ]
        preds = preds.merge(upcoming[[c for c in keep if c in upcoming.columns]], on="match_id", how="left")
    match_results = load_match_results(target["upcoming_path"].with_name("match_results.csv"))
    if not match_results.empty:
        preds = preds.merge(match_results, on="match_id", how="left")
        if "result_start_raw" in preds.columns:
            preds["start_raw"] = preds.get("start_raw", pd.Series(index=preds.index)).combine_first(preds["result_start_raw"])
        if "result_odds1_avg" in preds.columns:
            preds["odds1_avg"] = preds.get("odds1_avg", pd.Series(index=preds.index)).combine_first(preds["result_odds1_avg"])
        if "result_odds2_avg" in preds.columns:
            preds["odds2_avg"] = preds.get("odds2_avg", pd.Series(index=preds.index)).combine_first(preds["result_odds2_avg"])
    target_detail_odds = target["upcoming_path"].with_name("upcoming_match_details.csv")
    detail_odds = load_detail_odds([target_detail_odds])
    if not detail_odds.empty:
        preds = preds.merge(detail_odds, on="match_id", how="left")
        if "homeaway_avg_odds1" in preds.columns:
            preds["odds1_avg"] = preds.get("odds1_avg", pd.Series(index=preds.index)).combine_first(preds["homeaway_avg_odds1"])
        if "homeaway_avg_odds2" in preds.columns:
            preds["odds2_avg"] = preds.get("odds2_avg", pd.Series(index=preds.index)).combine_first(preds["homeaway_avg_odds2"])

    p1 = "prob_gana_jugador_1"
    legacy_p1 = f"prob_gana_jugador_1_{MODEL_NAME}"
    if p1 not in preds.columns and legacy_p1 in preds.columns:
        preds[p1] = preds[legacy_p1]
    preds[p1] = to_num(preds[p1])
    preds["odds1_avg"] = to_num(preds.get("odds1_avg", pd.Series(index=preds.index)))
    preds["odds2_avg"] = to_num(preds.get("odds2_avg", pd.Series(index=preds.index)))

    preds["prob_modelo_j1"] = preds[p1]
    preds["prob_modelo_j2"] = 1 - preds[p1]
    preds["prob_mercado_j1"] = implied_probability(preds["odds1_avg"], preds["odds2_avg"], side=1)
    preds["prob_mercado_j2"] = implied_probability(preds["odds1_avg"], preds["odds2_avg"], side=2)
    preds["edge_j1"] = preds["prob_modelo_j1"] - preds["prob_mercado_j1"]
    preds["edge_j2"] = preds["prob_modelo_j2"] - preds["prob_mercado_j2"]
    preds["kelly_j1"] = kelly_fraction(preds["prob_modelo_j1"], preds["odds1_avg"])
    preds["kelly_j2"] = kelly_fraction(preds["prob_modelo_j2"], preds["odds2_avg"])
    preds["favorito_modelo"] = preds.apply(
        lambda row: row["jugador_1"] if row["prob_modelo_j1"] >= row["prob_modelo_j2"] else row["jugador_2"],
        axis=1,
    )
    preds["prob_favorito_modelo"] = preds[["prob_modelo_j1", "prob_modelo_j2"]].max(axis=1)
    add_set_market_columns(preds)
    preds["recomendacion"] = preds.apply(recommendation, axis=1)
    preds["resultado_recomendacion"] = preds.apply(recommendation_result, axis=1)
    preds["edge_recomendado"] = preds[["edge_j1", "edge_j2"]].max(axis=1)
    preds = add_schedule_columns(preds, target)
    if "result_date" in preds.columns and "fecha_hora_local" in preds.columns:
        result_dates = pd.to_datetime(preds["result_date"], errors="coerce")
        has_result_date = result_dates.notna()
        if has_result_date.any():
            times = pd.to_datetime(preds["fecha_hora_local"], errors="coerce").dt.time
            preds.loc[has_result_date, "fecha_hora_local"] = [
                datetime.combine(day.date(), time_value if pd.notna(time_value) else datetime.min.time())
                for day, time_value in zip(result_dates[has_result_date], times[has_result_date])
            ]
            preds.loc[has_result_date, "fecha_local"] = pd.to_datetime(preds.loc[has_result_date, "fecha_hora_local"]).dt.strftime("%Y-%m-%d %H:%M")
    sort_cols = [col for col in ["fecha_hora_local", "start_raw", "edge_recomendado"] if col in preds.columns]
    ascending = [True, True, False][: len(sort_cols)]
    return preds.sort_values(sort_cols, ascending=ascending)


def load_predictions() -> pd.DataFrame:
    return load_prediction_frame(load_target_config())


def load_match_results(path: Path) -> pd.DataFrame:
    results = read_csv(path)
    if results.empty or "match_id" not in results.columns:
        return pd.DataFrame()
    results = results.copy()
    results["match_id"] = results["match_id"].astype(str)
    rename = {
        "player1": "result_player1",
        "player2": "result_player2",
        "start_raw": "result_start_raw",
        "sets_player1": "result_sets_j1",
        "sets_player2": "result_sets_j2",
        "odds1_avg": "result_odds1_avg",
        "odds2_avg": "result_odds2_avg",
        "source_url": "result_source_url",
    }
    results = results.rename(columns=rename)
    for col in ["result_sets_j1", "result_sets_j2", "result_odds1_avg", "result_odds2_avg"]:
        if col in results.columns:
            results[col] = to_num(results[col])
    keep = [
        "match_id",
        "winner",
        "result_date",
        "result_start_raw",
        "score",
        "result_sets_j1",
        "result_sets_j2",
        "result_odds1_avg",
        "result_odds2_avg",
        "retired",
        "result_source_url",
    ]
    return results[[col for col in keep if col in results.columns]]


def implied_probability(odds1: pd.Series, odds2: pd.Series, side: int) -> pd.Series:
    raw1 = 1 / odds1
    raw2 = 1 / odds2
    total = raw1 + raw2
    return (raw1 if side == 1 else raw2) / total


def load_detail_odds(extra_paths: list[Path] | None = None) -> pd.DataFrame:
    frames = []
    keep = [
        "match_id",
        "homeaway_avg_odds1",
        "homeaway_avg_odds2",
        "set_odds_player1_wins_set",
        "set_odds_player2_wins_set",
        "set_odds_player1_wins_2_0",
        "set_odds_player2_wins_2_0",
        "set_odds_player1_wins_3_0",
        "set_odds_player2_wins_3_0",
        "set_odds_table_player1_minus_1_5",
        "set_odds_table_player2_minus_1_5",
        "set_odds_table_player1_plus_1_5",
        "set_odds_table_player2_plus_1_5",
    ]
    for path in [*(extra_paths or []), *DETAIL_ODDS_FILES]:
        frame = read_csv(path)
        if frame.empty or "match_id" not in frame.columns:
            continue
        frame["match_id"] = frame["match_id"].astype(str)
        frames.append(frame[[col for col in keep if col in frame.columns]])
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    return merged.drop_duplicates("match_id", keep="last")


def add_set_market_columns(preds: pd.DataFrame) -> None:
    aliases = {
        "prob_jugador_1_gana_al_menos_1_set": "prob_jugador_1_gana_al_menos_un_set",
        "prob_jugador_2_gana_al_menos_1_set": "prob_jugador_2_gana_al_menos_un_set",
    }
    for target_col, source_col in aliases.items():
        if target_col not in preds.columns and source_col in preds.columns:
            preds[target_col] = preds[source_col]
    is_bo3 = "prob_jugador_1_gana_2_0" in preds.columns or "prob_jugador_2_gana_2_0" in preds.columns
    if is_bo3:
        bo3_set_odds_fallbacks = {
            "set_odds_player1_wins_set": "set_odds_table_player1_plus_1_5",
            "set_odds_player2_wins_set": "set_odds_table_player2_minus_1_5",
        }
        for target_col, fallback_col in bo3_set_odds_fallbacks.items():
            if fallback_col not in preds.columns:
                continue
            fallback = to_num(preds[fallback_col])
            if target_col in preds.columns:
                preds[target_col] = to_num(preds[target_col]).combine_first(fallback)
            else:
                preds[target_col] = fallback
    market_map = {
        "j1_set": ("prob_jugador_1_gana_al_menos_1_set", "set_odds_player1_wins_set"),
        "j2_set": ("prob_jugador_2_gana_al_menos_1_set", "set_odds_player2_wins_set"),
        "j1_2_0": ("prob_jugador_1_gana_2_0", "set_odds_player1_wins_2_0"),
        "j2_2_0": ("prob_jugador_2_gana_2_0", "set_odds_player2_wins_2_0"),
        "j1_3_0": ("prob_jugador_1_gana_3_0", "set_odds_player1_wins_3_0"),
        "j2_3_0": ("prob_jugador_2_gana_3_0", "set_odds_player2_wins_3_0"),
        "j1_minus_1_5": ("prob_jugador_1_minus_1_5_sets", "set_odds_table_player1_minus_1_5"),
        "j2_minus_1_5": ("prob_jugador_2_minus_1_5_sets", "set_odds_table_player2_plus_1_5"),
        "j1_2sets": ("prob_jugador_1_gana_al_menos_2_sets", "set_odds_table_player1_plus_1_5"),
        "j2_2sets": ("prob_jugador_2_gana_al_menos_2_sets", "set_odds_table_player2_minus_1_5"),
    }
    for suffix, (prob_col, odds_col) in market_map.items():
        preds[prob_col] = to_num(preds.get(prob_col, pd.Series(index=preds.index)))
        preds[odds_col] = to_num(preds.get(odds_col, pd.Series(index=preds.index)))
        preds[f"prob_modelo_{suffix}"] = preds[prob_col]
        preds[f"prob_mercado_{suffix}"] = 1 / preds[odds_col]
        preds[f"edge_{suffix}"] = preds[f"prob_modelo_{suffix}"] - preds[f"prob_mercado_{suffix}"]
        preds[f"kelly_{suffix}"] = kelly_fraction(preds[f"prob_modelo_{suffix}"], preds[odds_col])


def kelly_fraction(prob: pd.Series, odds: pd.Series) -> pd.Series:
    b = odds - 1
    return ((prob * b) - (1 - prob)) / b


def recommendation(row: pd.Series) -> str:
    min_edge = st.session_state.get("min_edge", 0.05)
    min_prob = st.session_state.get("min_match_prob", DEFAULT_MIN_MATCH_PROB)
    candidates = []
    for side in (1, 2):
        edge = row[f"edge_j{side}"]
        kelly = row[f"kelly_j{side}"]
        prob = row[f"prob_modelo_j{side}"]
        player = row[f"jugador_{side}"]
        odds = row[f"odds{side}_avg"]
        if pd.notna(edge) and pd.notna(kelly) and pd.notna(prob) and edge >= min_edge and prob >= min_prob and kelly > 0:
            candidates.append((edge, f"{player} ML @ {odds:.2f} | prob {prob:.1%} | edge {edge:.1%}"))
    if not candidates:
        return "Sin apuesta"
    return max(candidates, key=lambda item: item[0])[1]


def normalize_player(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def is_retired_result(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "retired"}


def result_marker(won: bool | None) -> str:
    if won is None:
        return ""
    return "check" if won else "cross"


def player_side(match: pd.Series, player: object) -> int | None:
    player_key = normalize_player(player)
    if player_key and player_key == normalize_player(match.get("jugador_1")):
        return 1
    if player_key and player_key == normalize_player(match.get("jugador_2")):
        return 2
    return None


def evaluate_pick(match: pd.Series, group: str, market: str, player: object) -> str:
    winner = match.get("winner")
    if pd.isna(winner) or not str(winner).strip():
        return ""
    side = player_side(match, player)
    if side is None:
        return ""
    selected_sets = match.get(f"result_sets_j{side}")
    other_sets = match.get(f"result_sets_j{2 if side == 1 else 1}")
    if group != "Match winner" and is_retired_result(match.get("retired")):
        return "void"
    if group == "Match winner":
        return result_marker(normalize_player(player) == normalize_player(winner))
    if pd.isna(selected_sets) or pd.isna(other_sets):
        return ""
    if group == "Gana set":
        return result_marker(selected_sets >= 1)
    if group == "2-0":
        return result_marker(selected_sets == 2 and other_sets == 0)
    if group == "3-0":
        return result_marker(selected_sets == 3 and other_sets == 0)
    if group == "Handicap sets":
        return result_marker((selected_sets - other_sets) >= 2)
    if group == "Gana 2+ sets":
        return result_marker(selected_sets >= 2)
    return ""


def recommendation_result(row: pd.Series) -> str:
    recommendation_text = str(row.get("recomendacion") or "")
    if recommendation_text == "Sin apuesta":
        return ""
    for side in (1, 2):
        player = row.get(f"jugador_{side}")
        if recommendation_text.startswith(f"{player} ML"):
            return evaluate_pick(row, "Match winner", f"J{side} gana partido", player)
    return ""


def parse_round_number(round_name: object) -> int | None:
    match = re.fullmatch(r"(\d+)R", str(round_name or "").strip().upper())
    return int(match.group(1)) if match else None


def round_slug(round_name: str) -> str:
    return str(round_name).strip().lower()


def target_prefix_from_path(path: Path, round_name: object, suffix: str) -> str:
    stem = path.stem
    marker = f"_{round_slug(str(round_name))}{suffix}"
    return stem[: -len(marker)] if marker and stem.endswith(marker) else stem


def previous_round_configs(target: dict) -> list[dict]:
    current_round = parse_round_number(target.get("round"))
    if not current_round or current_round <= 1:
        return []

    predictions_prefix = target_prefix_from_path(target["predictions_path"], target.get("round"), "_predictions")
    upcoming_prefix = target_prefix_from_path(target["upcoming_path"].parent, target.get("round"), "")
    configs = []
    for number in range(1, current_round):
        previous_round = f"{number}R"
        predictions_path = target["predictions_path"].with_name(f"{predictions_prefix}_{round_slug(previous_round)}_predictions.csv")
        upcoming_path = target["upcoming_path"].parent.parent / f"{upcoming_prefix}_{round_slug(previous_round)}" / "upcoming_matches.csv"
        if not predictions_path.exists() or not upcoming_path.exists():
            continue
        configs.append(
            {
                **target,
                "label": f"{target.get('tournament')} {previous_round}",
                "round": previous_round,
                "predictions_path": predictions_path,
                "upcoming_path": upcoming_path,
            }
        )
    return configs


def load_historical_predictions(target: dict) -> pd.DataFrame:
    frames = []
    for config in previous_round_configs(target):
        frame = load_prediction_frame(config)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["Ronda historial"] = config.get("round")
        frames.append(frame)
    current_frame = load_prediction_frame(target)
    if not current_frame.empty and "result_date" in current_frame.columns:
        result_dates = pd.to_datetime(current_frame["result_date"], errors="coerce")
        current_frame = current_frame[result_dates.notna()].copy()
        if not current_frame.empty:
            current_frame["Ronda historial"] = target.get("round")
            frames.append(current_frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_historical_raw_results(target: dict) -> pd.DataFrame:
    frames = []
    for config in previous_round_configs(target):
        path = config["upcoming_path"].with_name("match_results.csv")
        frame = read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["match_id"] = frame["match_id"].astype(str)
        frame["Ronda historial"] = config.get("round")
        frames.append(frame)
    current_path = target["upcoming_path"].with_name("match_results.csv")
    current_frame = read_csv(current_path)
    if not current_frame.empty:
        current_frame = current_frame.copy()
        current_frame["match_id"] = current_frame["match_id"].astype(str)
        current_frame["Ronda historial"] = target.get("round")
        frames.append(current_frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_market_rows(df: pd.DataFrame, market_profile: str = "grand_slam_bo5") -> pd.DataFrame:
    rows = []
    common_markets = [
        ("Match winner", "J1 gana partido", "jugador_1", "prob_modelo_j1", "odds1_avg", "prob_mercado_j1", "edge_j1", "kelly_j1", DEFAULT_MIN_MATCH_PROB),
        ("Match winner", "J2 gana partido", "jugador_2", "prob_modelo_j2", "odds2_avg", "prob_mercado_j2", "edge_j2", "kelly_j2", DEFAULT_MIN_MATCH_PROB),
        ("Gana set", "J1 gana al menos un set", "jugador_1", "prob_modelo_j1_set", "set_odds_player1_wins_set", "prob_mercado_j1_set", "edge_j1_set", "kelly_j1_set", MIN_SET_PROB),
        ("Gana set", "J2 gana al menos un set", "jugador_2", "prob_modelo_j2_set", "set_odds_player2_wins_set", "prob_mercado_j2_set", "edge_j2_set", "kelly_j2_set", MIN_SET_PROB),
    ]
    bo3_markets = [
        ("2-0", "J1 gana 2-0", "jugador_1", "prob_modelo_j1_2_0", "set_odds_player1_wins_2_0", "prob_mercado_j1_2_0", "edge_j1_2_0", "kelly_j1_2_0", MIN_SWEEP_PROB),
        ("2-0", "J2 gana 2-0", "jugador_2", "prob_modelo_j2_2_0", "set_odds_player2_wins_2_0", "prob_mercado_j2_2_0", "edge_j2_2_0", "kelly_j2_2_0", MIN_SWEEP_PROB),
    ]
    bo5_markets = [
        ("3-0", "J1 gana 3-0", "jugador_1", "prob_modelo_j1_3_0", "set_odds_player1_wins_3_0", "prob_mercado_j1_3_0", "edge_j1_3_0", "kelly_j1_3_0", MIN_SWEEP_PROB),
        ("3-0", "J2 gana 3-0", "jugador_2", "prob_modelo_j2_3_0", "set_odds_player2_wins_3_0", "prob_mercado_j2_3_0", "edge_j2_3_0", "kelly_j2_3_0", MIN_SWEEP_PROB),
        ("Handicap sets", "J1 -1.5 sets", "jugador_1", "prob_modelo_j1_minus_1_5", "set_odds_table_player1_minus_1_5", "prob_mercado_j1_minus_1_5", "edge_j1_minus_1_5", "kelly_j1_minus_1_5", MIN_HANDICAP_PROB),
        ("Handicap sets", "J2 -1.5 sets", "jugador_2", "prob_modelo_j2_minus_1_5", "set_odds_table_player2_plus_1_5", "prob_mercado_j2_minus_1_5", "edge_j2_minus_1_5", "kelly_j2_minus_1_5", MIN_HANDICAP_PROB),
        ("Gana 2+ sets", "J1 gana al menos 2 sets", "jugador_1", "prob_modelo_j1_2sets", "set_odds_table_player1_plus_1_5", "prob_mercado_j1_2sets", "edge_j1_2sets", "kelly_j1_2sets", MIN_HANDICAP_PROB),
        ("Gana 2+ sets", "J2 gana al menos 2 sets", "jugador_2", "prob_modelo_j2_2sets", "set_odds_table_player2_minus_1_5", "prob_mercado_j2_2sets", "edge_j2_2sets", "kelly_j2_2sets", MIN_HANDICAP_PROB),
    ]
    markets = common_markets + (bo3_markets if market_profile == "non_grand_slam_bo3" else bo5_markets)
    for _, match in df.iterrows():
        for group, market, player_col, prob_col, odds_col, implied_col, edge_col, kelly_col, min_prob in markets:
            rows.append(
                {
                    "fecha_hora_local": match.get("fecha_hora_local"),
                    "Fecha local": match.get("fecha_local"),
                    "Torneo": match.get("torneo", match.get("tournament")),
                    "Ronda historial": match.get("Ronda historial"),
                    "Hora": match.get("start_raw"),
                    "Partido": f"{match.get('jugador_1')} vs {match.get('jugador_2')}",
                    "Grupo": group,
                    "Mercado": market,
                    "Jugador": match.get(player_col),
                    "Prob. modelo": match.get(prob_col),
                    "Cuota": match.get(odds_col),
                    "Prob. cuota": match.get(implied_col),
                    "Edge": match.get(edge_col),
                    "Kelly": match.get(kelly_col),
                    "Prob. minima": min_prob,
                    "Resultado pick": evaluate_pick(match, group, market, match.get(player_col)),
                    "Ganador": match.get("winner"),
                    "Score": match.get("score"),
                }
            )
    return pd.DataFrame(rows)


def value_market_rows(market_rows: pd.DataFrame) -> pd.DataFrame:
    if market_rows.empty:
        return market_rows
    min_edge = st.session_state.get("min_edge", 0.05)
    min_prob = st.session_state.get("min_match_prob", DEFAULT_MIN_MATCH_PROB)
    rows = market_rows.copy()
    rows["Prob. minima"] = min_prob
    high_probability_pick = (
        rows["Cuota"].between(PARLAY_MIN_ODDS, PARLAY_MAX_ODDS, inclusive="both")
        & rows["Prob. modelo"].notna()
    )
    rows = rows[
        rows["Cuota"].notna()
        & rows["Edge"].notna()
        & (rows["Prob. modelo"] >= rows["Prob. minima"])
        & ((rows["Edge"] >= min_edge) | high_probability_pick)
    ].copy()
    if rows.empty:
        return rows
    rows = add_recommendation_reason(rows)
    sort_cols = [col for col in ["fecha_hora_local", "Hora", "Edge", "Prob. modelo"] if col in rows.columns]
    ascending = [True, True, False, False][: len(sort_cols)]
    return rows.sort_values(sort_cols, ascending=ascending)


def add_recommendation_reason(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    min_edge = st.session_state.get("min_edge", 0.05)
    out = rows.copy()
    out["Recomendada por"] = out["Edge"].apply(
        lambda edge: "Valor" if pd.notna(edge) and edge >= min_edge else "Probabilidad modelo"
    )
    return out


def suggested_parlay(market_rows: pd.DataFrame, day: date | None = None) -> pd.DataFrame:
    if market_rows.empty:
        return market_rows
    min_prob = st.session_state.get("min_match_prob", DEFAULT_MIN_MATCH_PROB)
    candidates = market_rows.copy()
    candidates["Prob. minima"] = min_prob
    if day is not None and "fecha_hora_local" in candidates.columns:
        candidates = candidates[pd.to_datetime(candidates["fecha_hora_local"], errors="coerce").dt.date == day]
    candidates = candidates[
        candidates["Cuota"].between(PARLAY_MIN_ODDS, PARLAY_MAX_ODDS, inclusive="both")
        & candidates["Prob. modelo"].notna()
        & (candidates["Prob. modelo"] >= candidates["Prob. minima"])
    ].copy()
    if candidates.empty:
        return pd.DataFrame()

    candidates = candidates.sort_values(["Prob. modelo", "Edge", "fecha_hora_local"], ascending=[False, False, True])
    best_combo = None
    best_score = -999.0
    max_candidates = 18
    for size in range(2, 5):
        for combo in combinations(candidates.head(max_candidates).to_dict("records"), size):
            partidos = [item["Partido"] for item in combo]
            if len(set(partidos)) != len(partidos):
                continue
            final_odds = 1.0
            for item in combo:
                final_odds *= item["Cuota"]
            if not PARLAY_MIN_TOTAL_ODDS <= final_odds <= PARLAY_MAX_TOTAL_ODDS:
                continue
            score = sum(
                float(item["Prob. modelo"])
                + ((float(item["Edge"]) if pd.notna(item.get("Edge")) else 0) * 0.05)
                + (0.03 if item.get("Grupo") == "Match winner" else 0)
                for item in combo
            ) / size
            if score > best_score:
                best_combo = combo
                best_score = score

    if not best_combo:
        return pd.DataFrame()
    out = pd.DataFrame(best_combo).copy()
    out["Cuota combinada"] = out["Cuota"].prod()
    return add_recommendation_reason(out)


def sort_table_controls(df: pd.DataFrame, key: str, default_col: str | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    columns = list(df.columns)
    default_index = columns.index(default_col) if default_col in columns else 0
    left, right = st.columns([2, 1])
    with left:
        sort_col = st.selectbox("Ordenar por", columns, index=default_index, key=f"{key}_sort_col")
    with right:
        direction = st.selectbox("Orden", ["Ascendente", "Descendente"], key=f"{key}_sort_dir")
    ascending = direction == "Ascendente"
    try:
        return df.sort_values(sort_col, ascending=ascending, kind="mergesort")
    except TypeError:
        return df.assign(_sort_key=df[sort_col].astype(str)).sort_values("_sort_key", ascending=ascending).drop(columns="_sort_key")


def metric_card(label: str, value: str) -> None:
    st.metric(label, value)


def format_pct(value: float | int | None) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.1%}"


def style_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for col in [
        "prob_modelo_j1",
        "prob_modelo_j2",
        "prob_modelo_j1_set",
        "prob_modelo_j2_set",
        "prob_modelo_j1_3_0",
        "prob_modelo_j2_3_0",
        "prob_modelo_j1_minus_1_5",
        "prob_modelo_j2_minus_1_5",
        "prob_modelo_j1_2sets",
        "prob_modelo_j2_2sets",
        "prob_mercado_j1",
        "prob_mercado_j2",
        "edge_j1",
        "edge_j2",
        "kelly_j1",
        "kelly_j2",
        "Prob. J1",
        "Prob. J2",
        "J1 gana set",
        "J2 gana set",
        "J1 3-0",
        "J2 3-0",
        "J1 -1.5 sets",
        "J2 -1.5 sets",
        "J1 2+ sets",
        "J2 2+ sets",
        "Edge J1",
        "Edge J2",
    ]:
        if col in display.columns:
            display[col] = display[col].map(format_pct)
    for col in ["Cuota J1", "Cuota J2"]:
        if col in display.columns:
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
    return style_result_column(display)


def style_market_table(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for col in ["Prob. modelo", "Prob. cuota", "Edge", "Kelly", "Prob. minima"]:
        if col in display.columns:
            display[col] = display[col].map(format_pct)
    for col in ["Cuota", "Cuota combinada"]:
        if col in display.columns:
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
    if "Stake" in display.columns:
        display["Stake"] = display["Stake"].map(lambda value: "" if pd.isna(value) else f"{value:.1f} u")
    if "Ganancia" in display.columns:
        display["Ganancia"] = display["Ganancia"].map(format_units)
    return style_result_column(display)


def friendly_result(value: object) -> str:
    markers = {
        "check": "✓",
        "cross": "✗",
        "void": "-",
    }
    return markers.get(str(value or "").strip().lower(), "")


def result_cell_style(value: object) -> str:
    if value == "✓":
        return "color: #15803d; font-weight: 700; font-size: 1.1rem;"
    if value == "✗":
        return "color: #b91c1c; font-weight: 700; font-size: 1.1rem;"
    if value == "-":
        return "color: #6b7280; font-weight: 700;"
    return ""


def style_result_column(display: pd.DataFrame):
    if "Resultado pick" not in display.columns:
        return display
    styled = display.copy()
    styled["Resultado pick"] = styled["Resultado pick"].map(friendly_result)
    return styled.style.map(result_cell_style, subset=["Resultado pick"])


def friendly_table(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    available = [col for col in columns if col in df.columns]
    return df[available].rename(columns={col: columns[col] for col in available})


def main() -> None:
    st.title("Tennis Predictions")
    st.caption("Modelo operativo: Regresion logistica v1. Las cuotas se usan solo para evaluar value, no como feature de entrenamiento.")

    st.sidebar.header("Parametros")
    target = load_target_config()
    st.sidebar.caption(f"Torneo activo: {target.get('label')}")
    st.session_state["min_edge"] = st.sidebar.slider("Edge minimo", 0.00, 0.20, 0.05, 0.01)
    st.session_state["min_match_prob"] = st.sidebar.slider("Prob. minima match winner", 0.50, 0.95, DEFAULT_MIN_MATCH_PROB, 0.01)
    page_tabs = st.tabs(["Predicciones", "Historial", "Modelos", "Datos"])
    with page_tabs[0]:
        recommendations_view()
    with page_tabs[1]:
        history_view()
    with page_tabs[2]:
        models_view()
    with page_tabs[3]:
        data_view()


def finished_pick_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or "Resultado pick" not in rows.columns:
        return rows
    return rows[rows["Resultado pick"].isin(["check", "cross", "void"])].copy()


def result_counts(rows: pd.DataFrame) -> tuple[int, int, int]:
    if rows.empty or "Resultado pick" not in rows.columns:
        return 0, 0, 0
    green = int((rows["Resultado pick"] == "check").sum())
    red = int((rows["Resultado pick"] == "cross").sum())
    void = int((rows["Resultado pick"] == "void").sum())
    return green, red, void


def accuracy_pct(rows: pd.DataFrame) -> float | None:
    green, red, _ = result_counts(rows)
    settled = green + red
    return (green / settled) if settled else None


def pick_profit(result: object, odds: object, stake: float = 1.0) -> float:
    marker = str(result or "").strip().lower()
    if marker == "void":
        return 0.0
    if marker == "check":
        value = pd.to_numeric(pd.Series([odds]), errors="coerce").iloc[0]
        return float((value - 1) * stake) if pd.notna(value) else 0.0
    if marker == "cross":
        return -stake
    return 0.0


def format_units(value: float | int | None) -> str:
    if pd.isna(value):
        return ""
    return f"{value:+.2f} u"


def parlay_result(rows: pd.DataFrame) -> str:
    if rows.empty or "Resultado pick" not in rows.columns:
        return ""
    settled = rows[rows["Resultado pick"].isin(["check", "cross"])]
    if settled.empty:
        return ""
    if (settled["Resultado pick"] == "cross").any():
        return "cross"
    return "check"


def build_daily_parlay_history(market_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if market_rows.empty or "fecha_hora_local" not in market_rows.columns:
        return pd.DataFrame(), pd.DataFrame()
    day_values = sorted(pd.to_datetime(market_rows["fecha_hora_local"], errors="coerce").dt.date.dropna().unique())
    parlay_frames = []
    summary_rows = []
    for day_value in day_values:
        day_parlay = suggested_parlay(market_rows, day_value)
        if day_parlay.empty:
            continue
        result = parlay_result(day_parlay)
        day_parlay = day_parlay.copy()
        day_parlay["Dia"] = day_value.isoformat()
        day_parlay["Resultado combinada"] = result
        parlay_frames.append(day_parlay)
        summary_rows.append(
            {
                "Dia": day_value.isoformat(),
                "Resultado pick": result,
                "Picks": len(day_parlay),
                "Cuota combinada": day_parlay["Cuota"].prod(),
                "Stake": PARLAY_STAKE_UNITS,
                "Ganancia": pick_profit(result, day_parlay["Cuota"].prod(), PARLAY_STAKE_UNITS),
                "Detalle": " + ".join(f"{row['Jugador']} ({row['Mercado']})" for _, row in day_parlay.iterrows()),
            }
        )
    details = pd.concat(parlay_frames, ignore_index=True) if parlay_frames else pd.DataFrame()
    return pd.DataFrame(summary_rows), details


def build_moneyline_history_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, match in df.iterrows():
        prob_j1 = match.get("prob_modelo_j1")
        prob_j2 = match.get("prob_modelo_j2")
        if pd.isna(prob_j1) and pd.isna(prob_j2):
            continue
        side = 1 if pd.notna(prob_j1) and (pd.isna(prob_j2) or prob_j1 >= prob_j2) else 2
        player = match.get(f"jugador_{side}")
        rows.append(
            {
                "fecha_hora_local": match.get("fecha_hora_local"),
                "Fecha local": match.get("fecha_local"),
                "Ronda historial": match.get("Ronda historial"),
                "Hora": match.get("start_raw"),
                "Partido": f"{match.get('jugador_1')} vs {match.get('jugador_2')}",
                "Grupo": "Match winner",
                "Mercado": f"J{side} gana partido",
                "Jugador": player,
                "Prob. modelo": match.get(f"prob_modelo_j{side}"),
                "Cuota": match.get(f"odds{side}_avg"),
                "Prob. cuota": match.get(f"prob_mercado_j{side}"),
                "Edge": match.get(f"edge_j{side}"),
                "Kelly": match.get(f"kelly_j{side}"),
                "Resultado pick": evaluate_pick(match, "Match winner", f"J{side} gana partido", player),
                "Ganador": match.get("winner"),
                "Score": match.get("score"),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["Stake"] = MONEYLINE_STAKE_UNITS
        out["Ganancia"] = out.apply(lambda row: pick_profit(row["Resultado pick"], row["Cuota"], MONEYLINE_STAKE_UNITS), axis=1)
    return out


def history_view() -> None:
    target = load_target_config()
    history = load_historical_predictions(target)
    raw_results = load_historical_raw_results(target)
    if history.empty and raw_results.empty:
        st.info("Todavia no hay rondas anteriores disponibles para este target.")
        return

    prediction_match_ids = set(history["match_id"].astype(str)) if not history.empty and "match_id" in history.columns else set()
    missing_prediction_results = raw_results[~raw_results["match_id"].astype(str).isin(prediction_match_ids)].copy() if not raw_results.empty else pd.DataFrame()

    market_rows = build_market_rows(history, target.get("market_profile", "grand_slam_bo5"))
    moneyline_rows = build_moneyline_history_rows(history)
    moneyline_rows = finished_pick_rows(moneyline_rows)
    parlay_summary, parlay_details = build_daily_parlay_history(market_rows)

    green, red, void = result_counts(moneyline_rows)
    parlay_green, parlay_red, parlay_void = result_counts(parlay_summary)
    ml_profit = moneyline_rows["Ganancia"].sum() if "Ganancia" in moneyline_rows.columns and not moneyline_rows.empty else 0
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Partidos jugados", len(raw_results))
    c2.metric("Con prediccion", len(history))
    c3.metric("Picks ML cerrados", len(moneyline_rows))
    c4.metric("Verdes ML", green)
    c5.metric("Rojos ML", red)
    c6.metric("Acierto ML", format_pct(accuracy_pct(moneyline_rows)))
    c7.metric("Ganancia ML", format_units(ml_profit))

    hist_tabs = st.tabs(["Moneyline", "Combinadas", "Cobertura"])
    with hist_tabs[0]:
        if moneyline_rows.empty:
            st.info("No hay picks moneyline cerrados con los filtros actuales.")
        else:
            round_options = ["Todas", *sorted(moneyline_rows["Ronda historial"].dropna().unique())] if "Ronda historial" in moneyline_rows.columns else ["Todas"]
            selected_round = st.selectbox("Ronda", round_options, key="history_ml_round")
            display_rows = moneyline_rows
            if selected_round != "Todas":
                display_rows = display_rows[display_rows["Ronda historial"] == selected_round]
            columns = ["Resultado pick", "Ronda historial", "Fecha local", "Partido", "Mercado", "Jugador", "Prob. modelo", "Cuota", "Stake", "Ganancia", "Prob. cuota", "Edge", "Ganador", "Score"]
            table = sort_table_controls(display_rows[[col for col in columns if col in display_rows.columns]], "history_ml_table", "Fecha local")
            st.dataframe(style_market_table(table), use_container_width=True, hide_index=True)

    with hist_tabs[1]:
        parlay_profit = parlay_summary["Ganancia"].sum() if "Ganancia" in parlay_summary.columns and not parlay_summary.empty else 0
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Combinadas cerradas", len(parlay_summary))
        c2.metric("Verdes", parlay_green)
        c3.metric("Rojos", parlay_red)
        c4.metric("Acierto", format_pct(accuracy_pct(parlay_summary)))
        c5.metric("Ganancia", format_units(parlay_profit))
        if parlay_summary.empty:
            st.info("No hubo combinadas recomendadas en las rondas cerradas con los filtros actuales.")
        else:
            summary_table = sort_table_controls(parlay_summary, "history_parlay_summary", "Dia")
            st.dataframe(style_market_table(summary_table), use_container_width=True, hide_index=True)
            with st.expander("Detalle de picks por combinada"):
                detail_columns = ["Resultado combinada", "Resultado pick", "Dia", "Fecha local", "Partido", "Grupo", "Mercado", "Jugador", "Prob. modelo", "Cuota", "Edge", "Ganador", "Score"]
                detail_table = sort_table_controls(parlay_details[[col for col in detail_columns if col in parlay_details.columns]], "history_parlay_detail", "Dia")
                st.dataframe(style_market_table(detail_table), use_container_width=True, hide_index=True)

    with hist_tabs[2]:
        if missing_prediction_results.empty:
            st.info("Todas las filas de resultados historicos tienen prediccion asociada.")
        else:
            st.warning("Hay partidos jugados sin prediccion asociada; cuentan como resultado del torneo, pero no como verde/rojo del modelo.")
            coverage_columns = {
                "Ronda historial": "Ronda",
                "result_date": "Fecha",
                "player1": "Jugador 1",
                "player2": "Jugador 2",
                "winner": "Ganador",
                "score": "Score",
            }
            st.dataframe(friendly_table(missing_prediction_results, coverage_columns), use_container_width=True, hide_index=True)


def recommendations_view() -> None:
    target = load_target_config()
    df = load_predictions()
    if df.empty:
        st.warning(f"No hay predicciones disponibles para {target.get('label')}.")
        return

    market_rows = build_market_rows(df, target.get("market_profile", "grand_slam_bo5"))
    value_rows = value_market_rows(market_rows)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Partidos {target.get('label')}", len(df))
    c2.metric("Recomendaciones value", len(value_rows))
    c3.metric("Mejor edge", format_pct(df["edge_recomendado"].max()))
    c4.metric("Modelo", target.get("model_label", MODEL_LABEL))

    pred_tabs = st.tabs(["Recomendaciones", "Combinada", "Resumen", "Mercados", "Detalle"])
    with pred_tabs[0]:
        rec_rows, _ = tournament_filter(value_rows, "recommendations_tournament")
        rec_rows, _ = day_filter(rec_rows, "recommendations_day")
        if value_rows.empty:
            st.info("Sin apuestas con probabilidad, edge y Kelly suficientes. Evitamos picks tipo 50/50 aunque el modelo marque una leve ventaja.")
        else:
            rec_columns = ["Resultado pick", "Fecha local", "Torneo", "Partido", "Grupo", "Mercado", "Jugador", "Recomendada por", "Prob. modelo", "Cuota", "Prob. cuota", "Edge", "Ganador", "Score"]
            if rec_rows.empty:
                st.info("No hay recomendaciones para los filtros actuales.")
            else:
                rec_table = sort_table_controls(rec_rows[rec_columns], "recommendations_table", "Fecha local")
                st.dataframe(style_market_table(rec_table), use_container_width=True, hide_index=True)

    with pred_tabs[1]:
        parlay_source_rows, _ = tournament_filter(market_rows, "parlay_tournament")
        parlay_source_rows, selected_day = day_filter(parlay_source_rows, "parlay_day")
        parlay_rows = suggested_parlay(parlay_source_rows)
        if parlay_rows.empty:
            st.info("No hay combinada que cumpla 2-4 picks, cuotas individuales 1.10-1.50 y cuota total 1.60-2.00 con los filtros actuales.")
        else:
            final_odds = parlay_rows["Cuota"].prod()
            c1, c2 = st.columns(2)
            c1.metric("Picks", len(parlay_rows))
            c2.metric("Cuota combinada", f"{final_odds:.2f}")
            parlay_columns = ["Resultado pick", "Fecha local", "Torneo", "Partido", "Grupo", "Mercado", "Jugador", "Recomendada por", "Prob. modelo", "Cuota", "Prob. cuota", "Edge", "Ganador", "Score"]
            parlay_table = sort_table_controls(parlay_rows[parlay_columns], "parlay_table", "Prob. modelo")
            st.dataframe(style_market_table(parlay_table), use_container_width=True, hide_index=True)

    with pred_tabs[2]:
        main_columns = {
            "fecha_local": "Fecha local",
            "torneo": "Torneo",
            "jugador_1": "Jugador 1",
            "jugador_2": "Jugador 2",
            "odds1_avg": "Cuota J1",
            "odds2_avg": "Cuota J2",
            "prob_modelo_j1": "Prob. J1",
            "prob_modelo_j2": "Prob. J2",
            "prob_modelo_j1_set": "J1 gana set",
            "prob_modelo_j2_set": "J2 gana set",
            "prob_modelo_j1_2_0": "J1 2-0",
            "prob_modelo_j2_2_0": "J2 2-0",
            "prob_modelo_j1_3_0": "J1 3-0",
            "prob_modelo_j2_3_0": "J2 3-0",
            "prob_modelo_j1_minus_1_5": "J1 -1.5 sets",
            "prob_modelo_j2_minus_1_5": "J2 -1.5 sets",
            "prob_modelo_j1_2sets": "J1 2+ sets",
            "prob_modelo_j2_2sets": "J2 2+ sets",
            "edge_j1": "Edge J1",
            "edge_j2": "Edge J2",
            "recomendacion": "Recomendacion",
            "resultado_recomendacion": "Resultado pick",
            "winner": "Ganador",
            "score": "Score",
        }
        if target.get("market_profile") == "non_grand_slam_bo3":
            for col in [
                "prob_modelo_j1_3_0",
                "prob_modelo_j2_3_0",
                "prob_modelo_j1_minus_1_5",
                "prob_modelo_j2_minus_1_5",
                "prob_modelo_j1_2sets",
                "prob_modelo_j2_2sets",
            ]:
                main_columns.pop(col, None)
        summary_df = df
        summary_df, _ = tournament_filter(summary_df, "summary_tournament")
        summary_df, _ = day_filter(summary_df, "summary_day")
        main_table = friendly_table(summary_df, main_columns)
        main_table = sort_table_controls(main_table, "summary_table", "Fecha local")
        st.dataframe(style_recommendations(main_table), use_container_width=True, hide_index=True)

    with pred_tabs[3]:
        if target.get("market_profile") == "non_grand_slam_bo3":
            group_names = ["Match winner", "Gana set", "2-0"]
        else:
            group_names = ["Match winner", "Gana set", "3-0", "Handicap sets", "Gana 2+ sets"]
        filtered_market_rows = market_rows
        filtered_market_rows, _ = tournament_filter(filtered_market_rows, "markets_tournament")
        filtered_market_rows, _ = day_filter(filtered_market_rows, "markets_day")
        match_order = (
            filtered_market_rows[["fecha_hora_local", "Hora", "Partido"]]
            .dropna(subset=["Partido"])
            .drop_duplicates("Partido")
            .sort_values(["fecha_hora_local", "Hora", "Partido"])
        )
        match_options = ["Todos", *match_order["Partido"].tolist()]
        selected_market_match = st.selectbox("Partido", match_options, key="markets_match_filter")
        if selected_market_match != "Todos":
            filtered_market_rows = filtered_market_rows[filtered_market_rows["Partido"] == selected_market_match]
        tabs = st.tabs(group_names)
        for tab, group in zip(tabs, group_names):
            with tab:
                group_rows = filtered_market_rows[filtered_market_rows["Grupo"] == group].sort_values(["fecha_hora_local", "Hora", "Partido", "Mercado"])
                display_columns = ["Resultado pick", "Fecha local", "Torneo", "Partido", "Mercado", "Jugador", "Prob. modelo", "Cuota", "Prob. cuota", "Edge", "Kelly", "Ganador", "Score"]
                market_table = sort_table_controls(group_rows[display_columns], f"markets_{group}", "Fecha local")
                st.dataframe(style_market_table(market_table), use_container_width=True, hide_index=True)

    with pred_tabs[4]:
        detail_df, _ = tournament_filter(df, "detail_tournament")
        detail_df, _ = day_filter(detail_df, "detail_day")
        if detail_df.empty:
            st.info("No hay partidos para los filtros actuales.")
            return
        selected = st.selectbox("Partido", [f"{r.jugador_1} vs {r.jugador_2}" for r in detail_df.itertuples()])
        row = detail_df.iloc[[i for i, r in enumerate(detail_df.itertuples()) if f"{r.jugador_1} vs {r.jugador_2}" == selected][0]]
        left, right = st.columns(2)
        with left:
            st.markdown(f"### {row['jugador_1']}")
            st.metric("Prob. modelo", format_pct(row["prob_modelo_j1"]))
            st.metric("Prob. mercado", format_pct(row["prob_mercado_j1"]))
            st.metric("Edge", format_pct(row["edge_j1"]))
            st.metric("Victorias previas", f"{row.get('jugador_1_porcentaje_victorias_previas', 0):.1%}")
            st.metric("Victorias superficie", f"{row.get('jugador_1_superficie_porcentaje_victorias_previas', 0):.1%}")
            with st.expander("Mas datos"):
                st.write(
                    {
                        "Partidos previos": int(row.get("jugador_1_partidos_previos", 0)),
                        "Victorias ultimos 10": format_pct(row.get("jugador_1_porcentaje_victorias_ultimos_10")),
                        "Tiebreaks ganados": format_pct(row.get("jugador_1_porcentaje_tiebreaks_ganados")),
                        "Dias descanso": row.get("jugador_1_dias_descanso"),
                    }
                )
        with right:
            st.markdown(f"### {row['jugador_2']}")
            st.metric("Prob. modelo", format_pct(row["prob_modelo_j2"]))
            st.metric("Prob. mercado", format_pct(row["prob_mercado_j2"]))
            st.metric("Edge", format_pct(row["edge_j2"]))
            st.metric("Victorias previas", f"{row.get('jugador_2_porcentaje_victorias_previas', 0):.1%}")
            st.metric("Victorias superficie", f"{row.get('jugador_2_superficie_porcentaje_victorias_previas', 0):.1%}")
            with st.expander("Mas datos"):
                st.write(
                    {
                        "Partidos previos": int(row.get("jugador_2_partidos_previos", 0)),
                        "Victorias ultimos 10": format_pct(row.get("jugador_2_porcentaje_victorias_ultimos_10")),
                        "Tiebreaks ganados": format_pct(row.get("jugador_2_porcentaje_tiebreaks_ganados")),
                        "Dias descanso": row.get("jugador_2_dias_descanso"),
                    }
                )

def models_view() -> None:
    target = load_target_config()
    metrics = read_csv(target["training_path"] / "metrics.csv")
    st.subheader(f"{target.get('model_label', MODEL_LABEL)}: metricas de entrenamiento")
    st.dataframe(format_metrics(metrics), use_container_width=True, hide_index=True)
    if target.get("market_profile") == "grand_slam_bo5":
        st.info("Over 3.5 sets fue entrenado como experimento, pero queda fuera del dashboard operativo por mala metrica en test.")
    elif "mas_19_5_games" in set(metrics.get("target", [])):
        st.info("Over 19.5 games fue entrenado como experimento BO3, pero no se muestra en mercados operativos.")


def data_view() -> None:
    target = load_target_config()
    split_summary = read_json(target["model_data_path"] / "split_summary.json")
    dataset_summary = read_json(target["model_data_path"] / "model_dataset.summary.json")
    metrics = read_csv(target["training_path"] / "metrics.csv")
    raw_summaries = []
    if target.get("market_profile") == "grand_slam_bo5":
        raw_dir = BASE / "files" / "processed" / "grand_slam_moneyline" / "raw"
        for summary_path in sorted(raw_dir.glob("grand_slam_*/scrape_summary.json")):
            summary = read_json(summary_path)
            if summary:
                raw_summaries.append(
                    {
                        "Anio": summary.get("year"),
                        "Torneos": summary.get("tournaments", 0),
                        "Partidos": summary.get("matches", 0),
                    }
                )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Partidos {target.get('model_label', MODEL_LABEL)}", split_summary.get("grand_slam_match_rows", split_summary.get("bo3_match_rows", dataset_summary.get("matches", 0))))
    c2.metric("Train", split_summary.get("train_rows", 0))
    c3.metric("Test", split_summary.get("test_rows", split_summary.get("test_completed_rows", 0)))
    c4.metric("Targets entrenados", len(metrics))

    st.subheader(f"Dataset {target.get('model_label', MODEL_LABEL)}")
    summary_rows = [
        {"Concepto": "Periodo train", "Valor": split_summary.get("train_period", "")},
        {"Concepto": "Periodo test", "Valor": split_summary.get("test_period", "")},
        {"Concepto": "Target principal", "Valor": split_summary.get("target", "moneyline")},
        {"Concepto": "Filas modelo", "Valor": dataset_summary.get("model_rows", 0)},
        {"Concepto": "Filas historial jugadores", "Valor": dataset_summary.get("history_rows", 0)},
        {"Concepto": "Eventos historial", "Valor": dataset_summary.get("history_events", 0)},
        {"Concepto": "Filas lesiones", "Valor": dataset_summary.get("injury_rows", 0)},
        {"Concepto": "Filas ranking ATP", "Valor": dataset_summary.get("ranking_rows", 0)},
        {"Concepto": "Partidos con stats SofaScore", "Valor": dataset_summary.get("sofascore_stat_matches", 0)},
        {"Concepto": f"Partidos activos {target.get('label')}", "Valor": len(load_predictions())},
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    if raw_summaries:
        st.subheader("Scrapes base Grand Slam")
        raw_table = pd.DataFrame(raw_summaries)
        raw_table.loc[len(raw_table)] = {
            "Anio": "Total",
            "Torneos": raw_table["Torneos"].sum(),
            "Partidos": raw_table["Partidos"].sum(),
        }
        st.dataframe(raw_table, use_container_width=True, hide_index=True)

    st.subheader("Modelos entrenados")
    model_columns = {
        "descripcion": "Mercado",
        "model": "Modelo",
        "target": "Target",
        "test_accuracy": "Accuracy test",
        "test_brier_score": "Brier test",
        "test_roc_auc": "AUC test",
    }
    st.dataframe(format_model_targets(metrics, model_columns), use_container_width=True, hide_index=True)


def format_model_targets(metrics: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    out = friendly_table(metrics, columns).copy()
    if "Modelo" in out.columns:
        out["Modelo"] = out["Modelo"].replace(
            {
                "regresion_logistica_grand_slam": "Regresion logistica GS",
                "regresion_logistica_bo3": "Regresion logistica BO3",
                "regresion_logistica": "Regresion logistica",
            }
        )
    for col in ["Accuracy test", "Brier test", "AUC test"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(3)
    return out


@st.cache_data
def read_json_cached(path: Path, modified_at: float) -> dict:
    if not path.exists():
        return {}
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def read_json(path: Path) -> dict:
    modified_at = path.stat().st_mtime if path.exists() else 0
    return read_json_cached(path, modified_at)


def format_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    labels = {
        "model": "Modelo",
        "target": "Target",
        "train_accuracy": "Accuracy train",
        "test_accuracy": "Accuracy test",
        "train_log_loss": "Log loss train",
        "test_log_loss": "Log loss test",
        "train_brier_score": "Brier train",
        "test_brier_score": "Brier test",
        "train_roc_auc": "AUC train",
        "test_roc_auc": "AUC test",
    }
    out = df[[col for col in labels if col in df.columns]].rename(columns=labels).copy()
    out["Modelo"] = out["Modelo"].replace(
        {
            "regresion_logistica": "Regresion logistica",
            "regresion_logistica_grand_slam": "Regresion logistica GS",
            "regresion_logistica_bo3": "Regresion logistica BO3",
            "random_forest": "Random Forest",
            "gradient_boosting": "Gradient Boosting",
        }
    )
    for col in out.columns:
        if col not in {"Modelo", "Target"}:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(3)
    return out


if __name__ == "__main__":
    main()
