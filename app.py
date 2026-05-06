from __future__ import annotations

import re
from itertools import combinations
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st


BASE = Path(__file__).parent
MODEL_DATA = BASE / "files" / "processed" / "model_dataset_2025_2026"
TRAINING = BASE / "files" / "processed" / "model_training_2025_2026"
TEMPORAL = BASE / "files" / "processed" / "temporal_validation_2026"
ATP = BASE / "files" / "processed" / "atp_2026"
ATP_2025 = BASE / "files" / "processed" / "atp_2025"
ROME_QUALY = BASE / "files" / "processed" / "rome_2026_qualy"
DASHBOARD_TARGET = BASE / "files" / "processed" / "dashboard_target.json"

MODEL_NAME = "regresion_logistica"
DETAIL_ODDS_FILES = [
    ATP / "upcoming_match_details.csv",
    ATP / "match_details.csv",
]
MODEL_LABEL = "Regresion logistica"
DEFAULT_MIN_MATCH_PROB = 0.65
MIN_SET_PROB = 0.72
MIN_SWEEP_PROB = 0.65
PARLAY_MIN_ODDS = 1.10
PARLAY_MAX_ODDS = 1.50
PARLAY_MIN_TOTAL_ODDS = 1.60
PARLAY_MAX_TOTAL_ODDS = 2.00
DEFAULT_TARGET_CONFIG = {
    "label": "Madrid upcoming",
    "tournament": "Madrid",
    "round": None,
    "predictions_file": "files/processed/model_training_2025_2026/madrid_predictions.csv",
    "upcoming_file": "files/processed/atp_2026/upcoming_matches.csv",
    "local_reference_date": None,
    "raw_today_offset_days": 0,
    "raw_tomorrow_offset_days": 1,
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
    return config


def filter_prediction_target(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    if df.empty:
        return df
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


def load_predictions() -> pd.DataFrame:
    target = load_target_config()
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
    target_detail_odds = target["upcoming_path"].with_name("upcoming_match_details.csv")
    detail_odds = load_detail_odds([target_detail_odds])
    if not detail_odds.empty:
        preds = preds.merge(detail_odds, on="match_id", how="left")

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
    preds["edge_recomendado"] = preds[["edge_j1", "edge_j2"]].max(axis=1)
    preds = add_schedule_columns(preds, target)
    sort_cols = [col for col in ["fecha_hora_local", "start_raw", "edge_recomendado"] if col in preds.columns]
    ascending = [True, True, False][: len(sort_cols)]
    return preds.sort_values(sort_cols, ascending=ascending)


def load_rome_qualy_predictions() -> pd.DataFrame:
    df = read_csv(ROME_QUALY / "qualy_predictions.csv")
    if df.empty:
        return df

    numeric_cols = [
        "prob_gana_jugador_1",
        "prob_gana_jugador_2",
        "odds_jugador_1",
        "odds_jugador_2",
        "cuota_justa_jugador_1",
        "cuota_justa_jugador_2",
        "ev_jugador_1",
        "ev_jugador_2",
        "jugador_1_ranking",
        "jugador_2_ranking",
        "jugador_1_partidos_previos",
        "jugador_2_partidos_previos",
        "jugador_1_superficie_partidos_previos",
        "jugador_2_superficie_partidos_previos",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = to_num(df[col])

    df["ganador_modelo"] = df.apply(
        lambda row: row["jugador_1"] if row["prob_gana_jugador_1"] >= row["prob_gana_jugador_2"] else row["jugador_2"],
        axis=1,
    )
    df["prob_ganador_modelo"] = df[["prob_gana_jugador_1", "prob_gana_jugador_2"]].max(axis=1)
    df["favorito_cuotas"] = df.apply(
        lambda row: row["jugador_1"] if row["odds_jugador_1"] <= row["odds_jugador_2"] else row["jugador_2"],
        axis=1,
    )
    df["alineado_cuotas"] = df["ganador_modelo"] == df["favorito_cuotas"]
    df["mejor_ev"] = df[["ev_jugador_1", "ev_jugador_2"]].max(axis=1)
    df[["recomendacion_apuesta", "motivo_recomendacion"]] = df.apply(
        qualy_recommendation,
        axis=1,
        result_type="expand",
    )
    return df.sort_values(["recomendacion_apuesta", "mejor_ev"], ascending=[True, False])


def implied_probability(odds1: pd.Series, odds2: pd.Series, side: int) -> pd.Series:
    raw1 = 1 / odds1
    raw2 = 1 / odds2
    total = raw1 + raw2
    return (raw1 if side == 1 else raw2) / total


def load_detail_odds(extra_paths: list[Path] | None = None) -> pd.DataFrame:
    frames = []
    keep = [
        "match_id",
        "set_odds_player1_wins_set",
        "set_odds_player2_wins_set",
        "set_odds_player1_wins_2_0",
        "set_odds_player2_wins_2_0",
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
    market_map = {
        "j1_set": ("prob_jugador_1_gana_al_menos_un_set", "set_odds_player1_wins_set"),
        "j2_set": ("prob_jugador_2_gana_al_menos_un_set", "set_odds_player2_wins_set"),
        "j1_2_0": ("prob_jugador_1_gana_2_0", "set_odds_player1_wins_2_0"),
        "j2_2_0": ("prob_jugador_2_gana_2_0", "set_odds_player2_wins_2_0"),
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


def qualy_recommendation(row: pd.Series) -> tuple[str, str]:
    candidates = [
        (1, row.get("jugador_1"), row.get("prob_gana_jugador_1"), row.get("odds_jugador_1"), row.get("cuota_justa_jugador_1"), row.get("ev_jugador_1")),
        (2, row.get("jugador_2"), row.get("prob_gana_jugador_2"), row.get("odds_jugador_2"), row.get("cuota_justa_jugador_2"), row.get("ev_jugador_2")),
    ]
    side, player, prob, odds, fair_odds, ev = max(candidates, key=lambda item: item[5] if pd.notna(item[5]) else -999)
    if pd.isna(ev) or pd.isna(prob) or pd.isna(odds) or pd.isna(fair_odds):
        return "Sin apuesta", "Faltan cuotas o probabilidad del modelo."

    rank = row.get(f"jugador_{side}_ranking")
    prior = row.get(f"jugador_{side}_partidos_previos")
    clay_prior = row.get(f"jugador_{side}_superficie_partidos_previos")
    flags = []
    if pd.isna(rank) or rank >= 10000:
        flags.append("ranking faltante")
    if pd.notna(prior) and prior < 25:
        flags.append("poco historial total")
    if pd.notna(clay_prior) and clay_prior < 10:
        flags.append("poco historial en clay")
    if prob < 0.54:
        flags.append("probabilidad ajustada")
    if row.get("ganador_modelo") != row.get("favorito_cuotas"):
        flags.append("va contra mercado")

    value_text = f"{player}: cuota {odds:.2f} > cuota justa {fair_odds:.2f}, EV {ev:.1%}, prob. modelo {prob:.1%}"
    if ev >= 0.08 and prob >= 0.54 and len(flags) <= 2:
        return f"Apostar {player}", value_text + ("; " + ", ".join(flags) if flags else ".")
    if ev >= 0.08:
        return f"Solo stake chico {player}", value_text + "; " + ", ".join(flags)
    return "Sin apuesta", f"No hay edge suficiente. Mejor EV: {value_text}."


def build_market_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    markets = [
        ("Match winner", "J1 gana partido", "jugador_1", "prob_modelo_j1", "odds1_avg", "prob_mercado_j1", "edge_j1", "kelly_j1", DEFAULT_MIN_MATCH_PROB),
        ("Match winner", "J2 gana partido", "jugador_2", "prob_modelo_j2", "odds2_avg", "prob_mercado_j2", "edge_j2", "kelly_j2", DEFAULT_MIN_MATCH_PROB),
        ("Gana set", "J1 gana al menos un set", "jugador_1", "prob_modelo_j1_set", "set_odds_player1_wins_set", "prob_mercado_j1_set", "edge_j1_set", "kelly_j1_set", MIN_SET_PROB),
        ("Gana set", "J2 gana al menos un set", "jugador_2", "prob_modelo_j2_set", "set_odds_player2_wins_set", "prob_mercado_j2_set", "edge_j2_set", "kelly_j2_set", MIN_SET_PROB),
        ("2-0", "J1 gana 2-0", "jugador_1", "prob_modelo_j1_2_0", "set_odds_player1_wins_2_0", "prob_mercado_j1_2_0", "edge_j1_2_0", "kelly_j1_2_0", MIN_SWEEP_PROB),
        ("2-0", "J2 gana 2-0", "jugador_2", "prob_modelo_j2_2_0", "set_odds_player2_wins_2_0", "prob_mercado_j2_2_0", "edge_j2_2_0", "kelly_j2_2_0", MIN_SWEEP_PROB),
    ]
    for _, match in df.iterrows():
        for group, market, player_col, prob_col, odds_col, implied_col, edge_col, kelly_col, min_prob in markets:
            rows.append(
                {
                    "fecha_hora_local": match.get("fecha_hora_local"),
                    "Fecha local": match.get("fecha_local"),
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
        "prob_modelo_j1_2_0",
        "prob_modelo_j2_2_0",
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
        "J1 2-0",
        "J2 2-0",
        "Edge J1",
        "Edge J2",
    ]:
        if col in display.columns:
            display[col] = display[col].map(format_pct)
    return display


def style_market_table(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for col in ["Prob. modelo", "Prob. cuota", "Edge", "Kelly", "Prob. minima"]:
        if col in display.columns:
            display[col] = display[col].map(format_pct)
    for col in ["Cuota", "Cuota combinada"]:
        if col in display.columns:
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
    return display


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
    page_tabs = st.tabs(["Predicciones", "Roma qualy", "Modelos", "Datos"])
    with page_tabs[0]:
        recommendations_view()
    with page_tabs[1]:
        rome_qualy_view()
    with page_tabs[2]:
        models_view()
    with page_tabs[3]:
        data_view()


def recommendations_view() -> None:
    target = load_target_config()
    df = load_predictions()
    if df.empty:
        st.warning(f"No hay predicciones disponibles para {target.get('label')}.")
        return

    market_rows = build_market_rows(df)
    value_rows = value_market_rows(market_rows)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Partidos {target.get('label')}", len(df))
    c2.metric("Recomendaciones value", len(value_rows))
    c3.metric("Mejor edge", format_pct(df["edge_recomendado"].max()))
    c4.metric("Modelo", MODEL_LABEL)

    pred_tabs = st.tabs(["Recomendaciones", "Combinada", "Resumen", "Mercados", "Detalle"])
    with pred_tabs[0]:
        if value_rows.empty:
            st.info("Sin apuestas con probabilidad, edge y Kelly suficientes. Evitamos picks tipo 50/50 aunque el modelo marque una leve ventaja.")
        else:
            available_days = sorted(pd.to_datetime(value_rows["fecha_hora_local"], errors="coerce").dt.date.dropna().unique())
            day_options = ["Todos", *[day.isoformat() for day in available_days]]
            selected_day = st.selectbox("Dia", day_options, key="recommendations_day")
            rec_rows = value_rows
            if selected_day != "Todos":
                rec_day = date.fromisoformat(selected_day)
                rec_rows = rec_rows[pd.to_datetime(rec_rows["fecha_hora_local"], errors="coerce").dt.date == rec_day]
            rec_columns = ["Fecha local", "Hora", "Partido", "Grupo", "Mercado", "Jugador", "Recomendada por", "Prob. modelo", "Cuota", "Prob. cuota", "Edge"]
            if rec_rows.empty:
                st.info("No hay recomendaciones para el dia seleccionado con los filtros actuales.")
            else:
                rec_table = sort_table_controls(rec_rows[rec_columns], "recommendations_table", "Fecha local")
                st.dataframe(style_market_table(rec_table), use_container_width=True, hide_index=True)

    with pred_tabs[1]:
        available_days = sorted(pd.to_datetime(market_rows["fecha_hora_local"], errors="coerce").dt.date.dropna().unique())
        day_options = ["Todos", *[day.isoformat() for day in available_days]]
        selected_day = st.selectbox("Dia", day_options)
        parlay_day = None if selected_day == "Todos" else date.fromisoformat(selected_day)
        parlay_rows = suggested_parlay(market_rows, parlay_day)
        if parlay_rows.empty:
            st.info("No hay combinada que cumpla 2-4 picks, cuotas individuales 1.10-1.50 y cuota total 1.60-2.00 con los filtros actuales.")
        else:
            final_odds = parlay_rows["Cuota"].prod()
            c1, c2 = st.columns(2)
            c1.metric("Picks", len(parlay_rows))
            c2.metric("Cuota combinada", f"{final_odds:.2f}")
            parlay_columns = ["Fecha local", "Hora", "Partido", "Grupo", "Mercado", "Jugador", "Recomendada por", "Prob. modelo", "Cuota", "Prob. cuota", "Edge"]
            parlay_table = sort_table_controls(parlay_rows[parlay_columns], "parlay_table", "Prob. modelo")
            st.dataframe(style_market_table(parlay_table), use_container_width=True, hide_index=True)

    with pred_tabs[2]:
        main_columns = {
            "fecha_local": "Fecha local",
            "start_raw": "Hora sitio",
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
            "edge_j1": "Edge J1",
            "edge_j2": "Edge J2",
            "recomendacion": "Recomendacion",
        }
        main_table = friendly_table(df, main_columns)
        main_table = sort_table_controls(main_table, "summary_table", "Fecha local")
        st.dataframe(style_recommendations(main_table), use_container_width=True, hide_index=True)

    with pred_tabs[3]:
        tabs = st.tabs(["Match winner", "Gana set", "2-0"])
        for tab, group in zip(tabs, ["Match winner", "Gana set", "2-0"]):
            with tab:
                group_rows = market_rows[market_rows["Grupo"] == group].sort_values(["fecha_hora_local", "Hora", "Partido", "Mercado"])
                display_columns = ["Fecha local", "Hora", "Partido", "Mercado", "Jugador", "Prob. modelo", "Cuota", "Prob. cuota", "Edge", "Kelly"]
                market_table = sort_table_controls(group_rows[display_columns], f"markets_{group}", "Fecha local")
                st.dataframe(style_market_table(market_table), use_container_width=True, hide_index=True)

    with pred_tabs[4]:
        selected = st.selectbox("Partido", [f"{r.jugador_1} vs {r.jugador_2}" for r in df.itertuples()])
        row = df.iloc[[i for i, r in enumerate(df.itertuples()) if f"{r.jugador_1} vs {r.jugador_2}" == selected][0]]
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


def rome_qualy_view() -> None:
    df = load_rome_qualy_predictions()
    if df.empty:
        st.warning("No hay predicciones de Roma qualy disponibles.")
        return

    bets = df[df["recomendacion_apuesta"].str.startswith("Apostar", na=False)]
    small = df[df["recomendacion_apuesta"].str.startswith("Solo stake chico", na=False)]
    aligned = int(df["alineado_cuotas"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Partidos Roma qualy", len(df))
    c2.metric("Alineados con cuotas", f"{aligned}/{len(df)}")
    c3.metric("Apuestas", len(bets))
    c4.metric("Stake chico", len(small))

    st.subheader("Todos los partidos")
    table = df.copy()
    table["prob_gana_jugador_1"] = table["prob_gana_jugador_1"].map(format_pct)
    table["prob_gana_jugador_2"] = table["prob_gana_jugador_2"].map(format_pct)
    table["prob_ganador_modelo"] = table["prob_ganador_modelo"].map(format_pct)
    table["ev_jugador_1"] = table["ev_jugador_1"].map(format_pct)
    table["ev_jugador_2"] = table["ev_jugador_2"].map(format_pct)
    for col in ["odds_jugador_1", "odds_jugador_2", "cuota_justa_jugador_1", "cuota_justa_jugador_2"]:
        table[col] = table[col].map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
    display_columns = {
        "fecha": "Fecha",
        "ronda": "Ronda",
        "jugador_1": "Jugador 1",
        "jugador_2": "Jugador 2",
        "prob_gana_jugador_1": "Prob. J1",
        "prob_gana_jugador_2": "Prob. J2",
        "odds_jugador_1": "Cuota real J1",
        "odds_jugador_2": "Cuota real J2",
        "cuota_justa_jugador_1": "Cuota justa J1",
        "cuota_justa_jugador_2": "Cuota justa J2",
        "ganador_modelo": "Ganador modelo",
        "prob_ganador_modelo": "Prob. ganador",
        "favorito_cuotas": "Favorito cuotas",
        "ev_jugador_1": "EV J1",
        "ev_jugador_2": "EV J2",
        "recomendacion_apuesta": "Recomendacion",
        "motivo_recomendacion": "Por que",
    }
    st.dataframe(
        friendly_table(table, display_columns),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Datos de soporte"):
        support_columns = {
            "jugador_1": "Jugador 1",
            "jugador_1_ranking": "Ranking J1",
            "jugador_1_partidos_previos": "Partidos previos J1",
            "jugador_1_superficie_partidos_previos": "Partidos clay J1",
            "jugador_2": "Jugador 2",
            "jugador_2_ranking": "Ranking J2",
            "jugador_2_partidos_previos": "Partidos previos J2",
            "jugador_2_superficie_partidos_previos": "Partidos clay J2",
            "alineado_cuotas": "Alineado con cuotas",
        }
        st.dataframe(friendly_table(df, support_columns), use_container_width=True, hide_index=True)


def models_view() -> None:
    metrics = read_csv(TRAINING / "metrics.csv")
    temporal = read_csv(TEMPORAL / "metrics.csv")
    st.subheader("Split Barcelona + Munich")
    st.dataframe(format_metrics(metrics), use_container_width=True, hide_index=True)
    st.subheader("Validacion temporal: Monte Carlo + Barcelona + Munich")
    st.dataframe(format_metrics(temporal), use_container_width=True, hide_index=True)
    st.info("Usamos regresion logistica por estabilidad y menor sobreajuste. Random Forest queda como candidato alternativo para futuras calibraciones.")


def data_view() -> None:
    target = load_target_config()
    split_summary = read_json(MODEL_DATA / "split_summary.json")
    dataset_summary = read_json(MODEL_DATA / "model_dataset.summary.json")
    ranking_summary = read_json(BASE / "files" / "processed" / "atp_rankings" / "ranking_summary.json")
    scrape_summary = read_json(ATP / "scrape_summary.json")
    scrape_2025_summary = read_json(ATP_2025 / "scrape_summary.json")
    c1, c2, c3 = st.columns(3)
    c1.metric("Train", split_summary.get("train_rows", 0))
    c2.metric("Test", split_summary.get("test_total_rows", 0))
    c3.metric("Historial jugadores", dataset_summary.get("history_rows", 0))

    st.subheader("Resumen de datos")
    summary_rows = [
        {"Concepto": "Torneos ATP 2025", "Valor": scrape_2025_summary.get("tournaments", 0)},
        {"Concepto": "Partidos ATP 2025", "Valor": scrape_2025_summary.get("matches", 0)},
        {"Concepto": "Torneos ATP 2026", "Valor": scrape_summary.get("tournaments", 0)},
        {"Concepto": "Partidos ATP 2026", "Valor": scrape_summary.get("matches", 0)},
        {"Concepto": "Jugadores con ranking ATP", "Valor": ranking_summary.get("history_players", 0)},
        {"Concepto": "Filas de historial", "Valor": dataset_summary.get("history_rows", 0)},
        {"Concepto": "Filas de ranking", "Valor": dataset_summary.get("ranking_rows", 0)},
        {"Concepto": f"Partidos activos {target.get('label')}", "Valor": len(load_predictions())},
        {"Concepto": "Partidos proximos Roma qualy", "Valor": len(read_csv(ROME_QUALY / "qualy_predictions.csv"))},
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


@st.cache_data
def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    import json

    return json.loads(path.read_text(encoding="utf-8"))


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
