from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


BASE = Path(__file__).parent
MODEL_DATA = BASE / "files" / "processed" / "model_dataset_2025_2026"
TRAINING = BASE / "files" / "processed" / "model_training_2025_2026"
TEMPORAL = BASE / "files" / "processed" / "temporal_validation_2025_2026"
ATP = BASE / "files" / "processed" / "atp_2026"
ATP_2025 = BASE / "files" / "processed" / "atp_2025"

MODEL_NAME = "regresion_logistica"
DETAIL_ODDS_FILES = [
    ATP / "upcoming_match_details.csv",
    ATP / "match_details.csv",
]
MODEL_LABEL = "Regresión logística"


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


def load_predictions() -> pd.DataFrame:
    preds = read_csv(TRAINING / "madrid_predictions.csv")
    upcoming = read_csv(ATP / "upcoming_matches.csv")
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
        ]
        preds = preds.merge(upcoming[[c for c in keep if c in upcoming.columns]], on="match_id", how="left")
    detail_odds = load_detail_odds()
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
    preds["recomendacion"] = preds.apply(recommendation, axis=1)
    preds["edge_recomendado"] = preds[["edge_j1", "edge_j2"]].max(axis=1)
    preds["favorito_modelo"] = preds.apply(
        lambda row: row["jugador_1"] if row["prob_modelo_j1"] >= row["prob_modelo_j2"] else row["jugador_2"],
        axis=1,
    )
    preds["prob_favorito_modelo"] = preds[["prob_modelo_j1", "prob_modelo_j2"]].max(axis=1)
    add_set_market_columns(preds)
    return preds.sort_values("edge_recomendado", ascending=False)


def implied_probability(odds1: pd.Series, odds2: pd.Series, side: int) -> pd.Series:
    raw1 = 1 / odds1
    raw2 = 1 / odds2
    total = raw1 + raw2
    return (raw1 if side == 1 else raw2) / total


def load_detail_odds() -> pd.DataFrame:
    frames = []
    keep = [
        "match_id",
        "set_odds_player1_wins_set",
        "set_odds_player2_wins_set",
        "set_odds_player1_wins_2_0",
        "set_odds_player2_wins_2_0",
    ]
    for path in DETAIL_ODDS_FILES:
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
    max_kelly = st.session_state.get("max_kelly", 0.05)
    candidates = []
    for side in (1, 2):
        edge = row[f"edge_j{side}"]
        kelly = row[f"kelly_j{side}"]
        prob = row[f"prob_modelo_j{side}"]
        player = row[f"jugador_{side}"]
        odds = row[f"odds{side}_avg"]
        if pd.notna(edge) and pd.notna(kelly) and edge >= min_edge and kelly > 0:
            stake = min(max(kelly, 0), max_kelly)
            candidates.append((edge, f"{player} @ {odds:.2f} | edge {edge:.1%} | stake {stake:.1%}"))
    if not candidates:
        return "Sin apuesta"
    return max(candidates, key=lambda item: item[0])[1]


def build_market_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    markets = [
        ("J1 gana partido", "jugador_1", "prob_modelo_j1", "odds1_avg", "prob_mercado_j1", "edge_j1", "kelly_j1"),
        ("J2 gana partido", "jugador_2", "prob_modelo_j2", "odds2_avg", "prob_mercado_j2", "edge_j2", "kelly_j2"),
        ("J1 gana al menos un set", "jugador_1", "prob_modelo_j1_set", "set_odds_player1_wins_set", "prob_mercado_j1_set", "edge_j1_set", "kelly_j1_set"),
        ("J2 gana al menos un set", "jugador_2", "prob_modelo_j2_set", "set_odds_player2_wins_set", "prob_mercado_j2_set", "edge_j2_set", "kelly_j2_set"),
        ("J1 gana 2-0", "jugador_1", "prob_modelo_j1_2_0", "set_odds_player1_wins_2_0", "prob_mercado_j1_2_0", "edge_j1_2_0", "kelly_j1_2_0"),
        ("J2 gana 2-0", "jugador_2", "prob_modelo_j2_2_0", "set_odds_player2_wins_2_0", "prob_mercado_j2_2_0", "edge_j2_2_0", "kelly_j2_2_0"),
    ]
    for _, match in df.iterrows():
        for market, player_col, prob_col, odds_col, implied_col, edge_col, kelly_col in markets:
            rows.append(
                {
                    "Hora": match.get("start_raw"),
                    "Partido": f"{match.get('jugador_1')} vs {match.get('jugador_2')}",
                    "Mercado": market,
                    "Jugador": match.get(player_col),
                    "Prob. modelo": match.get(prob_col),
                    "Cuota": match.get(odds_col),
                    "Prob. cuota": match.get(implied_col),
                    "Edge": match.get(edge_col),
                    "Kelly": match.get(kelly_col),
                }
            )
    return pd.DataFrame(rows)


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
    ]:
        if col in display.columns:
            display[col] = display[col].map(format_pct)
    return display


def style_market_table(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for col in ["Prob. modelo", "Prob. cuota", "Edge", "Kelly"]:
        if col in display.columns:
            display[col] = display[col].map(format_pct)
    if "Cuota" in display.columns:
        display["Cuota"] = display["Cuota"].map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
    return display


def friendly_table(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    available = [col for col in columns if col in df.columns]
    return df[available].rename(columns={col: columns[col] for col in available})


def main() -> None:
    st.title("Tennis Predictions")
    st.caption("Modelo operativo: Regresión logística v1. Las cuotas se usan solo para evaluar value, no como feature de entrenamiento.")

    st.sidebar.header("Parámetros")
    st.session_state["min_edge"] = st.sidebar.slider("Edge mínimo", 0.00, 0.20, 0.05, 0.01)
    st.session_state["max_kelly"] = st.sidebar.slider("Stake máximo sugerido", 0.01, 0.10, 0.05, 0.01)
    page = st.sidebar.radio("Vista", ["Recomendaciones", "Modelos", "Datos"])

    if page == "Recomendaciones":
        recommendations_view()
    elif page == "Modelos":
        models_view()
    else:
        data_view()


def recommendations_view() -> None:
    df = load_predictions()
    if df.empty:
        st.warning("No hay predicciones disponibles.")
        return

    bets = df[df["recomendacion"] != "Sin apuesta"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Partidos Madrid", len(df))
    c2.metric("Recomendaciones", len(bets))
    c3.metric("Mejor edge", format_pct(df["edge_recomendado"].max()))
    c4.metric("Modelo", MODEL_LABEL)

    st.subheader("Partidos a predecir")
    main_columns = {
        "start_raw": "Hora",
        "jugador_1": "Jugador 1",
        "jugador_2": "Jugador 2",
        "odds1_avg": "Cuota J1",
        "odds2_avg": "Cuota J2",
        "favorito_modelo": "Favorito modelo",
        "prob_favorito_modelo": "Prob. favorito",
        "edge_j1": "Edge J1",
        "edge_j2": "Edge J2",
        "recomendacion": "Recomendación",
    }
    main_table = style_recommendations(friendly_table(df, main_columns).rename(columns={v: k for k, v in main_columns.items()}))
    main_table = main_table.rename(columns=main_columns)
    st.dataframe(main_table, use_container_width=True, hide_index=True)

    st.subheader("Mercados derivados")
    market_rows = build_market_rows(df)
    min_edge = st.session_state.get("min_edge", 0.05)
    value_rows = market_rows[
        market_rows["Cuota"].notna()
        & market_rows["Edge"].notna()
        & (market_rows["Edge"] >= min_edge)
        & (market_rows["Kelly"] > 0)
    ].sort_values("Edge", ascending=False)
    if value_rows.empty:
        st.caption("Sin value detectado en mercados derivados con cuotas disponibles.")
    else:
        st.dataframe(style_market_table(value_rows), use_container_width=True, hide_index=True)
    with st.expander("Ver todos los mercados"):
        st.dataframe(
            style_market_table(market_rows.sort_values(["Partido", "Mercado"])),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Ver probabilidades completas"):
        detail_columns = {
            "jugador_1": "Jugador 1",
            "jugador_2": "Jugador 2",
            "prob_modelo_j1": "Prob. modelo J1",
            "prob_modelo_j2": "Prob. modelo J2",
            "prob_modelo_j1_set": "Prob. J1 gana set",
            "prob_modelo_j2_set": "Prob. J2 gana set",
            "prob_modelo_j1_2_0": "Prob. J1 2-0",
            "prob_modelo_j2_2_0": "Prob. J2 2-0",
            "prob_mercado_j1": "Prob. mercado J1",
            "prob_mercado_j2": "Prob. mercado J2",
            "kelly_j1": "Kelly J1",
            "kelly_j2": "Kelly J2",
        }
        table = style_recommendations(
            friendly_table(df, detail_columns).rename(columns={v: k for k, v in detail_columns.items()})
        ).rename(columns=detail_columns)
        st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("Detalle")
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
        with st.expander("Más datos"):
            st.write(
                {
                    "Partidos previos": int(row.get("jugador_1_partidos_previos", 0)),
                    "Victorias últimos 10": format_pct(row.get("jugador_1_porcentaje_victorias_ultimos_10")),
                    "Tiebreaks ganados": format_pct(row.get("jugador_1_porcentaje_tiebreaks_ganados")),
                    "Días descanso": row.get("jugador_1_dias_descanso"),
                }
            )
    with right:
        st.markdown(f"### {row['jugador_2']}")
        st.metric("Prob. modelo", format_pct(row["prob_modelo_j2"]))
        st.metric("Prob. mercado", format_pct(row["prob_mercado_j2"]))
        st.metric("Edge", format_pct(row["edge_j2"]))
        st.metric("Victorias previas", f"{row.get('jugador_2_porcentaje_victorias_previas', 0):.1%}")
        st.metric("Victorias superficie", f"{row.get('jugador_2_superficie_porcentaje_victorias_previas', 0):.1%}")
        with st.expander("Más datos"):
            st.write(
                {
                    "Partidos previos": int(row.get("jugador_2_partidos_previos", 0)),
                    "Victorias últimos 10": format_pct(row.get("jugador_2_porcentaje_victorias_ultimos_10")),
                    "Tiebreaks ganados": format_pct(row.get("jugador_2_porcentaje_tiebreaks_ganados")),
                    "Días descanso": row.get("jugador_2_dias_descanso"),
                }
            )


def models_view() -> None:
    metrics = read_csv(TRAINING / "metrics.csv")
    temporal = read_csv(TEMPORAL / "metrics.csv")
    st.subheader("Split Barcelona + Munich")
    st.dataframe(format_metrics(metrics), use_container_width=True, hide_index=True)
    st.subheader("Validación temporal: Monte Carlo + Barcelona + Munich")
    st.dataframe(format_metrics(temporal), use_container_width=True, hide_index=True)
    st.info("Usamos regresión logística por estabilidad y menor sobreajuste. Random Forest queda como candidato alternativo para futuras calibraciones.")


def data_view() -> None:
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
        {"Concepto": "Partidos próximos Madrid", "Valor": split_summary.get("test_madrid_upcoming_rows", 0)},
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
            "regresion_logistica": "Regresión logística",
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
