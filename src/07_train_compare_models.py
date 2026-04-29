from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
# from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DROP_COLUMNS = {
    "match_id",
    "fecha",
    "torneo",
    "jugador_1",
    "jugador_2",
    "sets_jugador_1",
    "sets_jugador_2",
    "games_jugador_1",
    "games_jugador_2",
    "target_gana_jugador_1",
}
CATEGORICAL_COLUMNS = ["categoria_torneo", "superficie", "ronda"]
LOGISTIC_NUMERIC_PREFIXES = ("diferencia_", "partidos_previos_entre_ellos")
TARGETS = {
    "gana_jugador_1": "Gana jugador 1",
    "jugador_1_gana_2_0": "Jugador 1 gana 2-0",
    "jugador_2_gana_2_0": "Jugador 2 gana 2-0",
    # "mas_19_5_games": "Mas de 19.5 games",
}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def target_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("target_gana_jugador_1") in {"0", "1", 0, 1}]


def build_x(rows: list[dict], feature_columns: list[str]) -> pd.DataFrame:
    x_rows = pd.DataFrame([{col: row.get(col, "") for col in feature_columns} for row in rows])
    for col in x_rows.columns:
        if col not in CATEGORICAL_COLUMNS:
            x_rows[col] = pd.to_numeric(x_rows[col], errors="coerce")
    return x_rows


def target_value(row: dict, target: str) -> int | None:
    sets1 = to_int(row.get("sets_jugador_1"))
    sets2 = to_int(row.get("sets_jugador_2"))
    games1 = to_int(row.get("games_jugador_1"))
    games2 = to_int(row.get("games_jugador_2"))
    if target == "gana_jugador_1":
        value = row.get("target_gana_jugador_1")
        return int(value) if value in {"0", "1", 0, 1} else None
    if sets1 is None or sets2 is None:
        return None
    if target == "jugador_1_gana_al_menos_un_set":
        return int(sets1 >= 1)
    if target == "jugador_2_gana_al_menos_un_set":
        return int(sets2 >= 1)
    if target == "jugador_1_gana_2_0":
        return int(sets1 == 2 and sets2 == 0)
    if target == "jugador_2_gana_2_0":
        return int(sets2 == 2 and sets1 == 0)
    if target == "mas_19_5_games":
        if games1 is None or games2 is None:
            return None
        return int(games1 + games2 > 19.5)
    raise ValueError(f"Unknown target: {target}")


def to_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        return None


def build_y(rows: list[dict], target: str) -> np.ndarray:
    values = [target_value(row, target) for row in rows]
    if any(value is None for value in values):
        raise ValueError(f"Target {target} has missing values in completed rows")
    return np.array([int(value) for value in values])


def infer_feature_columns(rows: list[dict]) -> tuple[list[str], list[str], list[str]]:
    all_columns = sorted({key for row in rows for key in row.keys()})
    feature_columns = [col for col in all_columns if col not in DROP_COLUMNS]
    categorical = [col for col in CATEGORICAL_COLUMNS if col in feature_columns]
    numeric = [col for col in feature_columns if col not in categorical]
    return feature_columns, numeric, categorical


def make_preprocessor(numeric: list[str], categorical: list[str], scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ],
        remainder="drop",
    )


def logistic_columns(numeric: list[str]) -> list[str]:
    return [col for col in numeric if col.startswith(LOGISTIC_NUMERIC_PREFIXES)]


def build_logistic_model(numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor(numeric, categorical, scale_numeric=True)),
            ("model", LogisticRegression(max_iter=5000, C=1.5, random_state=42)),
        ]
    )


def build_models(numeric: list[str], categorical: list[str]) -> dict[str, Pipeline]:
    logistic_numeric = logistic_columns(numeric)
    return {
        "regresion_logistica": build_logistic_model(logistic_numeric, categorical),
        # "random_forest": Pipeline(
        #     [
        #         ("prep", make_preprocessor(numeric, categorical, scale_numeric=False)),
        #         (
        #             "model",
        #             RandomForestClassifier(
        #                 n_estimators=500,
        #                 max_depth=6,
        #                 min_samples_leaf=10,
        #                 class_weight="balanced_subsample",
        #                 random_state=42,
        #             ),
        #         ),
        #     ]
        # ),
        # "gradient_boosting": Pipeline(
        #     [
        #         ("prep", make_preprocessor(numeric, categorical, scale_numeric=False)),
        #         (
        #             "model",
        #             GradientBoostingClassifier(
        #                 n_estimators=150,
        #                 learning_rate=0.04,
        #                 max_depth=2,
        #                 min_samples_leaf=15,
        #                 random_state=42,
        #             ),
        #         ),
        #     ]
        # ),
    }


def transformed_feature_names(model: Pipeline, numeric: list[str], categorical: list[str]) -> list[str]:
    preprocessor = model.named_steps["prep"]
    names: list[str] = []
    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name == "num":
            names.extend(columns)
        elif transformer_name == "cat":
            onehot = transformer.named_steps["onehot"]
            names.extend(onehot.get_feature_names_out(columns).tolist())
    return names


def write_logistic_importance(path: Path, model: Pipeline, numeric: list[str], categorical: list[str]) -> None:
    coefficients = model.named_steps["model"].coef_[0]
    names = transformed_feature_names(model, numeric, categorical)
    rows = []
    for name, coefficient in zip(names, coefficients):
        rows.append(
            {
                "variable": name,
                "coeficiente": float(coefficient),
                "importancia_abs": abs(float(coefficient)),
                "direccion": "sube_probabilidad_objetivo" if coefficient > 0 else "baja_probabilidad_objetivo" if coefficient < 0 else "neutral",
            }
        )
    rows.sort(key=lambda row: row["importancia_abs"], reverse=True)
    write_csv(path, rows)


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float | None]:
    pred = (proba >= 0.5).astype(int)
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, proba)),
        "positive_rate": float(np.mean(y_true)),
        "rows": int(len(y_true)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def add_predictions(rows: list[dict], feature_columns: list[str], models: dict[str, Pipeline]) -> list[dict]:
    x_rows = build_x(rows, feature_columns)
    output = [dict(row) for row in rows]
    for name, model in models.items():
        proba = model.predict_proba(x_rows)[:, 1]
        for row, value in zip(output, proba):
            row[f"prob_{name}"] = float(value)
    for row in output:
        if "prob_jugador_2_gana_2_0" in row:
            row["prob_jugador_1_gana_al_menos_un_set"] = 1 - float(row["prob_jugador_2_gana_2_0"])
        if "prob_jugador_1_gana_2_0" in row:
            row["prob_jugador_2_gana_al_menos_un_set"] = 1 - float(row["prob_jugador_1_gana_2_0"])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare tennis prediction models.")
    parser.add_argument("--train", default="files/processed/model_dataset_2026/train.csv")
    parser.add_argument("--test", default="files/processed/model_dataset_2026/test_barcelona_munich.csv")
    parser.add_argument("--madrid", default="files/processed/model_dataset_2026/test_madrid_upcoming.csv")
    parser.add_argument("--out-dir", default="files/processed/model_training_2026")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = target_rows(read_csv(Path(args.train)))
    test_rows = target_rows(read_csv(Path(args.test)))
    madrid_rows = read_csv(Path(args.madrid))
    feature_columns, numeric, categorical = infer_feature_columns(train_rows)
    x_train = build_x(train_rows, feature_columns)
    x_test = build_x(test_rows, feature_columns)

    active_numeric = logistic_columns(numeric)
    active_features = active_numeric + categorical
    metrics_rows: list[dict] = []
    fitted_models: dict[str, Pipeline] = {}
    for target, label in TARGETS.items():
        y_train = build_y(train_rows, target)
        y_test = build_y(test_rows, target)
        model = build_logistic_model(active_numeric, categorical)
        model.fit(x_train, y_train)
        fitted_models[target] = model
        train_proba = model.predict_proba(x_train)[:, 1]
        test_proba = model.predict_proba(x_test)[:, 1]
        train_metrics = evaluate(y_train, train_proba)
        test_metrics = evaluate(y_test, test_proba)
        metrics = {
            "target": target,
            "descripcion": label,
            "train_rows": train_metrics["rows"],
            "test_rows": test_metrics["rows"],
            "train_positive_rate": train_metrics["positive_rate"],
            "test_positive_rate": test_metrics["positive_rate"],
            "train_accuracy": train_metrics["accuracy"],
            "train_log_loss": train_metrics["log_loss"],
            "train_brier_score": train_metrics["brier_score"],
            "train_roc_auc": train_metrics["roc_auc"],
            "test_accuracy": test_metrics["accuracy"],
            "test_log_loss": test_metrics["log_loss"],
            "test_brier_score": test_metrics["brier_score"],
            "test_roc_auc": test_metrics["roc_auc"],
        }
        metrics_rows.append({"model": "regresion_logistica", **metrics})
        joblib.dump(model, out_dir / f"regresion_logistica_{target}.joblib")
        write_logistic_importance(
            out_dir / f"logistic_feature_importance_{target}.csv",
            model,
            active_numeric,
            categorical,
        )

    write_csv(out_dir / "metrics.csv", metrics_rows)
    (out_dir / "feature_columns.json").write_text(
        json.dumps(
            {
                "features": feature_columns,
                "numeric": numeric,
                "categorical": categorical,
                "active_model": "regresion_logistica",
                "targets": TARGETS,
                "active_features": active_features,
                "active_numeric": active_numeric,
                "active_categorical": categorical,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_logistic_importance(
        out_dir / "logistic_feature_importance.csv",
        fitted_models["gana_jugador_1"],
        active_numeric,
        categorical,
    )
    write_csv(out_dir / "test_predictions.csv", add_predictions(test_rows, feature_columns, fitted_models))
    write_csv(out_dir / "madrid_predictions.csv", add_predictions(madrid_rows, feature_columns, fitted_models))


if __name__ == "__main__":
    main()
