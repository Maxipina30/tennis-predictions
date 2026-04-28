from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DROP_COLUMNS = {
    "match_id",
    "fecha",
    "jugador_1",
    "jugador_2",
    "sets_jugador_1",
    "sets_jugador_2",
    "games_jugador_1",
    "games_jugador_2",
    "target_gana_jugador_1",
}
CATEGORICAL_COLUMNS = ["torneo", "categoria_torneo", "superficie", "ronda"]


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


def split_xy(rows: list[dict], feature_columns: list[str]) -> tuple[list[dict], np.ndarray]:
    x_rows = pd.DataFrame([{col: row.get(col, "") for col in feature_columns} for row in rows])
    for col in x_rows.columns:
        if col not in CATEGORICAL_COLUMNS:
            x_rows[col] = pd.to_numeric(x_rows[col], errors="coerce")
    y = np.array([int(row["target_gana_jugador_1"]) for row in rows])
    return x_rows, y


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


def build_models(numeric: list[str], categorical: list[str]) -> dict[str, Pipeline]:
    return {
        "regresion_logistica": Pipeline(
            [
                ("prep", make_preprocessor(numeric, categorical, scale_numeric=True)),
                ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("prep", make_preprocessor(numeric, categorical, scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=6,
                        min_samples_leaf=10,
                        class_weight="balanced_subsample",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("prep", make_preprocessor(numeric, categorical, scale_numeric=False)),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=150,
                        learning_rate=0.04,
                        max_depth=2,
                        min_samples_leaf=15,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float | None]:
    pred = (proba >= 0.5).astype(int)
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, proba)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def add_predictions(rows: list[dict], feature_columns: list[str], models: dict[str, Pipeline]) -> list[dict]:
    x_rows = pd.DataFrame([{col: row.get(col, "") for col in feature_columns} for row in rows])
    for col in x_rows.columns:
        if col not in CATEGORICAL_COLUMNS:
            x_rows[col] = pd.to_numeric(x_rows[col], errors="coerce")
    output = [dict(row) for row in rows]
    for name, model in models.items():
        proba = model.predict_proba(x_rows)[:, 1]
        for row, value in zip(output, proba):
            row[f"prob_gana_jugador_1_{name}"] = float(value)
            row[f"prob_gana_jugador_2_{name}"] = float(1 - value)
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
    x_train, y_train = split_xy(train_rows, feature_columns)
    x_test, y_test = split_xy(test_rows, feature_columns)

    models = build_models(numeric, categorical)
    metrics_rows: list[dict] = []
    fitted_models: dict[str, Pipeline] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        fitted_models[name] = model
        train_proba = model.predict_proba(x_train)[:, 1]
        test_proba = model.predict_proba(x_test)[:, 1]
        train_metrics = evaluate(y_train, train_proba)
        test_metrics = evaluate(y_test, test_proba)
        metrics = {
            "train_accuracy": train_metrics["accuracy"],
            "train_log_loss": train_metrics["log_loss"],
            "train_brier_score": train_metrics["brier_score"],
            "train_roc_auc": train_metrics["roc_auc"],
            "test_accuracy": test_metrics["accuracy"],
            "test_log_loss": test_metrics["log_loss"],
            "test_brier_score": test_metrics["brier_score"],
            "test_roc_auc": test_metrics["roc_auc"],
        }
        metrics_rows.append({"model": name, **metrics})
        joblib.dump(model, out_dir / f"{name}.joblib")

    write_csv(out_dir / "metrics.csv", metrics_rows)
    (out_dir / "feature_columns.json").write_text(
        json.dumps({"features": feature_columns, "numeric": numeric, "categorical": categorical}, indent=2),
        encoding="utf-8",
    )
    write_csv(out_dir / "test_predictions.csv", add_predictions(test_rows, feature_columns, fitted_models))
    write_csv(out_dir / "madrid_predictions.csv", add_predictions(madrid_rows, feature_columns, fitted_models))


if __name__ == "__main__":
    main()
