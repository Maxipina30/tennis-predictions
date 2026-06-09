from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib


TRAIN_PATH = Path(__file__).resolve().parents[1] / "07_train_compare_models.py"
spec = importlib.util.spec_from_file_location("train_compare_models", TRAIN_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import helpers from {TRAIN_PATH}")
train_helpers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train_helpers
spec.loader.exec_module(train_helpers)


TARGETS = {
    "gana_jugador_1": "J1 gana partido",
    "jugador_1_gana_2_0": "J1 gana 2-0",
    "jugador_2_gana_2_0": "J2 gana 2-0",
    "jugador_1_gana_al_menos_un_set": "J1 gana al menos un set",
    "jugador_2_gana_al_menos_un_set": "J2 gana al menos un set",
    "mas_19_5_games": "Over 19.5 games",
}


def target_value(row: dict, target: str) -> int | None:
    return train_helpers.target_value(row, target)


def target_rows(rows: list[dict], target: str) -> list[dict]:
    return [row for row in rows if target_value(row, target) is not None]


def build_y(rows: list[dict], target: str):
    import numpy as np

    return np.array([int(target_value(row, target)) for row in rows])


def add_predictions(rows: list[dict], feature_columns: list[str], models: dict) -> list[dict]:
    return train_helpers.add_predictions(rows, feature_columns, models)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train non-Grand-Slam ATP BO3 logistic models.")
    parser.add_argument("--train", default="files/processed/non_grand_slam_bo3/model_dataset/train.csv")
    parser.add_argument("--test", default="files/processed/non_grand_slam_bo3/model_dataset/test.csv")
    parser.add_argument("--upcoming", default=None)
    parser.add_argument("--prediction-output", default="upcoming_predictions.csv")
    parser.add_argument("--out-dir", default="files/processed/non_grand_slam_bo3/model_training")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_train_rows = train_helpers.read_csv(Path(args.train))
    all_test_rows = train_helpers.read_csv(Path(args.test))
    train_rows = target_rows(all_train_rows, "gana_jugador_1")
    test_rows = target_rows(all_test_rows, "gana_jugador_1")
    feature_columns, numeric, categorical = train_helpers.infer_feature_columns(train_rows)
    active_numeric = train_helpers.logistic_columns(numeric)
    active_features = active_numeric + categorical
    metrics = []
    fitted_models = {}

    for target, description in TARGETS.items():
        current_train_rows = target_rows(all_train_rows, target)
        current_test_rows = target_rows(all_test_rows, target)
        model = train_helpers.build_logistic_model(active_numeric, categorical)
        x_train = train_helpers.build_x(current_train_rows, feature_columns)
        x_test = train_helpers.build_x(current_test_rows, feature_columns)
        y_train = build_y(current_train_rows, target)
        y_test = build_y(current_test_rows, target)
        model.fit(x_train, y_train)
        fitted_models[target] = model
        train_proba = train_helpers.apply_probability_adjustments(current_train_rows, target, model.predict_proba(x_train)[:, 1])
        test_proba = train_helpers.apply_probability_adjustments(current_test_rows, target, model.predict_proba(x_test)[:, 1])
        train_metrics = train_helpers.evaluate(y_train, train_proba)
        test_metrics = train_helpers.evaluate(y_test, test_proba)
        metrics.append(
            {
                "model": "regresion_logistica_bo3",
                "target": target,
                "descripcion": description,
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
        )
        joblib.dump(model, out_dir / f"regresion_logistica_{target}.joblib")
        train_helpers.write_logistic_importance(
            out_dir / f"logistic_feature_importance_{target}.csv",
            model,
            active_numeric,
            categorical,
        )

    train_helpers.write_csv(out_dir / "metrics.csv", metrics)
    (out_dir / "feature_columns.json").write_text(
        json.dumps(
            {
                "features": feature_columns,
                "numeric": numeric,
                "categorical": categorical,
                "active_model": "regresion_logistica_bo3",
                "targets": TARGETS,
                "active_features": active_features,
                "active_numeric": active_numeric,
                "active_categorical": categorical,
                "probability_adjustments": {
                    "gana_jugador_1": {
                        "name": "category_experience_logit",
                        "feature": "diferencia_categoria_torneo_log_partidos_previos",
                        "weight": train_helpers.CATEGORY_EXPERIENCE_LOGIT_WEIGHT,
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    train_helpers.write_csv(out_dir / "test_predictions.csv", add_predictions(test_rows, feature_columns, fitted_models))
    if args.upcoming:
        upcoming_rows = train_helpers.read_csv(Path(args.upcoming))
        train_helpers.write_csv(out_dir / args.prediction_output, add_predictions(upcoming_rows, feature_columns, fitted_models))


if __name__ == "__main__":
    main()
