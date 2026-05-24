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
    "jugador_1_gana_3_0": "J1 gana 3-0",
    "jugador_2_gana_3_0": "J2 gana 3-0",
    "jugador_1_minus_1_5_sets": "J1 gana -1.5 sets",
    "jugador_2_minus_1_5_sets": "J2 gana -1.5 sets",
    "jugador_1_gana_al_menos_2_sets": "J1 gana al menos 2 sets",
    "jugador_2_gana_al_menos_2_sets": "J2 gana al menos 2 sets",
    "over_3_5_sets": "Over 3.5 sets",
}


def target_value(row: dict, target: str) -> int | None:
    sets1 = train_helpers.to_int(row.get("sets_jugador_1"))
    sets2 = train_helpers.to_int(row.get("sets_jugador_2"))
    if target == "gana_jugador_1":
        value = row.get("target_gana_jugador_1")
        return int(value) if value in {"0", "1", 0, 1} else None
    if sets1 is None or sets2 is None:
        return None
    if target == "jugador_1_gana_3_0":
        return int(sets1 == 3 and sets2 == 0)
    if target == "jugador_2_gana_3_0":
        return int(sets2 == 3 and sets1 == 0)
    if target == "jugador_1_minus_1_5_sets":
        return int(sets1 == 3 and sets2 <= 1)
    if target == "jugador_2_minus_1_5_sets":
        return int(sets2 == 3 and sets1 <= 1)
    if target == "jugador_1_gana_al_menos_2_sets":
        return int(sets1 >= 2)
    if target == "jugador_2_gana_al_menos_2_sets":
        return int(sets2 >= 2)
    if target == "over_3_5_sets":
        return int(sets1 >= 1 and sets2 >= 1)
    raise ValueError(f"Unknown target: {target}")


def target_rows(rows: list[dict], target: str) -> list[dict]:
    return [row for row in rows if target_value(row, target) is not None]


def build_y(rows: list[dict], target: str):
    import numpy as np

    return np.array([int(target_value(row, target)) for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Grand Slam BO5 moneyline model.")
    parser.add_argument("--train", default="files/processed/grand_slam_moneyline/model_dataset/train.csv")
    parser.add_argument("--test", default="files/processed/grand_slam_moneyline/model_dataset/test_australian_open_2026.csv")
    parser.add_argument("--out-dir", default="files/processed/grand_slam_moneyline/model_training")
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
        model = train_helpers.build_logistic_model(active_numeric, categorical)
        current_train_rows = target_rows(all_train_rows, target)
        current_test_rows = target_rows(all_test_rows, target)
        x_train = train_helpers.build_x(current_train_rows, feature_columns)
        x_test = train_helpers.build_x(current_test_rows, feature_columns)
        y_train = build_y(current_train_rows, target)
        y_test = build_y(current_test_rows, target)
        model.fit(x_train, y_train)
        fitted_models[target] = model
        train_proba = model.predict_proba(x_train)[:, 1]
        test_proba = model.predict_proba(x_test)[:, 1]
        if target == "gana_jugador_1":
            train_proba = train_helpers.apply_probability_adjustments(current_train_rows, target, train_proba)
            test_proba = train_helpers.apply_probability_adjustments(current_test_rows, target, test_proba)
        train_metrics = train_helpers.evaluate(y_train, train_proba)
        test_metrics = train_helpers.evaluate(y_test, test_proba)
        metrics.append(
            {
                "model": "regresion_logistica_grand_slam",
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
                "active_model": "regresion_logistica_grand_slam",
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
    train_helpers.write_csv(
        out_dir / "test_predictions.csv",
        add_predictions(test_rows, feature_columns, fitted_models),
    )


def add_predictions(rows: list[dict], feature_columns: list[str], models: dict) -> list[dict]:
    x_rows = train_helpers.build_x(rows, feature_columns)
    output = [dict(row) for row in rows]
    for target, model in models.items():
        proba = model.predict_proba(x_rows)[:, 1]
        if target == "gana_jugador_1":
            proba = train_helpers.apply_probability_adjustments(rows, target, proba)
        for row, value in zip(output, proba):
            row[f"prob_{target}"] = float(value)
    for row in output:
        if "prob_jugador_1_gana_3_0" in row:
            row["prob_jugador_2_gana_al_menos_1_set"] = 1 - float(row["prob_jugador_1_gana_3_0"])
        if "prob_jugador_2_gana_3_0" in row:
            row["prob_jugador_1_gana_al_menos_1_set"] = 1 - float(row["prob_jugador_2_gana_3_0"])
        if "prob_jugador_1_minus_1_5_sets" in row:
            row["prob_jugador_2_plus_1_5_sets"] = 1 - float(row["prob_jugador_1_minus_1_5_sets"])
        if "prob_jugador_2_minus_1_5_sets" in row:
            row["prob_jugador_1_plus_1_5_sets"] = 1 - float(row["prob_jugador_2_minus_1_5_sets"])
    return output


if __name__ == "__main__":
    main()
