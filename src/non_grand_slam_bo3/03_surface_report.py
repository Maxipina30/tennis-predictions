from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib


TRAIN_PATH = Path(__file__).with_name("02_train_moneyline.py")
spec = importlib.util.spec_from_file_location("bo3_train", TRAIN_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import helpers from {TRAIN_PATH}")
bo3_train = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bo3_train
spec.loader.exec_module(bo3_train)
train_helpers = bo3_train.train_helpers


def surface_metrics(rows: list[dict], feature_columns: list[str], models: dict) -> list[dict]:
    output = []
    for surface in sorted({row.get("superficie", "unknown") or "unknown" for row in rows}):
        surface_rows = [row for row in rows if (row.get("superficie", "unknown") or "unknown") == surface]
        if len(surface_rows) < 10:
            continue
        x_rows = train_helpers.build_x(surface_rows, feature_columns)
        for target, model in models.items():
            target_rows = bo3_train.target_rows(surface_rows, target)
            if len(target_rows) < 10:
                continue
            x_target = train_helpers.build_x(target_rows, feature_columns)
            y = bo3_train.build_y(target_rows, target)
            proba = model.predict_proba(x_target)[:, 1]
            proba = train_helpers.apply_probability_adjustments(target_rows, target, proba)
            metrics = train_helpers.evaluate(y, proba)
            output.append(
                {
                    "surface": surface,
                    "model_scope": "global_bo3",
                    "target": target,
                    "rows": metrics["rows"],
                    "positive_rate": metrics["positive_rate"],
                    "accuracy": metrics["accuracy"],
                    "log_loss": metrics["log_loss"],
                    "brier_score": metrics["brier_score"],
                    "roc_auc": metrics["roc_auc"],
                }
            )
    return output


def grass_holdout_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grass = sorted([row for row in rows if row.get("superficie") == "grass"], key=lambda row: row.get("fecha", ""))
    if len(grass) < 80:
        return [], []
    split_index = max(40, int(len(grass) * 0.7))
    return grass[:split_index], grass[split_index:]


def grass_specific_metrics(all_rows: list[dict]) -> list[dict]:
    train_rows, test_rows = grass_holdout_rows(all_rows)
    if not train_rows or not test_rows:
        return []
    feature_columns, numeric, categorical = train_helpers.infer_feature_columns(train_rows)
    active_numeric = train_helpers.logistic_columns(numeric)
    output = []
    for target in bo3_train.TARGETS:
        current_train_rows = bo3_train.target_rows(train_rows, target)
        current_test_rows = bo3_train.target_rows(test_rows, target)
        if len(current_train_rows) < 30 or len(current_test_rows) < 10:
            continue
        y_train = bo3_train.build_y(current_train_rows, target)
        y_test = bo3_train.build_y(current_test_rows, target)
        if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
            continue
        model = train_helpers.build_logistic_model(active_numeric, categorical)
        model.fit(
            train_helpers.build_x(current_train_rows, feature_columns),
            y_train,
        )
        proba = model.predict_proba(train_helpers.build_x(current_test_rows, feature_columns))[:, 1]
        proba = train_helpers.apply_probability_adjustments(current_test_rows, target, proba)
        metrics = train_helpers.evaluate(y_test, proba)
        output.append(
            {
                "surface": "grass",
                "model_scope": "grass_only_temporal_holdout",
                "target": target,
                "rows": metrics["rows"],
                "positive_rate": metrics["positive_rate"],
                "accuracy": metrics["accuracy"],
                "log_loss": metrics["log_loss"],
                "brier_score": metrics["brier_score"],
                "roc_auc": metrics["roc_auc"],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BO3 model performance by surface.")
    parser.add_argument("--all-matches", default="files/processed/non_grand_slam_bo3/model_dataset/bo3_matches.csv")
    parser.add_argument("--test", default="files/processed/non_grand_slam_bo3/model_dataset/test.csv")
    parser.add_argument("--model-dir", default="files/processed/non_grand_slam_bo3/model_training")
    parser.add_argument("--out-dir", default="files/processed/non_grand_slam_bo3/model_training")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    feature_config = json.loads((model_dir / "feature_columns.json").read_text(encoding="utf-8"))
    feature_columns = feature_config["features"]
    targets = feature_config.get("targets") or bo3_train.TARGETS
    models = {
        target: joblib.load(model_dir / f"regresion_logistica_{target}.joblib")
        for target in targets
    }
    test_rows = train_helpers.read_csv(Path(args.test))
    all_rows = train_helpers.read_csv(Path(args.all_matches))
    rows = surface_metrics(test_rows, feature_columns, models) + grass_specific_metrics(all_rows)
    train_helpers.write_csv(Path(args.out_dir) / "surface_metrics.csv", rows)


if __name__ == "__main__":
    main()
