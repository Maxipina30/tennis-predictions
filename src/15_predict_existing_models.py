from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib


TRAIN_PATH = Path(__file__).with_name("07_train_compare_models.py")
spec = importlib.util.spec_from_file_location("train_compare_models", TRAIN_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import helpers from {TRAIN_PATH}")
train_helpers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train_helpers
spec.loader.exec_module(train_helpers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions using already trained models.")
    parser.add_argument("--upcoming", required=True)
    parser.add_argument("--model-dir", default="files/processed/model_training_2025_2026")
    parser.add_argument("--prediction-output", required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    upcoming_rows = train_helpers.read_csv(Path(args.upcoming))
    feature_config = json.loads((model_dir / "feature_columns.json").read_text(encoding="utf-8"))
    feature_columns = feature_config["features"]
    targets = feature_config.get("targets") or train_helpers.TARGETS

    models = {
        target: joblib.load(model_dir / f"regresion_logistica_{target}.joblib")
        for target in targets
    }
    predictions = train_helpers.add_predictions(upcoming_rows, feature_columns, models)
    train_helpers.write_csv(model_dir / args.prediction_output, predictions)


if __name__ == "__main__":
    main()
