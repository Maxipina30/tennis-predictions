from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib


TRAIN_PATH = Path(__file__).with_name("02_train_moneyline.py")
spec = importlib.util.spec_from_file_location("grand_slam_train", TRAIN_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import helpers from {TRAIN_PATH}")
train_helpers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train_helpers
spec.loader.exec_module(train_helpers)


EXCLUDED_TARGETS = {"over_3_5_sets"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grand Slam BO5 predictions using trained models.")
    parser.add_argument("--upcoming", required=True)
    parser.add_argument("--model-dir", default="files/processed/grand_slam_moneyline/model_training")
    parser.add_argument("--prediction-output", required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    rows = train_helpers.train_helpers.read_csv(Path(args.upcoming))
    feature_config = json.loads((model_dir / "feature_columns.json").read_text(encoding="utf-8"))
    feature_columns = feature_config["features"]
    targets = {
        target: label
        for target, label in feature_config["targets"].items()
        if target not in EXCLUDED_TARGETS
    }
    models = {
        target: joblib.load(model_dir / f"regresion_logistica_{target}.joblib")
        for target in targets
    }
    predictions = train_helpers.add_predictions(rows, feature_columns, models)
    train_helpers.train_helpers.write_csv(model_dir / args.prediction_output, predictions)


if __name__ == "__main__":
    main()
