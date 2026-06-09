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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate non-Grand-Slam BO3 predictions using trained models.")
    parser.add_argument("--upcoming", required=True)
    parser.add_argument("--model-dir", default="files/processed/non_grand_slam_bo3/model_training")
    parser.add_argument("--prediction-output", required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    rows = bo3_train.train_helpers.read_csv(Path(args.upcoming))
    feature_config = json.loads((model_dir / "feature_columns.json").read_text(encoding="utf-8"))
    feature_columns = feature_config["features"]
    targets = feature_config.get("targets") or bo3_train.TARGETS
    models = {
        target: joblib.load(model_dir / f"regresion_logistica_{target}.joblib")
        for target in targets
    }
    predictions = bo3_train.add_predictions(rows, feature_columns, models)
    bo3_train.train_helpers.write_csv(model_dir / args.prediction_output, predictions)


if __name__ == "__main__":
    main()
