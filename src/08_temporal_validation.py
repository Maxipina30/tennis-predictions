from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import importlib.util
import sys


SPLIT_PATH = Path(__file__).with_name("06_split_train_test.py")
TRAIN_PATH = Path(__file__).with_name("07_train_compare_models.py")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


split_helpers = load_module(SPLIT_PATH, "split_helpers")
train_helpers = load_module(TRAIN_PATH, "train_helpers")


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
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


def train_eval(train_rows: list[dict], test_rows: list[dict], out_dir: Path) -> list[dict]:
    feature_columns, numeric, categorical = train_helpers.infer_feature_columns(train_rows)
    x_train, y_train = train_helpers.split_xy(train_rows, feature_columns)
    x_test, y_test = train_helpers.split_xy(test_rows, feature_columns)
    models = train_helpers.build_models(numeric, categorical)
    metric_rows: list[dict] = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        train_proba = model.predict_proba(x_train)[:, 1]
        test_proba = model.predict_proba(x_test)[:, 1]
        train_metrics = train_helpers.evaluate(y_train, train_proba)
        test_metrics = train_helpers.evaluate(y_test, test_proba)
        metric_rows.append(
            {
                "model": name,
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
    write_csv(out_dir / "metrics.csv", metric_rows)
    return metric_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run temporal validation splits.")
    parser.add_argument("--dataset", default="files/processed/model_dataset_2026/model_dataset.csv")
    parser.add_argument("--out-dir", default="files/processed/temporal_validation_2026")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(Path(args.dataset))

    test_tournaments = {"Monte Carlo", "Barcelona", "Munich"}
    train_rows = split_helpers.orient_completed_rows(
        [
            row
            for row in rows
            if row.get("target_gana_jugador_1") in {"0", "1"}
            and row.get("torneo") not in test_tournaments
            and row.get("fecha", "") < "2026-04-05"
        ]
    )
    test_rows = split_helpers.orient_completed_rows(
        [
            row
            for row in rows
            if row.get("target_gana_jugador_1") in {"0", "1"}
            and row.get("torneo") in test_tournaments
        ]
    )

    write_csv(out_dir / "train_until_miami.csv", train_rows)
    write_csv(out_dir / "test_montecarlo_barcelona_munich.csv", test_rows)
    metrics = train_eval(train_rows, test_rows, out_dir)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "train_rows": len(train_rows),
                "test_rows": len(test_rows),
                "test_tournaments": sorted(test_tournaments),
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
