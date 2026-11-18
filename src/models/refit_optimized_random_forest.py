"""Refit the saved Optuna best Random Forest parameters without rerunning Optuna.

Reads only train_trips from the persisted split and writes the model artifact
expected by the HMM/API. Validation and student-test trips are never loaded.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "results/features/training_features_labeled.csv"
SPLIT_PATH = ROOT / "results/models/trip_split.json"
PARAMS_PATH = ROOT / "results/models/bayesian_optimization/best_random_forest_params.json"
OUTPUT_PATH = ROOT / "results/models/bayesian_optimization/optimized_random_forest.joblib"
FEATURES = [
    "distance_m",
    "heading_difference_deg",
    "gps_speed_mps",
    "time_gap_s",
    "gps_displacement_m",
    "candidate_rank",
]
RANDOM_STATE = 42


def main() -> None:
    split = json.loads(SPLIT_PATH.read_text())
    train_trip_ids = set(map(str, split["train_trips"]))
    val_trip_ids = set(map(str, split.get("val_trips", [])))
    if train_trip_ids & val_trip_ids:
        raise ValueError("Persisted training and validation trips overlap")

    params = json.loads(PARAMS_PATH.read_text())["best_parameters"]
    columns = ["trip_id", "label", *FEATURES]
    data = pd.read_csv(DATA_PATH, dtype={"trip_id": str}, usecols=columns)
    train = data[data.trip_id.isin(train_trip_ids)].copy()
    if train.empty or not set(train.trip_id.unique()).issubset(train_trip_ids):
        raise ValueError("Training data did not match the persisted train_trips split")
    if set(train.trip_id.unique()) & val_trip_ids:
        raise AssertionError("Held-out validation trips entered the refit data")

    train[FEATURES] = train[FEATURES].replace([np.inf, -np.inf], np.nan).astype("float32")
    train["label"] = train["label"].astype("int8")
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    started = time.perf_counter()
    pipeline.fit(train[FEATURES], train["label"])
    elapsed = time.perf_counter() - started
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, OUTPUT_PATH)
    print(json.dumps({
        "saved_model": str(OUTPUT_PATH),
        "training_rows": len(train),
        "training_trips": train["trip_id"].nunique(),
        "validation_trips_used": 0,
        "student_test_trips_used": 0,
        "features": FEATURES,
        "parameters": params,
        "fit_seconds": elapsed,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
