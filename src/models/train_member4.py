
import os
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)

DATA = "results/features/training_features_labeled.csv"
OUT = "results/models"
os.makedirs(OUT, exist_ok=True)

FEATURES = [
    "distance_m",
    "heading_difference_deg",
    "gps_speed_mps",
    "time_gap_s",
    "gps_displacement_m",
    "candidate_rank"
]

print("Loading dataset...", flush=True)

df = pd.read_csv(
    DATA,
    dtype={"trip_id": str},
    usecols=["trip_id", "observation_id", "label"] + FEATURES
)

df[FEATURES] = (
    df[FEATURES]
    .replace([np.inf, -np.inf], np.nan)
    .astype("float32")
)

# Reuse Member 3's exact split
with open(f"{OUT}/trip_split.json") as f:
    split = json.load(f)

train = df[df["trip_id"].isin(split["train_trips"])]
val = df[df["trip_id"].isin(split["val_trips"])]

assert len(train) > 0 and len(val) > 0
assert not set(train.trip_id) & set(val.trip_id)

X_train, y_train = train[FEATURES], train["label"]
X_val, y_val = val[FEATURES], val["label"]

print("Training rows:", len(train), flush=True)
print("Validation rows:", len(val), flush=True)

models = {
    "random_forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=80,
            max_depth=15,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ))
    ]),

    "gradient_boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=31,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=42
        ))
    ])
}

results = []

for name, pipeline in models.items():
    print(f"\nTraining {name}...", flush=True)
    start = time.time()

    pipeline.fit(X_train, y_train)

    predictions = []
    probabilities = []

    for i in range(0, len(X_val), 20000):
        batch = X_val.iloc[i:i + 20000]
        predictions.extend(pipeline.predict(batch))
        probabilities.extend(pipeline.predict_proba(batch)[:, 1])

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)

    scored = val[["trip_id", "observation_id", "label"]].copy()
    scored["score"] = probabilities

    best = scored.groupby(
        ["trip_id", "observation_id"]
    )["score"].idxmax()

    result = {
        "model": name,
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_val, predictions, zero_division=0
        ),
        "macro_f1": f1_score(
            y_val, predictions, average="macro",
            zero_division=0
        ),
        "top1_accuracy": scored.loc[best, "label"].mean(),
        "time_seconds": time.time() - start
    }

    results.append(result)
    print(result, flush=True)

    joblib.dump(pipeline, f"{OUT}/{name}.joblib")

    pd.DataFrame(results).to_csv(
        f"{OUT}/member4_metrics.csv",
        index=False
    )

print("\nBOTH MODELS COMPLETE!", flush=True)
print(pd.DataFrame(results).to_string(index=False))

