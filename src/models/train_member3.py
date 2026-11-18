
import os
import time
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

DATA_PATH = "results/features/training_features_labeled.csv"
OUTPUT_DIR = "results/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    DATA_PATH,
    dtype={"trip_id": str},
    usecols=["trip_id", "observation_id", "label"] + FEATURES
)

df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
df[FEATURES] = df[FEATURES].astype("float32")

print("Dataset:", df.shape, flush=True)

# Split entire trips, never individual candidate rows
splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.2, random_state=42
)

train_idx, val_idx = next(
    splitter.split(df, df["label"], groups=df["trip_id"])
)

train = df.iloc[train_idx]
val = df.iloc[val_idx]

train_trips = sorted(train["trip_id"].unique().tolist())
val_trips = sorted(val["trip_id"].unique().tolist())

with open(f"{OUTPUT_DIR}/trip_split.json", "w") as f:
    json.dump(
        {"train_trips": train_trips, "val_trips": val_trips},
        f, indent=2
    )

X_train = train[FEATURES]
y_train = train["label"]

X_val = val[FEATURES]
y_val = val["label"]

print("Training rows:", len(train), flush=True)
print("Validation rows:", len(val), flush=True)
print("Validation trips:", len(val_trips), flush=True)

models = {
    "logistic_regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=500,
            class_weight="balanced"
        ))
    ]),

    "knn": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=5,
            algorithm="kd_tree",
            n_jobs=-1
        ))
    ]),

    "decision_tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(
            max_depth=15,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42
        ))
    ])
}

results = []

for name, pipeline in models.items():
    print(f"\nTraining {name}...", flush=True)
    start = time.time()

    # KNN is expensive on ~1.5M training rows.
    # Use a reproducible training-only subset for KNN.
    if name == "knn":
        positive = train[train["label"] == 1]
        negative = train[train["label"] == 0]

        positive = positive.sample(
            n=min(len(positive), 50000),
            random_state=42
        )
        negative = negative.sample(
            n=min(len(negative), 50000),
            random_state=42
        )

        knn_train = pd.concat(
            [positive, negative]
        ).sample(frac=1, random_state=42)

        pipeline.fit(
            knn_train[FEATURES],
            knn_train["label"]
        )
        print("KNN training subset:", len(knn_train), flush=True)
    else:
        pipeline.fit(X_train, y_train)

    # Predict in batches to control memory usage
    predictions = []
    probabilities = []
    batch_size = 20000

    for start_idx in range(0, len(X_val), batch_size):
        batch = X_val.iloc[start_idx:start_idx + batch_size]
        predictions.extend(pipeline.predict(batch))
        probabilities.extend(pipeline.predict_proba(batch)[:, 1])

    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)

    # Select the highest-scoring candidate per GPS observation
    scored = val[
        ["trip_id", "observation_id", "label"]
    ].copy()

    scored["score"] = probabilities

    best_indices = scored.groupby(
        ["trip_id", "observation_id"]
    )["score"].idxmax()

    top1_accuracy = scored.loc[
        best_indices, "label"
    ].mean()

    elapsed = time.time() - start

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
            y_val, predictions,
            average="macro", zero_division=0
        ),
        "top1_accuracy": top1_accuracy,
        "time_seconds": elapsed
    }

    results.append(result)

    print(result, flush=True)
    print(
        "Confusion matrix:\n",
        confusion_matrix(y_val, predictions),
        flush=True
    )

    joblib.dump(
        pipeline,
        f"{OUTPUT_DIR}/{name}.joblib"
    )

    pd.DataFrame(results).to_csv(
        f"{OUTPUT_DIR}/member3_metrics.csv",
        index=False
    )

print("\nALL MODELS COMPLETE", flush=True)
print(pd.DataFrame(results).to_string(index=False))

