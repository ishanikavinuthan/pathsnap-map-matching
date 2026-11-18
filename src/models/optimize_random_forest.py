
from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "results/features/training_features_labeled.csv"
SPLIT_PATH = ROOT / "results/models/trip_split.json"

OUTPUT_DIR = ROOT / "results/models/bayesian_optimization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "distance_m",
    "heading_difference_deg",
    "gps_speed_mps",
    "time_gap_s",
    "gps_displacement_m",
    "candidate_rank",
]

N_TRIALS = 15
RANDOM_STATE = 42

# --------------------------------------------------
# 2. LOAD DATA AND EXISTING TRIP SPLIT
# --------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH,
    dtype={"trip_id": str},
    usecols=["trip_id", "observation_id", "label"] + FEATURES,
)

df[FEATURES] = (
    df[FEATURES]
    .replace([np.inf, -np.inf], np.nan)
    .astype("float32")
)

df["label"] = df["label"].astype("int8")

with open(SPLIT_PATH, "r") as file:
    split = json.load(file)

train = df[df["trip_id"].isin(split["train_trips"])].copy()
val = df[df["trip_id"].isin(split["val_trips"])].copy()

assert len(train) > 0 and len(val) > 0
assert set(train["trip_id"]).isdisjoint(set(val["trip_id"]))

X_train = train[FEATURES]
y_train = train["label"]
groups_train = train["trip_id"]

X_val = val[FEATURES]
y_val = val["label"]

print("Training rows:", len(train))
print("Validation rows:", len(val))
print("Training trips:", train["trip_id"].nunique())
print("Validation trips:", val["trip_id"].nunique())

# --------------------------------------------------
# 3. RANDOM FOREST PIPELINE
# --------------------------------------------------

def create_pipeline(params):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            RandomForestClassifier(
                **params,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])


# --------------------------------------------------
# 4. BAYESIAN OPTIMISATION OBJECTIVE
# --------------------------------------------------

cv = GroupKFold(n_splits=3)


def objective(trial):

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", 60, 160, step=20
        ),
        "max_depth": trial.suggest_categorical(
            "max_depth", [10, 15, 20, 25]
        ),
        "min_samples_split": trial.suggest_int(
            "min_samples_split", 2, 40
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", 5, 30
        ),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", 0.7, 1.0]
        ),
        "class_weight": trial.suggest_categorical(
            "class_weight", ["balanced", "balanced_subsample"]
        ),
    }

    fold_scores = []

    for fold_number, (train_idx, val_idx) in enumerate(
        cv.split(X_train, y_train, groups_train), start=1
    ):

        model = create_pipeline(params)

        model.fit(
            X_train.iloc[train_idx],
            y_train.iloc[train_idx],
        )

        predictions = model.predict(X_train.iloc[val_idx])

        score = f1_score(
            y_train.iloc[val_idx],
            predictions,
            average="macro",
            zero_division=0,
        )

        fold_scores.append(score)

        print(
            f"Trial {trial.number + 1}, "
            f"Fold {fold_number}: Macro F1 = {score:.5f}"
        )

    mean_score = float(np.mean(fold_scores))

    print(
        f"Trial {trial.number + 1} completed. "
        f"Mean Macro F1 = {mean_score:.5f}\n"
    )

    return mean_score


# --------------------------------------------------
# 5. RUN OPTUNA TPE OPTIMISATION
# --------------------------------------------------

print("\nStarting Bayesian optimisation...")
print("Number of trials:", N_TRIALS)

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    study_name="random_forest_bayesian_optimization",
)

study.optimize(objective, n_trials=N_TRIALS)

print("\nOptimisation completed!")

print("Best cross-validation Macro F1:", study.best_value)
print("Best parameters:")
print(study.best_params)

# --------------------------------------------------
# 6. SAVE OPTIMISATION RESULTS
# --------------------------------------------------

trials_df = study.trials_dataframe()

trials_df.to_csv(
    OUTPUT_DIR / "bayesian_optimization_results.csv",
    index=False,
)

best_details = {
    "best_cv_macro_f1": float(study.best_value),
    "best_parameters": study.best_params,
    "number_of_trials": N_TRIALS,
    "cv_folds": 3,
    "objective": "maximize_macro_f1",
}

with open(OUTPUT_DIR / "best_random_forest_params.json", "w") as file:
    json.dump(best_details, file, indent=4)

# --------------------------------------------------
# 7. EVALUATION FUNCTION
# --------------------------------------------------

def evaluate_model(model, X, y, validation_frame):

    probabilities = []

    batch_size = 20000

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))

        batch_probabilities = model.predict_proba(
            X.iloc[start:end]
        )

        positive_index = list(model.classes_).index(1)

        probabilities.extend(
            batch_probabilities[:, positive_index]
        )

    probabilities = np.asarray(probabilities)

    predictions = (probabilities >= 0.5).astype(int)

    scored = validation_frame[
        ["trip_id", "observation_id", "label"]
    ].copy()

    scored["score"] = probabilities

    best_candidates = scored.groupby(
        ["trip_id", "observation_id"]
    )["score"].idxmax()

    top1_accuracy = float(
        scored.loc[best_candidates, "label"].mean()
    )

    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1": f1_score(
            y, predictions, zero_division=0
        ),
        "macro_f1": f1_score(
            y, predictions,
            average="macro",
            zero_division=0,
        ),
        "top1_accuracy": top1_accuracy,
    }


# --------------------------------------------------
# 8. BASELINE RANDOM FOREST
# --------------------------------------------------

baseline_params = {
    "n_estimators": 80,
    "max_depth": 15,
    "min_samples_leaf": 20,
    "class_weight": "balanced",
}

print("\nTraining baseline Random Forest...")

baseline_model = create_pipeline(baseline_params)

start = time.time()
baseline_model.fit(X_train, y_train)
baseline_time = time.time() - start

baseline_metrics = evaluate_model(
    baseline_model, X_val, y_val, val
)

baseline_metrics["model"] = "Baseline Random Forest"
baseline_metrics["time_seconds"] = baseline_time

print("\nBaseline validation metrics:")
print(baseline_metrics)

# --------------------------------------------------
# 9. TRAIN OPTIMISED RANDOM FOREST
# --------------------------------------------------

print("\nTraining optimised Random Forest...")

optimized_model = create_pipeline(study.best_params)

start = time.time()
optimized_model.fit(X_train, y_train)
optimized_time = time.time() - start

optimized_metrics = evaluate_model(
    optimized_model, X_val, y_val, val
)

optimized_metrics["model"] = "Bayesian Optimised Random Forest"
optimized_metrics["time_seconds"] = optimized_time

print("\nOptimised validation metrics:")
print(optimized_metrics)

# --------------------------------------------------
# 10. COMPARE BASELINE AND OPTIMISED MODEL
# --------------------------------------------------

comparison = pd.DataFrame([
    baseline_metrics,
    optimized_metrics,
])

comparison = comparison[
    [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "top1_accuracy",
        "time_seconds",
    ]
]

comparison.to_csv(
    OUTPUT_DIR / "random_forest_optimization_comparison.csv",
    index=False,
)

print("\nFINAL MODEL COMPARISON")
print(comparison.to_string(index=False))

# --------------------------------------------------
# 11. SAVE OPTIMISED MODEL
# --------------------------------------------------

joblib.dump(
    optimized_model,
    OUTPUT_DIR / "optimized_random_forest.joblib",
)

# --------------------------------------------------
# 12. GENERATE OPTIMISATION PLOTS
# --------------------------------------------------

try:
    import matplotlib.pyplot as plt
    from optuna.visualization.matplotlib import (
        plot_optimization_history,
        plot_param_importances,
    )

    ax = plot_optimization_history(study)
    ax.figure.tight_layout()
    ax.figure.savefig(
        OUTPUT_DIR / "optimization_history.png",
        dpi=200,
    )
    plt.close(ax.figure)

    ax = plot_param_importances(study)
    ax.figure.tight_layout()
    ax.figure.savefig(
        OUTPUT_DIR / "parameter_importances.png",
        dpi=200,
    )
    plt.close(ax.figure)

    print("\nOptimisation plots saved.")

except Exception as error:
    print("\nCould not generate plots:", error)

print("\nAll outputs saved in:")
print(OUTPUT_DIR)

print("\nBAYESIAN OPTIMISATION FINISHED!")
