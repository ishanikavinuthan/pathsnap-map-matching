import os
import sys
import time
import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, log_loss, confusion_matrix, ConfusionMatrixDisplay
)

# =====================================================================
# 1. IMPORT MEMBER 1'S OFFICIAL PIPELINE MODULES
# =====================================================================
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))
sys.path.insert(0, str(ROOT / "src" / "features"))

from load_data import load_trip
from align_route import align_trip
from features import extract_features

RANDOM_SEED = 42
K = 10  # Matches K=10 in Member 1's build_training_dataset.py

TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "student_test"
FEATURE_DIR = ROOT / "results" / "features"
OUTPUT_DIR = ROOT / "results" / "models_member3"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "candidate_rank",
    "distance_m",
    "heading_difference_deg",
    "gps_speed_mps",
    "time_gap_s",
    "gps_displacement_m",
]
TARGET_COL = "label"


# =====================================================================
# 2. BUILD / LOAD STUDENT-TEST DATASET USING MEMBER 1'S EXACT LOGIC
# =====================================================================
def build_student_test_dataset() -> pd.DataFrame:
    """
    Applies Member 1's exact K=10 feature extraction + route alignment logic
    to the 10 student_test trips and saves results/features/student_test_features_labeled.csv.
    """
    out_csv = FEATURE_DIR / "student_test_features_labeled.csv"
    if out_csv.exists():
        print(f"Loading existing student test dataset: {out_csv}")
        return pd.read_csv(out_csv, dtype={"trip_id": str})

    trip_dirs = sorted(
        p for p in TEST_DIR.iterdir()
        if p.is_dir() and (p / f"{p.name}.track").exists()
    )
    print(f"Building student_test dataset from {len(trip_dirs)} trips (K={K})...")

    all_frames = []
    for idx, path in enumerate(trip_dirs, start=1):
        trip_id = path.name
        trip_cache = FEATURE_DIR / f"test_{trip_id}_labeled.csv"
        if trip_cache.exists():
            all_frames.append(pd.read_csv(trip_cache, dtype={"trip_id": str}))
            continue

        trip = load_trip(path)
        alignment = align_trip(trip)
        features = extract_features(trip, k=K)

        merged = features.merge(
            alignment[["trip_id", "observation_id", "aligned_arc_id", "route_position", "alignment_distance_m"]],
            on=["trip_id", "observation_id"],
            how="left",
            validate="many_to_one",
        )
        merged["label"] = (merged["arc_id"] == merged["aligned_arc_id"]).astype(int)
        covered = merged.groupby("observation_id")["label"].sum().eq(1)
        merged["label_available"] = merged["observation_id"].map(covered).astype(int)
        merged.to_csv(trip_cache, index=False)
        all_frames.append(merged)
        print(f"  [{idx}/{len(trip_dirs)}] {trip_id} | Coverage: {covered.mean():.2%}")

    full_test = pd.concat(all_frames, ignore_index=True)
    full_test.to_csv(out_csv, index=False)
    print(f"Saved student test CSV: {out_csv}")
    return full_test


def attach_arc_nodes(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Attaches from_node_id and to_node_id from each trip's .arcs file for Route Continuity."""
    if "from_node_id" in df.columns and "to_node_id" in df.columns:
        return df

    df = df.copy()
    from_nodes = np.empty(len(df), dtype=int)
    to_nodes = np.empty(len(df), dtype=int)

    for trip_id, idx in df.groupby("trip_id").groups.items():
        arcs_path = data_dir / str(trip_id) / f"{trip_id}.arcs"
        arcs = np.loadtxt(arcs_path, dtype=int)
        if arcs.ndim == 1:
            arcs = arcs.reshape(1, -1)
        arc_ids = df.loc[idx, "arc_id"].to_numpy()
        from_nodes[idx] = arcs[arc_ids, 0]
        to_nodes[idx] = arcs[arc_ids, 1]

    df["from_node_id"] = from_nodes
    df["to_node_id"] = to_nodes
    return df


# =====================================================================
# 3. PATHSNAP METRICS (CANDIDATE + OBSERVATION-LEVEL + CONTINUITY)
# =====================================================================
def evaluate_predictions(df_meta: pd.DataFrame, y_true: np.ndarray, y_prob: np.ndarray, y_pred_bin: np.ndarray):
    macro_f1 = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred_bin, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred_bin, average="macro", zero_division=0)
    ll = log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15), labels=[0, 1])

    eval_df = df_meta.copy()
    eval_df["y_true"] = y_true
    eval_df["y_prob"] = y_prob

    # Observation-level Top-1 candidate per (trip_id, observation_id)
    top1_df = (
        eval_df.sort_values(["trip_id", "observation_id", "y_prob"], ascending=[True, True, False])
        .groupby(["trip_id", "observation_id"], as_index=False)
        .first()
    )

    top1_acc = float((top1_df["arc_id"] == top1_df["aligned_arc_id"]).mean())

    # Route Continuity across consecutive observations per trip
    continuity_scores = []
    for _, group in top1_df.groupby("trip_id"):
        group = group.sort_values("observation_id")
        arcs = group["arc_id"].to_numpy()
        from_nodes = group["from_node_id"].to_numpy()
        to_nodes = group["to_node_id"].to_numpy()

        if len(arcs) <= 1:
            continue

        same_arc = arcs[:-1] == arcs[1:]
        connected = (
            (to_nodes[:-1] == from_nodes[1:])
            | (to_nodes[:-1] == to_nodes[1:])
            | (from_nodes[:-1] == from_nodes[1:])
        )
        continuity_scores.append(np.mean(same_arc | connected))

    route_continuity = float(np.mean(continuity_scores)) if continuity_scores else 0.0

    return {
        "Macro_F1": round(macro_f1, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "Top1_Accuracy": round(top1_acc, 4),
        "Route_Continuity": round(route_continuity, 4),
        "Log_Loss": round(ll, 4),
    }


# =====================================================================
# 4. MEMBER 3 MODELS (3 DISTINCT ALGORITHM FAMILIES)
# =====================================================================
def get_member3_models():
    return {
        "Logistic_Regression": {
            "family": "Linear & Probabilistic",
            "config_summary": "SimpleImputer(median) + StandardScaler + LogReg(C=1.0, class_weight='balanced', l2)",
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                )),
            ]),
        },
        "Distance_Weighted_KNN": {
            "family": "Instance-Based",
            "config_summary": "SimpleImputer(median) + StandardScaler + KNN(n_neighbors=15, weights='distance', p=2)",
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(
                    n_neighbors=15,
                    weights="distance",
                    metric="minkowski",
                    p=2,
                    n_jobs=-1,
                )),
            ]),
        },
        "Decision_Tree": {
            "family": "Tree-Based",
            "config_summary": "SimpleImputer(median) + DecisionTree(criterion='gini', max_depth=8, min_samples_leaf=25, class_weight='balanced')",
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", DecisionTreeClassifier(
                    criterion="gini",
                    max_depth=8,
                    min_samples_split=50,
                    min_samples_leaf=25,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                )),
            ]),
        },
    }


# =====================================================================
# 5. TRAINING, 5-FOLD GROUP CV, AND EVALUATION
# =====================================================================
def run_experiments(train_df: pd.DataFrame, full_test_df: pd.DataFrame):
    # Compute Candidate Miss Rate on all student_test observations before filtering
    obs_covered = full_test_df.groupby(["trip_id", "observation_id"])["label_available"].first()
    candidate_miss_rate = round(float(1.0 - obs_covered.mean()), 4)
    print(f"Student-Test Candidate Miss Rate (K={K}): {candidate_miss_rate:.2%}")

    # Filter student_test to covered observations for supervised metrics (matches training_features_labeled.csv)
    test_df = full_test_df[full_test_df["label_available"] == 1].reset_index(drop=True)

    train_df = attach_arc_nodes(train_df, TRAIN_DIR)
    test_df = attach_arc_nodes(test_df, TEST_DIR)

    meta_cols = ["trip_id", "observation_id", "arc_id", "aligned_arc_id", "from_node_id", "to_node_id"]

    X_train = train_df[FEATURE_COLS].to_numpy()
    y_train = train_df[TARGET_COL].to_numpy()
    groups_train = train_df["trip_id"].to_numpy()

    X_test = test_df[FEATURE_COLS].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()

    gkf = GroupKFold(n_splits=5)
    models = get_member3_models()
    summary_rows = []

    for model_name, info in models.items():
        print(f"\n=======================================================")
        print(f"Training & Evaluating: {model_name} ({info['family']})")
        print(f"=======================================================")
        pipeline = info["pipeline"]

        # A. 5-Fold Group-Aware Cross Validation on 80 Training Trips
        cv_f1_scores, cv_top1_scores = [], []
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train), start=1):
            pipeline.fit(X_train[tr_idx], y_train[tr_idx])
            val_prob = pipeline.predict_proba(X_train[val_idx])[:, 1]
            val_pred = pipeline.predict(X_train[val_idx])
            val_metrics = evaluate_predictions(
                train_df.iloc[val_idx][meta_cols], y_train[val_idx], val_prob, val_pred
            )
            cv_f1_scores.append(val_metrics["Macro_F1"])
            cv_top1_scores.append(val_metrics["Top1_Accuracy"])
            print(f"  Fold {fold}/5 -> Macro F1: {val_metrics['Macro_F1']} | Top-1 Acc: {val_metrics['Top1_Accuracy']}")

        # B. Full Training Fit & Timing
        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_time_s = round(time.perf_counter() - t0, 4)

        train_prob = pipeline.predict_proba(X_train)[:, 1]
        train_pred = pipeline.predict(X_train)
        train_metrics = evaluate_predictions(train_df[meta_cols], y_train, train_prob, train_pred)

        # C. Student-Test Evaluation & Timing
        t1 = time.perf_counter()
        test_prob = pipeline.predict_proba(X_test)[:, 1]
        test_pred = pipeline.predict(X_test)
        inference_time_s = round(time.perf_counter() - t1, 4)

        test_metrics = evaluate_predictions(test_df[meta_cols], y_test, test_prob, test_pred)

        # Save Model Artifact & Confusion Matrix
        joblib.dump(pipeline, OUTPUT_DIR / f"{model_name}.joblib")

        cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Match (0)", "Match (1)"])
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"Confusion Matrix: {model_name}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{model_name}_confusion_matrix.png", dpi=300)
        plt.close()

        row = {
            "Model": model_name,
            "Algorithm_Family": info["family"],
            "Key_Configuration": info["config_summary"],
            "CV_Macro_F1": round(float(np.mean(cv_f1_scores)), 4),
            "CV_Macro_F1_Std": round(float(np.std(cv_f1_scores)), 4),
            "CV_Top1_Accuracy": round(float(np.mean(cv_top1_scores)), 4),
            "Train_Macro_F1": train_metrics["Macro_F1"],
            "Train_Top1_Accuracy": train_metrics["Top1_Accuracy"],
            "Top1_Accuracy": test_metrics["Top1_Accuracy"],
            "Route_Continuity": test_metrics["Route_Continuity"],
            "Student_Test_Macro_F1": test_metrics["Macro_F1"],
            "Student_Test_Precision": test_metrics["Precision"],
            "Student_Test_Recall": test_metrics["Recall"],
            "Candidate_Miss_Rate": candidate_miss_rate,
            "Log_Loss": test_metrics["Log_Loss"],
            "Training_Time_s": train_time_s,
            "Inference_Time_s": inference_time_s,
            "TN": int(cm[0, 0]), "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]), "TP": int(cm[1, 1]),
        }
        summary_rows.append(row)
        print(json.dumps(row, indent=2))

    results_df = pd.DataFrame(summary_rows)
    csv_path = OUTPUT_DIR / "member3_model_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison table to: {csv_path}")

    plot_member3_comparisons(results_df)
    return results_df


def plot_member3_comparisons(results_df: pd.DataFrame):
    models = results_df["Model"]
    x = np.arange(len(models))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - 1.5 * width, results_df["CV_Macro_F1"], width, label="CV Macro F1")
    ax.bar(x - 0.5 * width, results_df["Student_Test_Macro_F1"], width, label="Student-Test Macro F1")
    ax.bar(x + 0.5 * width, results_df["Top1_Accuracy"], width, label="Student-Test Top-1 Accuracy")
    ax.bar(x + 1.5 * width, results_df["Route_Continuity"], width, label="Route Continuity")

    ax.set_ylabel("Score")
    ax.set_title("Member 3 Models: Performance & Route Continuity Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.08)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "member3_f1_accuracy_comparison.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(models, results_df["Log_Loss"], color=["#2b5c8f", "#388e3c", "#d95f02"], width=0.45)
    ax.set_ylabel("Log Loss (Lower is Better)")
    ax.set_title("Member 3 Models: Student-Test Log Loss Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "member3_logloss_comparison.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    train_csv = FEATURE_DIR / "training_features_labeled.csv"

    # Build Member 1's official training CSV if not already present on disk
    if not train_csv.exists():
        print("Running Member 1's build_training_dataset.py first...")
        import build_training_dataset  # noqa: F401

    train_df = pd.read_csv(train_csv, dtype={"trip_id": str})
    full_test_df = build_student_test_dataset()
    run_experiments(train_df, full_test_df)