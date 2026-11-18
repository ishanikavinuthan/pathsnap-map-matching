from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "results" / "features" / "training_features_labeled.csv"

FEATURE_COLS = [
    "distance_m",
    "heading_difference_deg",
    "gps_speed_mps",
    "time_gap_s",
    "gps_displacement_m",
    "candidate_rank",
]
TARGET_COL = "label"


def main():
    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, dtype={"trip_id": str})

    print("\n--- 1. Shape & Columns ---")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
    print("Columns:", list(df.columns))

    print("\n--- 2. Feature & Target Verification ---")
    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    print("All 6 feature columns and target column verified!")

    print("\n--- 3. Trip Count Check ---")
    n_trips = df["trip_id"].nunique()
    print(f"Unique trip_id count: {n_trips}")

    print("\n--- 4. Missing Values (Features & Target) ---")
    print(df[FEATURE_COLS + [TARGET_COL]].isna().sum())

    print("\n--- 5. Class Distribution (y = label) ---")
    counts = df[TARGET_COL].value_counts().sort_index()
    props = df[TARGET_COL].value_counts(normalize=True).sort_index() * 100
    for cls in counts.index:
        print(f"Class {cls}: {counts[cls]:,} ({props[cls]:.2f}%)")

    print("\n--- 6. First 5 Rows (Features + Target) ---")
    print(df[["trip_id", "observation_id"] + FEATURE_COLS + [TARGET_COL]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()