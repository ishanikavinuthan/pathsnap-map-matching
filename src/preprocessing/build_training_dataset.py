
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src" / "preprocessing"))
sys.path.insert(0, str(ROOT / "src" / "features"))

from load_data import load_trip
from align_route import align_trip
from features import extract_features

TRAIN_DIR = ROOT / "data" / "train"
FEATURE_DIR = ROOT / "results" / "features"
ALIGN_DIR = ROOT / "results" / "alignment"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)
ALIGN_DIR.mkdir(parents=True, exist_ok=True)

K = 10

FEATURE_COLUMNS = [
    "candidate_rank",
    "distance_m",
    "heading_difference_deg",
    "gps_speed_mps",
    "time_gap_s",
    "gps_displacement_m",
]

trip_dirs = sorted(
    path for path in TRAIN_DIR.iterdir()
    if path.is_dir()
    and (path / f"{path.name}.track").exists()
)

if len(trip_dirs) != 80:
    raise ValueError(
        f"Expected 80 training trips, found {len(trip_dirs)}"
    )

print("Training trips:", len(trip_dirs), flush=True)

summary = []
start_time = time.time()

for number, path in enumerate(trip_dirs, start=1):
    trip_id = path.name

    print(
        f"\n[{number}/80] Processing {trip_id}",
        flush=True
    )

    output_file = FEATURE_DIR / f"{trip_id}_labeled.csv"
    alignment_file = ALIGN_DIR / f"{trip_id}_alignment.csv"

    # Resume without repeating completed trips.
    if output_file.exists():
        print("Already processed; skipping", flush=True)
        continue

    trip = load_trip(path)

    # Keep the entire GPS sequence.
    alignment = align_trip(trip)

    alignment.to_csv(alignment_file, index=False)

    features = extract_features(trip, k=K)

    # Join by both trip and observation.
    merged = features.merge(
        alignment[
            [
                "trip_id",
                "observation_id",
                "aligned_arc_id",
                "route_position",
                "alignment_distance_m",
            ]
        ],
        on=["trip_id", "observation_id"],
        how="left",
        validate="many_to_one",
    )

    if merged["aligned_arc_id"].isna().any():
        raise ValueError(
            f"Missing alignment for trip {trip_id}"
        )

    merged["label"] = (
        merged["arc_id"] == merged["aligned_arc_id"]
    ).astype(int)

    # Each observation should have exactly one positive
    # candidate if its aligned arc is present.
    positives = merged.groupby(
        "observation_id"
    )["label"].sum()

    covered = positives.eq(1)

    if not positives.isin([0, 1]).all():
        raise ValueError(
            f"Duplicate positive candidates: {trip_id}"
        )

    # Retain all candidate rows and mark whether the
    # provisional aligned arc was in the candidate set.
    merged["label_available"] = merged[
        "observation_id"
    ].map(covered).astype(int)

    if len(merged) != len(trip["track"]) * K:
        raise ValueError(
            f"Unexpected candidate count: {trip_id}"
        )

    # Preserve missing motion features as NaN.
    # Model pipelines can impute them using training data.
    merged.to_csv(output_file, index=False)

    summary.append({
        "trip_id": trip_id,
        "gps_observations": len(trip["track"]),
        "candidate_rows": len(merged),
        "covered_observations": int(covered.sum()),
        "missing_aligned_candidate": int((~covered).sum()),
        "coverage": float(covered.mean()),
        "median_alignment_distance_m": float(
            alignment["alignment_distance_m"].median()
        ),
        "p95_alignment_distance_m": float(
            alignment["alignment_distance_m"].quantile(.95)
        ),
        "max_route_advance": int(
            alignment["route_position"].diff()
            .fillna(0).max()
        ),
    })

    print(
        f"GPS: {len(trip['track'])} | "
        f"Rows: {len(merged)} | "
        f"Coverage: {covered.mean():.2%}",
        flush=True,
    )

# Combine all saved trips, including previous runs.
all_frames = []

for path in trip_dirs:
    file = FEATURE_DIR / f"{path.name}_labeled.csv"

    if not file.exists():
        raise FileNotFoundError(file)

    frame = pd.read_csv(file, dtype={"trip_id": str})
    all_frames.append(frame)

full_data = pd.concat(all_frames, ignore_index=True)

# Exclude observations without an aligned candidate
# from supervised classification, but retain them
# in the per-trip files for analysis.
training_data = full_data[
    full_data["label_available"] == 1
].copy()

if training_data["trip_id"].nunique() != 80:
    raise ValueError("Some training trips are missing")

output = FEATURE_DIR / "training_features_labeled.csv"
training_data.to_csv(output, index=False)

# Build an audit using every saved per-trip file.
audit = []

for trip_id, frame in full_data.groupby("trip_id"):
    by_obs = frame.groupby("observation_id")

    coverage = by_obs["label_available"].first()

    audit.append({
        "trip_id": trip_id,
        "gps_observations": len(coverage),
        "candidate_rows": len(frame),
        "covered_observations": int(coverage.sum()),
        "missing_aligned_candidate": int(
            (coverage == 0).sum()
        ),
        "coverage": float(coverage.mean()),
        "median_alignment_distance_m": float(
            frame["alignment_distance_m"].median()
        ),
    })

audit_df = pd.DataFrame(audit)
audit_df.to_csv(
    FEATURE_DIR / "training_quality_summary.csv",
    index=False,
)

print("\n===== COMPLETE =====")
print("Trips:", training_data["trip_id"].nunique())
print("Candidate rows:", len(training_data))
print("Positive labels:", int(training_data["label"].sum()))
print("Negative labels:", int((training_data["label"] == 0).sum()))
print(
    "Overall candidate coverage:",
    f"{audit_df['covered_observations'].sum() / audit_df['gps_observations'].sum():.2%}"
)
print("Training CSV:", output)
print(
    "Quality report:",
    FEATURE_DIR / "training_quality_summary.csv"
)
print("Elapsed seconds:", round(time.time() - start_time, 1))
