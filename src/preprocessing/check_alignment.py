
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "features"))

from features import extract_features
from load_data import load_trip


def main():
    trip = load_trip("data/train/00000001")
    features = extract_features(trip, k=5)

    reference_arcs = set(
        trip["route"]["arc_id"].dropna().astype(int)
    )

    features["in_reference_route"] = (
        features["arc_id"].isin(reference_arcs)
    )

    observation_has_reference = (
        features.groupby("observation_id")[
            "in_reference_route"
        ].any()
    )

    print("Trip:", trip["trip_id"])
    print("GPS observations:", len(observation_has_reference))
    print("Reference-route arcs:", len(reference_arcs))

    print(
        "Observations with a reference-route candidate:",
        int(observation_has_reference.sum())
    )

    print(
        "Reference-route candidate coverage:",
        f"{observation_has_reference.mean():.2%}"
    )

    print("\nCandidate reference membership by rank:")
    print(
        features.groupby("candidate_rank")[
            "in_reference_route"
        ].mean()
    )


if __name__ == "__main__":
    main()
