
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))

from load_data import load_trip
from candidates import CandidateGenerator, project_coordinates


def align_trip(trip, max_advance=10, jump_penalty=2.0):
    """
    Align GPS observations with ordered reference arcs.

    Produces provisional pseudo-labels, not ground truth.
    """
    track = trip["track"]
    route_ids = trip["route"]["arc_id"].to_numpy(dtype=int)

    generator = CandidateGenerator(
        trip["nodes"], trip["arcs"]
    )

    coords = project_coordinates(
        track["longitude"].to_numpy(),
        track["latitude"].to_numpy(),
        generator.lat0,
    )

    # Compute distances to reference-route segments
    starts = generator.starts[route_ids]
    ends = generator.ends[route_ids]
    vectors = ends - starts

    lengths_sq = np.sum(vectors ** 2, axis=1)

    distances = np.empty((len(coords), len(route_ids)))

    for i, point in enumerate(coords):
        offsets = point - starts

        numerators = np.sum(offsets * vectors, axis=1)

        fractions = np.divide(
            numerators,
            lengths_sq,
            out=np.zeros_like(numerators),
            where=lengths_sq > 0,
        )

        fractions = np.clip(fractions, 0, 1)

        projected = starts + fractions[:, None] * vectors

        distances[i] = np.linalg.norm(
            projected - point, axis=1
        )

    n, m = distances.shape

    dp = np.full((n, m), np.inf)
    backtrack = np.full((n, m), -1, dtype=int)

    # Allow starting at any reference-route position
    dp[0] = distances[0]

    for i in range(1, n):
        for j in range(m):
            low = max(0, j - max_advance)

            previous = np.arange(low, j + 1)

            transition_costs = (
                dp[i - 1, previous]
                + jump_penalty * (j - previous)
            )

            best_index = int(np.argmin(transition_costs))
            best_previous = int(previous[best_index])

            dp[i, j] = (
                transition_costs[best_index]
                + distances[i, j]
            )

            backtrack[i, j] = best_previous

    # Recover most likely sequence
    aligned_positions = np.empty(n, dtype=int)
    aligned_positions[-1] = int(np.argmin(dp[-1]))

    for i in range(n - 1, 0, -1):
        aligned_positions[i - 1] = (
            backtrack[i, aligned_positions[i]]
        )

    aligned_arcs = route_ids[aligned_positions]

    result = pd.DataFrame({
        "trip_id": trip["trip_id"],
        "observation_id": np.arange(n),
        "aligned_arc_id": aligned_arcs,
        "route_position": aligned_positions,
        "alignment_distance_m": distances[
            np.arange(n), aligned_positions
        ],
    })

    return result


def main():
    trip = load_trip("data/train/00000001")

    aligned = align_trip(trip)

    output = Path("results/alignment")
    output.mkdir(parents=True, exist_ok=True)

    aligned.to_csv(
        output / "trip_00000001_alignment.csv",
        index=False,
    )

    print("\nALIGNMENT COMPLETE")
    print("Trip:", trip["trip_id"])
    print("GPS observations:", len(aligned))
    print(
        "Unique aligned arcs:",
        aligned["aligned_arc_id"].nunique()
    )

    print("\nFirst 10 aligned observations:")
    print(aligned.head(10).to_string(index=False))

    print("\nAlignment distance statistics:")
    print(aligned["alignment_distance_m"].describe())

    print(
        "\nSaved:",
        output / "trip_00000001_alignment.csv"
    )


if __name__ == "__main__":
    main()
