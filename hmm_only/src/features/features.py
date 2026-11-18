
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))

from load_data import load_trip
from candidates import CandidateGenerator, project_coordinates


def heading_degrees(start, end):
    """Heading in degrees, clockwise from north."""
    delta = end - start

    return (
        np.degrees(np.arctan2(delta[0], delta[1]))
        + 360
    ) % 360


def heading_difference(a, b):
    """Smallest angular difference, between 0 and 180."""
    return abs((a - b + 180) % 360 - 180)


def extract_features(trip, k=5):
    track = trip["track"]
    generator = CandidateGenerator(
        trip["nodes"], trip["arcs"]
    )

    coords = project_coordinates(
        track["longitude"].to_numpy(),
        track["latitude"].to_numpy(),
        generator.lat0,
    )

    timestamps = track["timestamp"].to_numpy()

    records = []

    for i in range(len(track)):
        longitude = track.iloc[i]["longitude"]
        latitude = track.iloc[i]["latitude"]

        if i == 0:
            time_gap = np.nan
            displacement = np.nan
            speed = np.nan
            gps_heading = np.nan
        else:
            time_gap = timestamps[i] - timestamps[i - 1]

            displacement = float(
                np.linalg.norm(coords[i] - coords[i - 1])
            )

            speed = (
                displacement / time_gap
                if time_gap > 0 else np.nan
            )

            gps_heading = (
                heading_degrees(coords[i - 1], coords[i])
                if displacement > 0 else np.nan
            )

        candidates = generator.get_candidates(
            longitude, latitude, k=k
        )

        for rank, (arc_id, distance) in enumerate(
            candidates, start=1
        ):
            road_heading = heading_degrees(
                generator.starts[arc_id],
                generator.ends[arc_id],
            )

            difference = (
                heading_difference(gps_heading, road_heading)
                if np.isfinite(gps_heading)
                else np.nan
            )

            records.append({
                "trip_id": trip["trip_id"],
                "observation_id": i,
                "arc_id": arc_id,
                "candidate_rank": rank,
                "distance_m": distance,
                "heading_difference_deg": difference,
                "gps_speed_mps": speed,
                "time_gap_s": time_gap,
                "gps_displacement_m": displacement,
            })

    return pd.DataFrame(records)


def main():
    trip = load_trip("data/train/00000001")

    features = extract_features(trip, k=5)

    output = Path("results/features")
    output.mkdir(parents=True, exist_ok=True)

    features.to_csv(
        output / "trip_00000001_features.csv",
        index=False,
    )

    print("\nFEATURE EXTRACTION COMPLETE")
    print("Trip:", trip["trip_id"])
    print("GPS observations:", len(trip["track"]))
    print("Feature rows:", len(features))

    print("\nFirst 10 feature rows:")
    print(features.head(10).to_string(index=False))

    print("\nMissing values:")
    print(features.isna().sum())

    print(
        "\nSaved:",
        output / "trip_00000001_features.csv"
    )


if __name__ == "__main__":
    main()
