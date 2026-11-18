
from pathlib import Path
import numpy as np
import pandas as pd

from load_data import load_trip


def validate_trip(trip):
    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]
    route = trip["route"]

    # GPS coordinate validity
    valid_gps = (
        track["longitude"].between(-180, 180)
        & track["latitude"].between(-90, 90)
        & np.isfinite(track["longitude"])
        & np.isfinite(track["latitude"])
    )

    # Road node validity
    valid_nodes = (
        nodes["longitude"].between(-180, 180)
        & nodes["latitude"].between(-90, 90)
        & np.isfinite(nodes["longitude"])
        & np.isfinite(nodes["latitude"])
    )

    # Timestamp checks
    time_diff = track["timestamp"].diff()

    # Road segment endpoint checks
    valid_arcs = (
        arcs["from_node_id"].between(0, len(nodes) - 1)
        & arcs["to_node_id"].between(0, len(nodes) - 1)
    )

    # Reference route arc-ID checks
    valid_route = route["arc_id"].between(
        0, len(arcs) - 1
    )

    # Check consecutive reference arcs for continuity
    route_ids = route["arc_id"].dropna().astype(int)
    route_edges = arcs.iloc[route_ids.to_numpy()]

    from_nodes = route_edges["from_node_id"].to_numpy()
    to_nodes = route_edges["to_node_id"].to_numpy()

    route_discontinuities = int(
        np.sum(to_nodes[:-1] != from_nodes[1:])
    )

    return {
        "trip_id": trip["trip_id"],
        "gps_observations": len(track),
        "invalid_gps": int((~valid_gps).sum()),
        "invalid_nodes": int((~valid_nodes).sum()),
        "invalid_arcs": int((~valid_arcs).sum()),
        "invalid_route_ids": int((~valid_route).sum()),
        "duplicate_gps": int(
            track.duplicated(
                subset=["longitude", "latitude", "timestamp"]
            ).sum()
        ),
        "non_increasing_timestamps": int(
            (time_diff <= 0).sum()
        ),
        "gps_gaps_over_5s": int(
            (time_diff > 5).sum()
        ),
        "max_timestamp_gap_s": float(
            time_diff.max()
        ),
        "route_discontinuities": route_discontinuities,
    }


def main():
    records = []

    for partition in ["train", "student_test"]:
        root = Path("data") / partition

        for trip_dir in sorted(root.iterdir()):
            if not trip_dir.is_dir():
                continue

            trip = load_trip(trip_dir)
            result = validate_trip(trip)
            result["partition"] = partition
            records.append(result)

    df = pd.DataFrame(records)

    output = Path("results/eda")
    output.mkdir(parents=True, exist_ok=True)

    df.to_csv(output / "data_quality.csv", index=False)

    print("\nDATA QUALITY SUMMARY")
    print(df.groupby("partition").agg({
        "gps_observations": ["count", "sum", "mean"],
        "invalid_gps": "sum",
        "invalid_nodes": "sum",
        "invalid_arcs": "sum",
        "invalid_route_ids": "sum",
        "duplicate_gps": "sum",
        "non_increasing_timestamps": "sum",
        "gps_gaps_over_5s": "sum",
        "route_discontinuities": "sum",
    }))

    print("\nSaved: results/eda/data_quality.csv")


if __name__ == "__main__":
    main()
