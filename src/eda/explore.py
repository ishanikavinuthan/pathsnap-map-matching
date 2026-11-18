"""
explore.py
Member 2 - EDA
PathSnap: Dataset Overview and Statistics
Generates trip_statistics.csv with counts, file sizes, bounding boxes.
"""

import os
import pandas as pd
import numpy as np

DATA_ROOT = r"C:\Users\ayush\OneDrive\Desktop\ml assignment\ml hackathon\Student folder\Student folder"
PARTITIONS = {
    "train": os.path.join(DATA_ROOT, "train"),
    "student_test": os.path.join(DATA_ROOT, "student_test"),
}
OUTPUT_DIR = "results/eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_file(trip_path, extension):
    for f in os.listdir(trip_path):
        if f.endswith(extension):
            return os.path.join(trip_path, f)
    return None


def analyze_trip(trip_id, trip_path, partition):
    row = {"trip_id": trip_id, "partition": partition}

    track_file = find_file(trip_path, ".track")
    row["track_file_exists"] = track_file is not None
    if track_file:
        row["track_size_bytes"] = os.path.getsize(track_file)
        try:
            df = pd.read_csv(track_file, sep=r"\s+", header=None,
                             names=["longitude", "latitude", "timestamp"])
            row["gps_obs_count"] = len(df)
            row["missing_gps"] = df[["longitude", "latitude"]].isnull().sum().sum()
            row["lon_min"] = df["longitude"].min()
            row["lon_max"] = df["longitude"].max()
            row["lat_min"] = df["latitude"].min()
            row["lat_max"] = df["latitude"].max()
            row["timestamp_min"] = df["timestamp"].min()
            row["timestamp_max"] = df["timestamp"].max()
        except Exception as e:
            row["gps_obs_count"] = 0
            row["track_error"] = str(e)
    else:
        row["track_size_bytes"] = 0
        row["gps_obs_count"] = 0

    nodes_file = find_file(trip_path, ".nodes")
    row["nodes_file_exists"] = nodes_file is not None
    if nodes_file:
        row["nodes_size_bytes"] = os.path.getsize(nodes_file)
        try:
            df_nodes = pd.read_csv(nodes_file, sep=r"\s+", header=None,
                                   names=["longitude", "latitude"])
            row["node_count"] = len(df_nodes)
        except Exception as e:
            row["node_count"] = 0
    else:
        row["nodes_size_bytes"] = 0
        row["node_count"] = 0

    arcs_file = find_file(trip_path, ".arcs")
    row["arcs_file_exists"] = arcs_file is not None
    if arcs_file:
        row["arcs_size_bytes"] = os.path.getsize(arcs_file)
        try:
            df_arcs = pd.read_csv(arcs_file, sep=r"\s+", header=None,
                                  names=["from_node", "to_node"])
            row["arc_count"] = len(df_arcs)
        except Exception as e:
            row["arc_count"] = 0
    else:
        row["arcs_size_bytes"] = 0
        row["arc_count"] = 0

    route_file = find_file(trip_path, ".route")
    row["route_file_exists"] = route_file is not None
    if route_file:
        row["route_size_bytes"] = os.path.getsize(route_file)
        try:
            df_route = pd.read_csv(route_file, sep=r"\s+", header=None,
                                   names=["arc_id"])
            row["route_arc_count"] = len(df_route)
        except Exception as e:
            row["route_arc_count"] = 0
    else:
        row["route_size_bytes"] = 0
        row["route_arc_count"] = 0

    return row


def main():
    rows = []
    for partition, partition_path in PARTITIONS.items():
        if not os.path.exists(partition_path):
            print(f"[WARNING] Partition folder not found: {partition_path}")
            continue
        trip_dirs = sorted(os.listdir(partition_path))
        print(f"\nPartition: {partition} — {len(trip_dirs)} trips found")
        for trip_id in trip_dirs:
            trip_path = os.path.join(partition_path, trip_id)
            if not os.path.isdir(trip_path):
                continue
            row = analyze_trip(trip_id, trip_path, partition)
            rows.append(row)
            print(f"  Trip {trip_id}: {row.get('gps_obs_count',0)} GPS obs, "
                  f"{row.get('node_count',0)} nodes, "
                  f"{row.get('arc_count',0)} arcs, "
                  f"{row.get('route_arc_count',0)} route arcs")
    # ── FIXED: handle empty dataset gracefully ──
    if len(rows) == 0:
        print("\n[ERROR] No trips were found!")
        print("Please make sure your dataset is placed at:")
        print("  data/train/     <- 80 trip folders")
        print("  data/student_test/  <- 10 trip folders")
        return
    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total trips analysed: {len(df)}")
    for partition in df["partition"].unique():
        sub = df[df["partition"] == partition]
        print(f"\n  Partition: {partition} ({len(sub)} trips)")
        print(f"    GPS obs — min: {sub['gps_obs_count'].min()}, "
              f"max: {sub['gps_obs_count'].max()}, "
              f"mean: {sub['gps_obs_count'].mean():.1f}, "
              f"median: {sub['gps_obs_count'].median():.1f}")
        print(f"    Total GPS obs: {sub['gps_obs_count'].sum()}")
        print(f"    Nodes  — mean: {sub['node_count'].mean():.1f}")
        print(f"    Arcs   — mean: {sub['arc_count'].mean():.1f}")
        print(f"    Route arcs — mean: {sub['route_arc_count'].mean():.1f}")
    out_path = os.path.join(OUTPUT_DIR, "trip_statistics.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[SAVED] {out_path}")

if __name__ == "__main__":
    main()