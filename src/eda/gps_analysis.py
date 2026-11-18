"""
gps_analysis.py
Member 2 - EDA
PathSnap: GPS Data Quality Analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = r"C:\Users\ayush\OneDrive\Desktop\ml assignment\ml hackathon\Student folder\Student folder"
PARTITIONS = {
    "train": os.path.join(DATA_ROOT, "train"),
    "student_test": os.path.join(DATA_ROOT, "student_test"),
}
PLOTS_DIR = "results/eda/plots"
OUTPUT_DIR = "results/eda"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

GPS_GAP_THRESHOLD_S = 5.0
SPEED_JUMP_THRESHOLD_KMH = 150


def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def find_file(trip_path, extension):
    for f in os.listdir(trip_path):
        if f.endswith(extension):
            return os.path.join(trip_path, f)
    return None


def analyze_gps_quality(trip_id, trip_path, partition):
    result = {
        "trip_id": trip_id, "partition": partition,
        "gps_count": 0, "invalid_coords": 0, "duplicates": 0,
        "non_increasing_timestamps": 0, "gaps_over_5s": 0,
        "max_gap_s": 0.0, "mean_gap_s": 0.0,
        "speed_jumps": 0, "max_speed_kmh": 0.0,
        "mean_speed_kmh": 0.0, "trajectory_length_m": 0.0,
    }

    track_file = find_file(trip_path, ".track")
    if track_file is None:
        return result, [], [], []

    try:
        df = pd.read_csv(track_file, sep=r"\s+", header=None,
                         names=["longitude", "latitude", "timestamp"])
    except Exception:
        return result, [], [], []

    result["gps_count"] = len(df)

    invalid_mask = (
        df["longitude"].isnull() | df["latitude"].isnull() |
        (df["longitude"].abs() > 180) | (df["latitude"].abs() > 90)
    )
    result["invalid_coords"] = int(invalid_mask.sum())
    df = df[~invalid_mask].copy()

    result["duplicates"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    df = df.sort_values("timestamp").reset_index(drop=True)
    time_diffs = df["timestamp"].diff().dropna()
    result["non_increasing_timestamps"] = int((time_diffs <= 0).sum())
    result["gaps_over_5s"] = int((time_diffs > GPS_GAP_THRESHOLD_S).sum())
    result["max_gap_s"] = float(time_diffs.max()) if len(time_diffs) > 0 else 0.0
    result["mean_gap_s"] = float(time_diffs.mean()) if len(time_diffs) > 0 else 0.0

    if len(df) > 1:
        lons = df["longitude"].values
        lats = df["latitude"].values
        timestamps = df["timestamp"].values
        distances_m = haversine_distance(lons[:-1], lats[:-1], lons[1:], lats[1:])
        dt = np.diff(timestamps)
        dt_safe = np.where(dt == 0, np.nan, dt)
        speeds_ms = distances_m / dt_safe
        speeds_kmh = speeds_ms * 3.6
        valid_speeds = speeds_kmh[~np.isnan(speeds_kmh)]
        result["speed_jumps"] = int((valid_speeds > SPEED_JUMP_THRESHOLD_KMH).sum())
        result["max_speed_kmh"] = float(np.nanmax(speeds_kmh)) if len(valid_speeds) > 0 else 0.0
        result["mean_speed_kmh"] = float(np.nanmean(valid_speeds)) if len(valid_speeds) > 0 else 0.0
        result["trajectory_length_m"] = float(np.nansum(distances_m))
        return result, list(time_diffs), list(valid_speeds), list(distances_m)

    return result, list(time_diffs), [], []


def main():
    all_results = []
    all_gaps = []
    all_speeds = []
    all_obs_counts = []

    for partition, partition_path in PARTITIONS.items():
        if not os.path.exists(partition_path):
            print(f"[WARNING] Not found: {partition_path}")
            continue
        trip_dirs = sorted(os.listdir(partition_path))
        print(f"\nAnalysing GPS quality: {partition} ({len(trip_dirs)} trips)")
        for trip_id in trip_dirs:
            trip_path = os.path.join(partition_path, trip_id)
            if not os.path.isdir(trip_path):
                continue
            result, gaps, speeds, _ = analyze_gps_quality(trip_id, trip_path, partition)
            all_results.append(result)
            all_gaps.extend(gaps)
            all_speeds.extend(speeds)
            all_obs_counts.append(result["gps_count"])

    df = pd.DataFrame(all_results)

    print("\n" + "=" * 60)
    print("GPS QUALITY SUMMARY")
    print("=" * 60)
    print(f"Total GPS observations:      {df['gps_count'].sum()}")
    print(f"Invalid coordinates:         {df['invalid_coords'].sum()}")
    print(f"Duplicate rows:              {df['duplicates'].sum()}")
    print(f"Non-increasing timestamps:   {df['non_increasing_timestamps'].sum()}")
    print(f"Gaps > 5 seconds:            {df['gaps_over_5s'].sum()}")
    print(f"Speed jumps > {SPEED_JUMP_THRESHOLD_KMH} km/h:    {df['speed_jumps'].sum()}")
    print(f"Max gap observed:            {df['max_gap_s'].max():.1f} s")

    anomaly_df = df[
        (df["gaps_over_5s"] > 0) | (df["speed_jumps"] > 0) |
        (df["invalid_coords"] > 0) | (df["non_increasing_timestamps"] > 0)
    ][["trip_id", "partition", "gps_count", "gaps_over_5s",
       "max_gap_s", "speed_jumps", "max_speed_kmh", "invalid_coords"]]

    anomaly_df.to_csv(os.path.join(OUTPUT_DIR, "gps_anomaly_summary.csv"), index=False)
    df.to_csv(os.path.join(OUTPUT_DIR, "gps_quality_full.csv"), index=False)
    print(f"[SAVED] gps_anomaly_summary.csv and gps_quality_full.csv")

    # Plot 1: GPS obs per trip
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_obs_counts, bins=20, color="steelblue", edgecolor="white")
    ax.set_xlabel("GPS Observations per Trip", fontsize=12)
    ax.set_ylabel("Number of Trips", fontsize=12)
    ax.set_title("Distribution of GPS Observations per Trip", fontsize=14)
    ax.axvline(np.mean(all_obs_counts), color="red", linestyle="--",
               label=f"Mean = {np.mean(all_obs_counts):.0f}")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "plot1_gps_obs_per_trip.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[SAVED] {out}")

    # Plot 2: Timestamp gaps
    fig, ax = plt.subplots(figsize=(10, 5))
    gaps_arr = np.array(all_gaps)
    ax.hist(gaps_arr[gaps_arr <= 20], bins=40, color="darkorange", edgecolor="white")
    ax.set_xlabel("Time Gap Between Consecutive GPS Points (seconds)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Timestamp Gaps (<=20s shown)", fontsize=14)
    ax.axvline(5, color="red", linestyle="--", label="5-second threshold")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "plot2_timestamp_gaps.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[SAVED] {out}")

    # Plot 3: GPS speeds
    fig, ax = plt.subplots(figsize=(10, 5))
    speeds_arr = np.array(all_speeds)
    ax.hist(speeds_arr[speeds_arr < 200], bins=50, color="seagreen", edgecolor="white")
    ax.set_xlabel("GPS Speed (km/h)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of GPS Speeds (< 200 km/h shown)", fontsize=14)
    ax.axvline(SPEED_JUMP_THRESHOLD_KMH, color="red", linestyle="--",
               label=f"Anomaly threshold = {SPEED_JUMP_THRESHOLD_KMH} km/h")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "plot3_gps_speeds.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[SAVED] {out}")

    print("\n[DONE] GPS analysis complete.")


if __name__ == "__main__":
    main()