"""
visualize.py
Member 2 - EDA
PathSnap: GPS Trajectory + Road Network Visualizations
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

DATA_ROOT = (
    r"C:\Users\ayush\OneDrive\Desktop\ml assignment"
    r"\ml hackathon\Student folder\Student folder"
)

PARTITIONS = {
    "train": os.path.join(DATA_ROOT, "train"),
    "student_test": os.path.join(DATA_ROOT, "student_test"),
}

OUTPUT_DIR = os.path.join("results", "eda")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
MAPS_DIR = os.path.join(OUTPUT_DIR, "maps")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MAPS_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def find_file(trip_path, extension):
    """Find the first file ending with the requested extension."""
    if not os.path.isdir(trip_path):
        return None

    for filename in sorted(os.listdir(trip_path)):
        if filename.endswith(extension):
            return os.path.join(trip_path, filename)

    return None


def normalize_trip_id(value):
    """
    Normalize IDs to the eight-character directory naming convention.

    Examples:
        2          -> 00000002
        2.0        -> 00000002
        '00000002' -> 00000002
    """
    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    if value.isdigit():
        return value.zfill(8)

    return value


def read_numeric_file(path, names):
    """Read a whitespace-separated file and coerce its columns to numeric."""
    try:
        frame = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=names,
        )

        for column in names:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

        return frame.dropna(subset=names).reset_index(drop=True)

    except Exception as exc:
        print(f"  [WARNING] Could not read {path}: {exc}")
        return None


def load_trip(trip_path):
    """Load the track, nodes, arcs, and route files for one trip."""
    data = {
        "track": None,
        "nodes": None,
        "arcs": None,
        "route": None,
    }

    track_file = find_file(trip_path, ".track")
    if track_file:
        data["track"] = read_numeric_file(
            track_file,
            ["longitude", "latitude", "timestamp"],
        )

    nodes_file = find_file(trip_path, ".nodes")
    if nodes_file:
        data["nodes"] = read_numeric_file(
            nodes_file,
            ["longitude", "latitude"],
        )

    arcs_file = find_file(trip_path, ".arcs")
    if arcs_file:
        data["arcs"] = read_numeric_file(
            arcs_file,
            ["from_node", "to_node"],
        )

    route_file = find_file(trip_path, ".route")
    if route_file:
        data["route"] = read_numeric_file(
            route_file,
            ["arc_id"],
        )

    # Node and arc IDs are expected to be integer indices.
    if data["arcs"] is not None and not data["arcs"].empty:
        data["arcs"]["from_node"] = data["arcs"]["from_node"].astype(int)
        data["arcs"]["to_node"] = data["arcs"]["to_node"].astype(int)

    if data["route"] is not None and not data["route"].empty:
        data["route"]["arc_id"] = data["route"]["arc_id"].astype(int)

    return data


def find_gaps(df_track, threshold=5.0):
    """Return row indices where the time gap exceeds threshold seconds."""
    if df_track is None or df_track.empty:
        return []

    df = df_track.sort_values("timestamp").reset_index(drop=True)
    timestamps = pd.to_numeric(df["timestamp"], errors="coerce")
    differences = timestamps.diff()

    return df.index[differences > threshold].tolist()


def valid_node_index(value, node_count):
    """Check whether a node index is valid for positional indexing."""
    try:
        index = int(value)
        return 0 <= index < node_count
    except (TypeError, ValueError, OverflowError):
        return False


# ---------------------------------------------------------------------
# MATPLOTLIB VISUALIZATION
# ---------------------------------------------------------------------

def plot_gps_and_network(trip_id, trip_path, partition):
    """Plot GPS observations, road network, and reference route."""
    data = load_trip(trip_path)

    df_track = data["track"]
    df_nodes = data["nodes"]
    df_arcs = data["arcs"]
    df_route = data["route"]

    if (
        df_track is None or df_track.empty
        or df_nodes is None or df_nodes.empty
        or df_arcs is None or df_arcs.empty
    ):
        print(f"  [SKIP] Missing or empty track/nodes/arcs for trip {trip_id}")
        return

    df_sorted = df_track.sort_values("timestamp").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw road network in grey.
    drawn = 0
    for _, arc in df_arcs.iterrows():
        if drawn >= 5000:
            break

        from_node = int(arc["from_node"])
        to_node = int(arc["to_node"])

        if not (
            valid_node_index(from_node, len(df_nodes))
            and valid_node_index(to_node, len(df_nodes))
        ):
            continue

        x = [
            df_nodes.iloc[from_node]["longitude"],
            df_nodes.iloc[to_node]["longitude"],
        ]
        y = [
            df_nodes.iloc[from_node]["latitude"],
            df_nodes.iloc[to_node]["latitude"],
        ]

        ax.plot(
            x, y,
            color="lightgrey",
            linewidth=0.6,
            zorder=1,
        )
        drawn += 1

    # Draw reference route in green.
    if df_route is not None and not df_route.empty:
        first_route_arc = True

        for arc_id in df_route["arc_id"].tolist():
            if not valid_node_index(arc_id, len(df_arcs)):
                continue

            arc = df_arcs.iloc[int(arc_id)]
            from_node = int(arc["from_node"])
            to_node = int(arc["to_node"])

            if not (
                valid_node_index(from_node, len(df_nodes))
                and valid_node_index(to_node, len(df_nodes))
            ):
                continue

            label = "Reference Route" if first_route_arc else "_nolegend_"

            x = [
                df_nodes.iloc[from_node]["longitude"],
                df_nodes.iloc[to_node]["longitude"],
            ]
            y = [
                df_nodes.iloc[from_node]["latitude"],
                df_nodes.iloc[to_node]["latitude"],
            ]

            ax.plot(
                x, y,
                color="green",
                linewidth=2.5,
                zorder=2,
                label=label,
            )
            first_route_arc = False

    # Draw GPS observations in blue.
    ax.scatter(
        df_sorted["longitude"],
        df_sorted["latitude"],
        c="royalblue",
        s=4,
        zorder=3,
        alpha=0.6,
        label="GPS Observations",
    )

    # Mark start and end.
    ax.scatter(
        df_sorted.iloc[0]["longitude"],
        df_sorted.iloc[0]["latitude"],
        c="lime",
        s=150,
        marker="^",
        zorder=5,
        label="Start",
    )
    ax.scatter(
        df_sorted.iloc[-1]["longitude"],
        df_sorted.iloc[-1]["latitude"],
        c="red",
        s=150,
        marker="v",
        zorder=5,
        label="End",
    )

    ax.set_title(
        f"Trip {trip_id} ({partition}) — GPS + Road Network + Reference Route",
        fontsize=13,
    )
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=9)

    plt.tight_layout()

    output_path = os.path.join(
        PLOTS_DIR,
        f"plot_trip_{trip_id}_{partition}.png",
    )
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"  [SAVED] {output_path}")


# ---------------------------------------------------------------------
# FOLIUM INTERACTIVE MAP
# ---------------------------------------------------------------------

def folium_map(trip_id, trip_path, partition, label=""):
    """Create an interactive map with GPS, network, route, and markers."""
    data = load_trip(trip_path)

    df_track = data["track"]
    df_nodes = data["nodes"]
    df_arcs = data["arcs"]
    df_route = data["route"]

    if df_track is None or df_track.empty:
        print(f"  [SKIP] No valid track data for trip {trip_id}")
        return

    df_sorted = df_track.sort_values("timestamp").reset_index(drop=True)

    center_lat = float(df_sorted["latitude"].mean())
    center_lon = float(df_sorted["longitude"].mean())

    map_object = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="CartoDB positron",
    )

    # Layer 1: GPS trajectory.
    gps_layer = folium.FeatureGroup(
        name="GPS Observations (blue)"
    )

    gps_coords = list(
        zip(
            df_sorted["latitude"].astype(float),
            df_sorted["longitude"].astype(float),
        )
    )

    if len(gps_coords) >= 2:
        folium.PolyLine(
            gps_coords,
            color="blue",
            weight=1.5,
            opacity=0.7,
            tooltip="GPS Track",
        ).add_to(gps_layer)

    # Mark GPS gaps greater than five seconds.
    gap_indices = find_gaps(df_sorted, threshold=5.0)

    for gap_index in gap_indices:
        if gap_index < len(df_sorted):
            folium.CircleMarker(
                location=[
                    float(df_sorted.iloc[gap_index]["latitude"]),
                    float(df_sorted.iloc[gap_index]["longitude"]),
                ],
                radius=5,
                color="red",
                fill=True,
                tooltip=f"GPS Gap at index {gap_index}",
            ).add_to(gps_layer)

    gps_layer.add_to(map_object)

    # Layer 2: Road network, limited to the GPS bounding box.
    if (
        df_nodes is not None and not df_nodes.empty
        and df_arcs is not None and not df_arcs.empty
    ):
        network_layer = folium.FeatureGroup(
            name="Road Network (grey)"
        )

        lat_min = df_sorted["latitude"].min() - 0.005
        lat_max = df_sorted["latitude"].max() + 0.005
        lon_min = df_sorted["longitude"].min() - 0.005
        lon_max = df_sorted["longitude"].max() + 0.005

        drawn = 0

        for _, arc in df_arcs.iterrows():
            if drawn >= 3000:
                break

            from_node = int(arc["from_node"])
            to_node = int(arc["to_node"])

            if not (
                valid_node_index(from_node, len(df_nodes))
                and valid_node_index(to_node, len(df_nodes))
            ):
                continue

            from_point = df_nodes.iloc[from_node]
            to_point = df_nodes.iloc[to_node]

            # Keep arcs whose endpoints are near the GPS bounding box.
            if not (
                lat_min <= from_point["latitude"] <= lat_max
                and lon_min <= from_point["longitude"] <= lon_max
            ):
                continue

            folium.PolyLine(
                [
                    (float(from_point["latitude"]),
                     float(from_point["longitude"])),
                    (float(to_point["latitude"]),
                     float(to_point["longitude"])),
                ],
                color="grey",
                weight=1.0,
                opacity=0.5,
            ).add_to(network_layer)

            drawn += 1

        network_layer.add_to(map_object)

    # Layer 3: Reference route.
    if (
        df_route is not None and not df_route.empty
        and df_nodes is not None and not df_nodes.empty
        and df_arcs is not None and not df_arcs.empty
    ):
        route_layer = folium.FeatureGroup(
            name="Reference Route (green)"
        )

        for arc_id in df_route["arc_id"].tolist():
            if not valid_node_index(arc_id, len(df_arcs)):
                continue

            arc = df_arcs.iloc[int(arc_id)]
            from_node = int(arc["from_node"])
            to_node = int(arc["to_node"])

            if not (
                valid_node_index(from_node, len(df_nodes))
                and valid_node_index(to_node, len(df_nodes))
            ):
                continue

            from_point = df_nodes.iloc[from_node]
            to_point = df_nodes.iloc[to_node]

            folium.PolyLine(
                [
                    (float(from_point["latitude"]),
                     float(from_point["longitude"])),
                    (float(to_point["latitude"]),
                     float(to_point["longitude"])),
                ],
                color="green",
                weight=3.5,
                opacity=0.9,
                tooltip=f"Arc {int(arc_id)}",
            ).add_to(route_layer)

        route_layer.add_to(map_object)

    # Start and end markers.
    folium.Marker(
        location=[
            float(df_sorted.iloc[0]["latitude"]),
            float(df_sorted.iloc[0]["longitude"]),
        ],
        popup=f"START — Trip {trip_id}",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(map_object)

    folium.Marker(
        location=[
            float(df_sorted.iloc[-1]["latitude"]),
            float(df_sorted.iloc[-1]["longitude"]),
        ],
        popup=f"END — Trip {trip_id}",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(map_object)

    # Title box.
    title_html = f"""
    <div style="
        position:fixed;
        top:10px;
        left:50%;
        transform:translateX(-50%);
        z-index:9999;
        background:white;
        padding:8px 16px;
        border-radius:6px;
        border:2px solid #333;
        font-size:14px;
        font-weight:bold;">
        Trip {trip_id} ({partition}) {label}
    </div>
    """

    map_object.get_root().html.add_child(
        folium.Element(title_html)
    )
    folium.LayerControl(collapsed=False).add_to(map_object)

    output_path = os.path.join(
        MAPS_DIR,
        f"map_trip_{trip_id}_{partition}{label}.html",
    )
    map_object.save(output_path)

    print(f"  [SAVED] {output_path}")


# ---------------------------------------------------------------------
# HELPERS TO PICK TRIPS
# ---------------------------------------------------------------------

def pick_trips(partition_path, partition, n=5):
    """Return up to n trip directories from a partition."""
    if not os.path.isdir(partition_path):
        print(f"[WARNING] Partition directory not found: {partition_path}")
        return []

    trips = []

    for trip_id in sorted(os.listdir(partition_path)):
        trip_path = os.path.join(partition_path, trip_id)

        if os.path.isdir(trip_path):
            trips.append((trip_id, trip_path, partition))

    return trips[:n]


def find_gap_trips(partition, top_n=1):
    """Find trips with the largest number of GPS gaps."""
    if partition not in PARTITIONS:
        print(f"[WARNING] Unknown partition: {partition}")
        return []

    csv_path = os.path.join(
        OUTPUT_DIR,
        "gps_quality_full.csv",
    )

    if not os.path.isfile(csv_path):
        print(
            f"[WARNING] {csv_path} not found. "
            "Using the first available trips instead."
        )
        return pick_trips(
            PARTITIONS[partition],
            partition,
            n=top_n,
        )

    try:
        # Critical fix: read trip_id as text, then normalize it.
        df = pd.read_csv(
            csv_path,
            dtype={"trip_id": str},
        )

        required = {"trip_id", "partition", "gaps_over_5s"}
        missing = required - set(df.columns)

        if missing:
            print(
                "[WARNING] gps_quality_full.csv is missing columns: "
                f"{sorted(missing)}. Using fallback trips."
            )
            return pick_trips(
                PARTITIONS[partition],
                partition,
                n=top_n,
            )

        df["trip_id"] = df["trip_id"].apply(normalize_trip_id)
        df["partition"] = df["partition"].astype(str).str.strip()
        df["gaps_over_5s"] = pd.to_numeric(
            df["gaps_over_5s"],
            errors="coerce",
        )

        df = df[df["partition"] == partition]
        df = df.dropna(subset=["gaps_over_5s"])
        df = df.sort_values(
            "gaps_over_5s",
            ascending=False,
        )

        result = []

        for _, row in df.head(top_n).iterrows():
            trip_id = normalize_trip_id(row["trip_id"])
            trip_path = os.path.join(
                PARTITIONS[partition],
                trip_id,
            )

            if os.path.isdir(trip_path):
                result.append((trip_id, trip_path, partition))

        return result if result else pick_trips(
            PARTITIONS[partition],
            partition,
            n=top_n,
        )

    except Exception as exc:
        print(f"[WARNING] Could not select GPS-gap trips: {exc}")
        return pick_trips(
            PARTITIONS[partition],
            partition,
            n=top_n,
        )


def find_discontinuity_trips(partition, top_n=1):
    """Find trips with the largest number of route discontinuities."""
    if partition not in PARTITIONS:
        print(f"[WARNING] Unknown partition: {partition}")
        return []

    csv_path = os.path.join(
        OUTPUT_DIR,
        "network_analysis.csv",
    )

    if not os.path.isfile(csv_path):
        print(
            f"[WARNING] {csv_path} not found. "
            "Using the first available trips instead."
        )
        return pick_trips(
            PARTITIONS[partition],
            partition,
            n=top_n,
        )

    try:
        # Critical fix: read trip_id as text, then normalize it.
        df = pd.read_csv(
            csv_path,
            dtype={"trip_id": str},
        )

        required = {
            "trip_id",
            "partition",
            "route_discontinuities",
        }
        missing = required - set(df.columns)

        if missing:
            print(
                "[WARNING] network_analysis.csv is missing columns: "
                f"{sorted(missing)}. Using fallback trips."
            )
            return pick_trips(
                PARTITIONS[partition],
                partition,
                n=top_n,
            )

        df["trip_id"] = df["trip_id"].apply(normalize_trip_id)
        df["partition"] = df["partition"].astype(str).str.strip()
        df["route_discontinuities"] = pd.to_numeric(
            df["route_discontinuities"],
            errors="coerce",
        )

        df = df[
            (df["partition"] == partition)
            & (df["route_discontinuities"] > 0)
        ]

        df = df.sort_values(
            "route_discontinuities",
            ascending=False,
        )

        result = []

        for _, row in df.head(top_n).iterrows():
            trip_id = normalize_trip_id(row["trip_id"])
            trip_path = os.path.join(
                PARTITIONS[partition],
                trip_id,
            )

            if os.path.isdir(trip_path):
                result.append((trip_id, trip_path, partition))

        return result if result else pick_trips(
            PARTITIONS[partition],
            partition,
            n=top_n,
        )

    except Exception as exc:
        print(f"[WARNING] Could not select discontinuity trips: {exc}")
        return pick_trips(
            PARTITIONS[partition],
            partition,
            n=top_n,
        )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    # Check that the input partition folders exist.
    available_partitions = {
        name: path
        for name, path in PARTITIONS.items()
        if os.path.isdir(path)
    }

    if not available_partitions:
        print("[ERROR] No dataset partition folders were found.")
        print(f"[ERROR] Check DATA_ROOT: {DATA_ROOT}")
        return

    # Matplotlib: 3 training + 2 student-test plots.
    print("\n=== Matplotlib: GPS + Network plots ===")

    sample = pick_trips(
        PARTITIONS["train"],
        "train",
        n=3,
    )
    sample += pick_trips(
        PARTITIONS["student_test"],
        "student_test",
        n=2,
    )

    for trip_id, trip_path, partition in sample:
        print(f"  Plotting trip {trip_id} ({partition})")
        plot_gps_and_network(
            trip_id,
            trip_path,
            partition,
        )

    # Folium Map 1: Normal training trip.
    print("\n=== Folium Map 1: Normal trip ===")

    normal_trips = pick_trips(
        PARTITIONS["train"],
        "train",
        n=1,
    )

    for trip_id, trip_path, partition in normal_trips:
        print(f"  Trip {trip_id}")
        folium_map(
            trip_id,
            trip_path,
            partition,
            label="_normal",
        )

    # Folium Map 2: Training trip with GPS gaps.
    print("\n=== Folium Map 2: GPS gap trip ===")

    for trip_id, trip_path, partition in find_gap_trips(
        "train",
        top_n=1,
    ):
        print(f"  Trip {trip_id}")
        folium_map(
            trip_id,
            trip_path,
            partition,
            label="_gps_gaps",
        )

    # Folium Map 3: Training trip with route discontinuities.
    print("\n=== Folium Map 3: Route discontinuity trip ===")

    for trip_id, trip_path, partition in find_discontinuity_trips(
        "train",
        top_n=1,
    ):
        print(f"  Trip {trip_id}")
        folium_map(
            trip_id,
            trip_path,
            partition,
            label="_route_discontinuity",
        )

    # Folium Map 4: Student-test trip.
    print("\n=== Folium Map 4: Student test trip ===")

    student_test_trips = pick_trips(
        PARTITIONS["student_test"],
        "student_test",
        n=1,
    )

    for trip_id, trip_path, partition in student_test_trips:
        print(f"  Trip {trip_id}")
        folium_map(
            trip_id,
            trip_path,
            partition,
            label="_student_test",
        )

    print("\n[DONE] Visualization script finished.")
    print(f"  Plots -> {PLOTS_DIR}/")
    print(f"  Maps  -> {MAPS_DIR}/")


if __name__ == "__main__":
    main()
