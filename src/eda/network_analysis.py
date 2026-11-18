"""
network_analysis.py
Member 2 - EDA
PathSnap: Road Network Connectivity Analysis
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def find_file(trip_path, extension):
    """Return the first file in a trip directory with the given extension."""
    for filename in sorted(os.listdir(trip_path)):
        if filename.endswith(extension):
            return os.path.join(trip_path, filename)
    return None


def normalize_trip_id(value):
    """
    Normalize trip IDs to strings.

    The dataset uses IDs such as 00000002. This also handles IDs that may
    have been read or exported as values such as 2 or 2.0.
    """
    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    if value.isdigit():
        return value.zfill(8)

    return value


def analyze_network(trip_id, trip_path, partition):
    """Analyze nodes, arcs, connectivity, degrees, and route continuity."""

    result = {
        "trip_id": str(trip_id),
        "partition": partition,
        "node_count": 0,
        "arc_count": 0,
        "invalid_arc_endpoints": 0,
        "isolated_nodes": 0,
        "weakly_connected_components": 0,
        "largest_component_nodes": 0,
        "network_density": 0.0,
        "route_arc_count": 0,
        "route_discontinuities": 0,
        "max_in_degree": 0,
        "max_out_degree": 0,
        "mean_out_degree": 0.0,
    }

    nodes_file = find_file(trip_path, ".nodes")
    if nodes_file is None:
        print(f"[WARNING] No .nodes file found for trip {trip_id}")
        return result

    try:
        df_nodes = pd.read_csv(
            nodes_file,
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude"],
        )
        result["node_count"] = len(df_nodes)
    except Exception as exc:
        print(f"[WARNING] Could not read nodes for trip {trip_id}: {exc}")
        return result

    arcs_file = find_file(trip_path, ".arcs")
    if arcs_file is None:
        print(f"[WARNING] No .arcs file found for trip {trip_id}")
        return result

    try:
        df_arcs = pd.read_csv(
            arcs_file,
            sep=r"\s+",
            header=None,
            names=["from_node", "to_node"],
        )

        # Ensure endpoint columns are numeric.
        df_arcs["from_node"] = pd.to_numeric(
            df_arcs["from_node"], errors="coerce"
        )
        df_arcs["to_node"] = pd.to_numeric(
            df_arcs["to_node"], errors="coerce"
        )
    except Exception as exc:
        print(f"[WARNING] Could not read arcs for trip {trip_id}: {exc}")
        return result
