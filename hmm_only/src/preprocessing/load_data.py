
from pathlib import Path
import numpy as np
import pandas as pd


FILE_COLUMNS = {
    "track": ["longitude", "latitude", "timestamp"],
    "nodes": ["longitude", "latitude"],
    "arcs": ["from_node_id", "to_node_id"],
    "route": ["arc_id"],
}


def read_raw_file(path, file_type):
    """Read one headerless, whitespace-separated PathSnap file."""
    if file_type not in FILE_COLUMNS:
        raise ValueError(f"Unknown file type: {file_type}")

    columns = FILE_COLUMNS[file_type]

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=columns,
        dtype=str,
    )

    if file_type in ("arcs", "route"):
        for col in columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

            # Preserve missing or malformed integer values
            df[col] = df[col].astype("Int64")
    else:
        for col in columns:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

    return df


def load_trip(trip_dir):
    """Load GPS, road network and reference route for one trip."""
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name

    data = {"trip_id": trip_id}

    for file_type in FILE_COLUMNS:
        path = trip_dir / f"{trip_id}.{file_type}"

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {file_type} file: {path}"
            )

        data[file_type] = read_raw_file(path, file_type)

    # Preserve the original zero-based identifiers
    data["nodes"].index.name = "node_id"
    data["arcs"].index.name = "arc_id"

    return data


if __name__ == "__main__":
    trip = load_trip("data/train/00000001")

    print("Trip ID:", trip["trip_id"])

    for file_type in FILE_COLUMNS:
        df = trip[file_type]

        print(f"\n--- {file_type.upper()} ---")
        print("Rows:", len(df))
        print("Columns:", list(df.columns))
        print("Missing values:")
        print(df.isna().sum())
        print(df.head())
