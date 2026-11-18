
import numpy as np
from scipy.spatial import cKDTree

from load_data import load_trip


EARTH_RADIUS_M = 6371000.0


def project_coordinates(lon, lat, lat0):
    """Local equirectangular projection into metres."""
    x = (
        np.radians(lon)
        * EARTH_RADIUS_M
        * np.cos(np.radians(lat0))
    )
    y = np.radians(lat) * EARTH_RADIUS_M
    return np.column_stack((x, y))


def point_to_segment_distance(point, start, end):
    """Distance in metres from a point to a line segment."""
    vector = end - start
    denominator = np.dot(vector, vector)

    if denominator == 0:
        return float(np.linalg.norm(point - start))

    t = np.clip(
        np.dot(point - start, vector) / denominator,
        0.0,
        1.0,
    )

    projection = start + t * vector
    return float(np.linalg.norm(point - projection))


class CandidateGenerator:
    def __init__(self, nodes, arcs):
        self.arcs = arcs

        lat0 = float(nodes["latitude"].mean())

        node_xy = project_coordinates(
            nodes["longitude"].to_numpy(),
            nodes["latitude"].to_numpy(),
            lat0,
        )

        from_ids = arcs["from_node_id"].to_numpy(dtype=int)
        to_ids = arcs["to_node_id"].to_numpy(dtype=int)

        self.starts = node_xy[from_ids]
        self.ends = node_xy[to_ids]

        self.midpoints = (self.starts + self.ends) / 2.0
        self.half_lengths = np.linalg.norm(
            self.ends - self.starts, axis=1
        ) / 2.0

        self.tree = cKDTree(self.midpoints)
        self.lat0 = lat0

    def get_candidates(self, longitude, latitude, k=5):
        point = project_coordinates(
            np.array([longitude]),
            np.array([latitude]),
            self.lat0,
        )[0]

        # Initial shortlist using midpoint proximity
        shortlist_size = min(
            max(k * 20, 100),
            len(self.arcs),
        )

        _, indices = self.tree.query(
            point, k=shortlist_size
        )

        indices = np.atleast_1d(indices)

        starts = self.starts[indices]
        vectors = self.ends[indices] - starts
        offsets = point - starts
        squared_lengths = np.einsum("ij,ij->i", vectors, vectors)
        fractions = np.divide(
            np.einsum("ij,ij->i", offsets, vectors),
            squared_lengths,
            out=np.zeros_like(squared_lengths),
            where=squared_lengths > 0,
        )
        projections = starts + np.clip(fractions, 0, 1)[:, None] * vectors
        distances = np.linalg.norm(point - projections, axis=1)
        order = np.argsort(distances, kind="stable")[:k]
        return [(int(indices[i]), float(distances[i])) for i in order]


def main():
    trip = load_trip("data/train/00000001")

    generator = CandidateGenerator(
        trip["nodes"],
        trip["arcs"],
    )

    first_gps = trip["track"].iloc[0]

    candidates = generator.get_candidates(
        first_gps["longitude"],
        first_gps["latitude"],
        k=5,
    )

    print("Trip:", trip["trip_id"])
    print("First GPS observation:")
    print(first_gps)

    print("\nTop 5 candidate road segments:")

    for rank, (arc_id, distance) in enumerate(
        candidates, start=1
    ):
        print(
            f"{rank}. Arc {arc_id}: "
            f"{distance:.2f} metres"
        )


if __name__ == "__main__":
    main()
