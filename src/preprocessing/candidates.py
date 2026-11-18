
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
    """Distance in metres from a point to a finite segment."""
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

        self.max_half_length = float(
            self.half_lengths.max()
        )

    def get_candidates(self, longitude, latitude, k=5):
        point = project_coordinates(
            np.array([longitude]),
            np.array([latitude]),
            self.lat0,
        )[0]

        total = len(self.arcs)

        if not 1 <= k <= total:
            raise ValueError("k must be between 1 and arc count")

        # Find initial nearby segments.
        shortlist_size = min(max(k * 20, 100), total)

        while True:
            midpoint_distances, indices = self.tree.query(
                point,
                k=shortlist_size,
            )

            indices = np.atleast_1d(indices)
            midpoint_distances = np.atleast_1d(
                midpoint_distances
            )

            candidates = []

            for arc_id in indices:
                arc_id = int(arc_id)

                distance = point_to_segment_distance(
                    point,
                    self.starts[arc_id],
                    self.ends[arc_id],
                )

                candidates.append((arc_id, distance))

            candidates.sort(
                key=lambda item: (item[1], item[0])
            )

            top_k = candidates[:k]

            if shortlist_size == total:
                return top_k

            kth_distance = top_k[-1][1]

            # All unseen midpoint distances are at least
            # as large as the farthest queried midpoint.
            # The maximum half-length provides a safe bound.
            unseen_lower_bound = (
                midpoint_distances[-1]
                - self.max_half_length
            )

            if unseen_lower_bound > kth_distance:
                return top_k

            shortlist_size = min(
                shortlist_size * 2,
                total,
            )


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
