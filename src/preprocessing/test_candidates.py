
import numpy as np

from load_data import load_trip
from candidates import (
    CandidateGenerator,
    project_coordinates,
)


def exact_candidates(generator, longitude, latitude, k=5):
    """Compute distances to every arc, vectorized."""
    point = project_coordinates(
        np.array([longitude]),
        np.array([latitude]),
        generator.lat0,
    )[0]

    starts = generator.starts
    vectors = generator.ends - starts

    squared_lengths = np.einsum(
        "ij,ij->i", vectors, vectors
    )

    offsets = point - starts

    numerator = np.einsum(
        "ij,ij->i", offsets, vectors
    )

    fractions = np.divide(
        numerator,
        squared_lengths,
        out=np.zeros_like(numerator),
        where=squared_lengths > 0,
    )

    fractions = np.clip(fractions, 0, 1)

    projections = starts + fractions[:, None] * vectors

    distances = np.linalg.norm(
        projections - point, axis=1
    )

    indices = np.argsort(distances)[:k]

    return [
        (int(index), float(distances[index]))
        for index in indices
    ]


def main():
    trip = load_trip("data/train/00000001")
    generator = CandidateGenerator(
        trip["nodes"], trip["arcs"]
    )

    observations = trip["track"].iloc[:20]

    total_matches = 0
    total_expected = 0
    exact_top1_found = 0

    for observation in observations.itertuples():
        approximate = generator.get_candidates(
            observation.longitude,
            observation.latitude,
            k=5,
        )

        exact = exact_candidates(
            generator,
            observation.longitude,
            observation.latitude,
            k=5,
        )

        approx_ids = {arc_id for arc_id, _ in approximate}
        exact_ids = {arc_id for arc_id, _ in exact}

        total_matches += len(approx_ids & exact_ids)
        total_expected += len(exact_ids)

        if exact[0][0] in approx_ids:
            exact_top1_found += 1

    print("Trip:", trip["trip_id"])
    print("GPS observations tested:", len(observations))

    print(
        "Top-5 candidate recall:",
        f"{total_matches / total_expected:.2%}"
    )

    print(
        "Exact nearest arc included:",
        f"{exact_top1_found / len(observations):.2%}"
    )


if __name__ == "__main__":
    main()
