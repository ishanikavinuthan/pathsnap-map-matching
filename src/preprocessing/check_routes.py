
from pathlib import Path

from load_data import load_trip


def inspect_route(trip):
    arcs = trip["arcs"]
    route = trip["route"]["arc_id"].dropna().astype(int)

    problems = []

    for position in range(len(route) - 1):
        current_id = int(route.iloc[position])
        next_id = int(route.iloc[position + 1])

        current_arc = arcs.iloc[current_id]
        next_arc = arcs.iloc[next_id]

        current_end = int(current_arc["to_node_id"])
        next_start = int(next_arc["from_node_id"])

        if current_end != next_start:
            problems.append({
                "trip_id": trip["trip_id"],
                "position": position,
                "current_arc": current_id,
                "next_arc": next_id,
                "current_end_node": current_end,
                "next_start_node": next_start,
            })

    return problems


def main():
    for partition in ["train", "student_test"]:
        root = Path("data") / partition

        for trip_dir in sorted(root.iterdir()):
            if not trip_dir.is_dir():
                continue

            trip = load_trip(trip_dir)
            problems = inspect_route(trip)

            for problem in problems:
                print(f"\nPartition: {partition}")
                for key, value in problem.items():
                    print(f"{key}: {value}")


if __name__ == "__main__":
    main()
