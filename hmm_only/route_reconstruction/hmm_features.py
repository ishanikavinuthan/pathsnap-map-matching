"""Six emission and twelve transition features for the extended decoder."""
import numpy as np


def angle(first, second):
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator < 1e-9:
        return 0.0
    return float(np.arccos(np.clip(np.dot(first, second) / denominator, -1, 1)))


def emission_features(network, point, movement, states, probabilities):
    features = []
    for (arc, offset), probability in zip(states, probabilities):
        length = float(network.lengths[arc])
        fraction = offset / length if length else 0.0
        direction = network.vectors[arc]
        projection = network.starts[arc] + fraction * direction
        junction = int(offset <= 15 and network.degree.get(network.arcs[arc, 0], 0) > 2
                       or length - offset <= 15 and network.degree.get(network.arcs[arc, 1], 0) > 2)
        features.append([
            -np.log(max(probability, 1e-12)), np.linalg.norm(point - projection),
            angle(movement, direction), fraction, length, junction,
        ])
    return np.array(features)


def emission_cost(features):
    nll, distance, heading, fraction, length, junction = features.T
    endpoint_proximity = 1 - 2 * np.minimum(fraction, 1 - fraction)
    return (nll + 0.12 * np.minimum(distance / 20, 4)
            + 0.14 * heading / np.pi * (1 - 0.5 * junction)
            + 0.04 * endpoint_proximity / (1 + length / 50))


def transition_features(network, first, second, bridge, distance, dt, movement,
                        previous_movement, previous_dt):
    a, offset_a = first
    b, offset_b = second
    displacement = float(np.linalg.norm(movement))
    observed_speed = displacement / dt
    has_previous_motion = previous_dt > 0 and np.linalg.norm(previous_movement) > 1e-9
    acceleration = (
        (observed_speed - np.linalg.norm(previous_movement) / previous_dt) / dt
        if has_previous_motion else 0.0
    )
    observed_turn = angle(previous_movement, movement) if has_previous_motion else 0.0
    path = (a, *bridge)
    turns = [angle(network.vectors[x], network.vectors[y]) for x, y in zip(path, path[1:])]
    return (
        dt, displacement, acceleration, observed_turn, distance,
        abs(distance - displacement), distance / dt,
        float(network.lengths[a] - offset_a) if bridge else 0.0,
        offset_b if bridge else 0.0, max(0, len(bridge) - 1),
        sum(turn > np.pi / 4 for turn in turns), int(any(turn >= 5 * np.pi / 6 for turn in turns)),
    )


def transition_cost(features, beta):
    (dt, displacement, acceleration, turn, distance, mismatch, speed,
     exit_distance, entry_distance, connectors, turns, u_turn) = features
    uncertainty = beta + min(20, displacement * 0.2) + min(15, abs(acceleration))
    return (0.25 * mismatch / uncertainty
            + 0.02 * min(distance / (30 + 5 * dt), 4)
            + 0.08 * min(abs(speed - displacement / dt) / 15, 4)
            + 0.03 * min((exit_distance + entry_distance) / 50, 4)
            + 0.02 * min(connectors, 5)
            + 0.07 * min(turns, 3) * (1 - turn / np.pi)
            + 0.12 * u_turn * (1 - turn / np.pi))
