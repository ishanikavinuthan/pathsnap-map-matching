"""Directed-road HMM and Viterbi decoding."""
from collections import defaultdict
from heapq import heappop, heappush

import numpy as np
import pandas as pd

from hmm_features import emission_cost, emission_features, transition_cost, transition_features


class RoadNetwork:
    def __init__(self, nodes_xy, arcs):
        self.arcs = np.asarray(arcs, dtype=int)
        if np.any(self.arcs < 0) or np.any(self.arcs >= len(nodes_xy)):
            raise ValueError("Road arc refers to an invalid node.")
        self.starts = nodes_xy[self.arcs[:, 0]]
        self.vectors = nodes_xy[self.arcs[:, 1]] - self.starts
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        neighbors = defaultdict(set)
        self.outgoing = defaultdict(list)
        for arc, (source, target) in enumerate(self.arcs):
            if source != target:
                neighbors[source].add(target)
                neighbors[target].add(source)
            self.outgoing[source].append((target, float(self.lengths[arc]), arc))

        self.degree = {node: len(adjacent) for node, adjacent in neighbors.items()}

    def position(self, point, arc):
        vector = self.vectors[arc]
        length = self.lengths[arc]
        if length == 0:
            return 0.0
        return float(np.clip(np.dot(point - self.starts[arc], vector) / length, 0, length))

    def shortest_paths(self, start, targets, limit):
        """Find bounded directed shortest paths, preserving IDs of parallel arcs."""
        remaining = set(targets)
        distances = {start: 0.0}
        previous = {}
        queue = [(0.0, start)]
        found = {}
        while queue and remaining:
            distance, node = heappop(queue)
            if distance != distances[node]:
                continue
            if node in remaining:
                remaining.remove(node)
                path = []
                cursor = node
                while cursor != start:
                    cursor, arc = previous[cursor]
                    path.append(int(arc))
                found[node] = (distance, tuple(reversed(path)))
            for target, length, arc in self.outgoing[node]:
                candidate = distance + length
                if candidate <= limit and candidate < distances.get(target, np.inf):
                    distances[target] = candidate
                    previous[target] = (node, arc)
                    heappush(queue, (candidate, target))
        return found

    def transitions(self, first, second, limit, gps_displacement, beta, motion=None):
        scores = np.full((len(first), len(second)), -np.inf)
        bridges = {}
        targets = {self.arcs[arc, 0] for arc, _ in second}
        paths_by_node = {}
        for i, (a, offset_a) in enumerate(first):
            end = self.arcs[a, 1]
            if end not in paths_by_node:
                paths_by_node[end] = self.shortest_paths(end, targets, limit)
            for j, (b, offset_b) in enumerate(second):
                start = self.arcs[b, 0]
                if a == b and offset_b >= offset_a - 10.0:
                    distance, bridge = max(0.0, offset_b - offset_a), ()
                elif start in paths_by_node[end]:
                    gap, between = paths_by_node[end][start]
                    distance = self.lengths[a] - offset_a + gap + offset_b
                    bridge = (*between, int(b))
                else:
                    continue
                if distance <= limit:
                    scores[i, j] = -abs(distance - gps_displacement) / beta
                    if motion is not None:
                        features = transition_features(
                            self, first[i], second[j], bridge, distance, *motion
                        )
                        scores[i, j] = -transition_cost(features, beta)
                    bridges[i, j] = bridge
        return scores, bridges


def decode(points, timestamps, network, scored, *, beta=25.0, max_speed=55.0,
           max_gap=30.0, gps_slack=30.0, extended_features=False):
    """Return one row per observation and the ordered arc IDs for each fragment.

    scored has integer observation_id and arc_id columns and a score in [0, 1].
    Transition limits use metres and seconds. Unreachable transitions start a
    new fragment only when no continuation of the current fragment is possible.
    """
    points = np.asarray(points, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    parameters = [beta, max_speed, max_gap, gps_slack]
    if not np.isfinite(parameters).all() or min(parameters[:3]) <= 0 or gps_slack < 0:
        raise ValueError("Use positive beta, speed and gap, and nonnegative slack.")
    if not len(points) or points.shape != (len(timestamps), 2):
        raise ValueError("Expected at least one GPS point and one timestamp per point.")
    scores = scored["score"].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Candidate scores must be finite and between 0 and 1.")
    for column, upper in (("observation_id", len(points)), ("arc_id", len(network.arcs))):
        ids = scored[column].to_numpy(dtype=float)
        if not np.isfinite(ids).all() or np.any((ids % 1 != 0) | (ids < 0) | (ids >= upper)):
            raise ValueError(f"Invalid {column} in candidate scores.")
    if scored.duplicated(["observation_id", "arc_id"]).any():
        raise ValueError("Each candidate arc must occur only once per observation.")

    groups = {int(key): group for key, group in scored.groupby("observation_id", sort=False)}
    rows = []
    fragments = []
    history = []
    values = None

    def finish_fragment():
        if not history:
            return
        chosen = int(np.argmax(values))
        steps = []
        fragment_id = len(fragments)
        for observation, arc_ids, backpointers, bridges in reversed(history):
            arc = int(arc_ids[chosen])
            rows[observation]["viterbi_arc_id"] = arc
            rows[observation]["fragment_id"] = fragment_id
            previous = int(backpointers[chosen])
            steps.append((arc,) if previous < 0 else bridges[previous, chosen])
            chosen = previous
        fragments.append([arc for step in reversed(steps) for arc in step])
        history.clear()

    for i, (point, timestamp) in enumerate(zip(points, timestamps)):
        row = {"observation_id": i, "classifier_arc_id": None,
               "viterbi_arc_id": None, "fragment_id": None, "break_reason": ""}
        rows.append(row)
        candidates = groups.get(i)
        if not np.isfinite(point).all() or not np.isfinite(timestamp) or candidates is None:
            finish_fragment()
            row["break_reason"] = "invalid_gps" if (
                not np.isfinite(point).all() or not np.isfinite(timestamp)
            ) else "no_candidates"
            continue
        arc_ids = candidates["arc_id"].to_numpy(dtype=int)
        emissions = np.log(np.maximum(candidates["score"].to_numpy(dtype=float), 1e-12))
        row["classifier_arc_id"] = int(arc_ids[np.argmax(candidates["score"].to_numpy(dtype=float))])
        states = [(arc, network.position(point, arc)) for arc in arc_ids]
        dt = timestamp - timestamps[i - 1] if i else 0.0
        movement = point - points[i - 1] if i and 0 < dt <= max_gap else np.zeros(2)
        if not np.isfinite(movement).all():
            movement = np.zeros(2)
        previous_dt = timestamps[i - 1] - timestamps[i - 2] if i >= 2 else 0.0
        if i >= 2 and 0 < previous_dt <= max_gap and np.isfinite(points[i - 2:i]).all():
            previous_movement = points[i - 1] - points[i - 2]
        else:
            previous_movement = np.zeros(2)
            previous_dt = 0.0
        if extended_features:
            emissions = -emission_cost(emission_features(
                network, point, movement, states, candidates["score"].to_numpy(dtype=float)
            ))
        backpointers = np.full(len(states), -1, dtype=int)
        bridges = {}
        next_values = emissions
        if history:
            dt = timestamp - timestamps[i - 1]
            if not 0 < dt <= max_gap:
                finish_fragment()
                row["break_reason"] = "timestamp_gap"
            else:
                displacement = np.linalg.norm(point - points[i - 1])
                transition, bridges = network.transitions(
                    previous_states, states, max_speed * dt + gps_slack, displacement, beta,
                    (dt, movement, previous_movement, previous_dt) if extended_features else None,
                )
                totals = values[:, None] + transition
                choices = totals.argmax(axis=0)
                best = totals[choices, np.arange(len(states))]
                if np.isfinite(best).any():
                    backpointers = choices
                    next_values = emissions + best
                else:
                    finish_fragment()
                    row["break_reason"] = "no_feasible_transition"
        elif i:
            row["break_reason"] = "after_unmatched_observation"
        else:
            row["break_reason"] = "start"
        history.append((i, arc_ids, backpointers, bridges))
        previous_states = states
        values = next_values - next_values.max()
    finish_fragment()
    result = pd.DataFrame(rows).astype(
        {name: "Int64" for name in ("classifier_arc_id", "viterbi_arc_id", "fragment_id")}
    )
    return result, fragments
