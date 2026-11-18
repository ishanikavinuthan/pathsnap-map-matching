"""Road-network-aware HMM/Viterbi decoder for PathSnap GPS trips."""

from __future__ import annotations

import heapq
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))
sys.path.insert(0, str(ROOT / "src" / "features"))
from candidates import CandidateGenerator, project_coordinates  # noqa: E402
from load_data import load_trip  # noqa: E402

FEATURES = [
    "distance_m", "heading_difference_deg", "gps_speed_mps", "time_gap_s",
    "gps_displacement_m", "candidate_rank",
]
MODEL_PATHS = [
    ROOT / "results/models/bayesian_optimization/optimized_random_forest.joblib",
    ROOT / "results/models/random_forest.joblib",
]


class DirectedRoadGraph:
    """Directed graph with bounded, cached Dijkstra searches."""

    def __init__(self, nodes: pd.DataFrame, arcs: pd.DataFrame, lat0: float,
                 cache_size: int = 64):
        node_xy = project_coordinates(nodes.longitude.to_numpy(), nodes.latitude.to_numpy(), lat0)
        self.starts = node_xy[arcs.from_node_id.to_numpy(dtype=int)]
        self.ends = node_xy[arcs.to_node_id.to_numpy(dtype=int)]
        self.lengths = np.linalg.norm(self.ends - self.starts, axis=1)
        self.sources = arcs.from_node_id.to_numpy(dtype=int)
        self.targets = arcs.to_node_id.to_numpy(dtype=int)
        self.adjacency: list[list[tuple[int, int, float]]] = [[] for _ in range(len(nodes))]
        for arc_id, (src, dst, length) in enumerate(zip(self.sources, self.targets, self.lengths)):
            self.adjacency[src].append((int(dst), arc_id, float(length)))
        self.cache_size = cache_size
        self.cache: OrderedDict[int, tuple[dict[int, float], dict[int, tuple[int, int]]]] = OrderedDict()

    def distances_from(self, source: int, cutoff: float) -> tuple[dict[int, float], dict[int, tuple[int, int]]]:
        """Return shortest directed distances and (parent node, arc) links."""
        if source in self.cache:
            distances, parents = self.cache.pop(source)
            self.cache[source] = (distances, parents)
            if max(distances.values(), default=0.0) >= cutoff:
                return distances, parents
        else:
            distances, parents = {}, {}
        start_distance = distances.get(source, 0.0)
        queue = [(start_distance, source)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances.get(node, 0.0 if node == source else math.inf):
                continue
            if distance > cutoff:
                break
            for neighbor, arc_id, weight in self.adjacency[node]:
                new_distance = distance + weight
                if new_distance <= cutoff and new_distance < distances.get(neighbor, math.inf):
                    distances[neighbor] = new_distance
                    parents[neighbor] = (node, arc_id)
                    heapq.heappush(queue, (new_distance, neighbor))
        self.cache[source] = (distances, parents)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return distances, parents

    def route_distance(self, previous: dict, following: dict, cutoff: float) -> float:
        prev_arc, next_arc = previous["arc_id"], following["arc_id"]
        prev_pos, next_pos = previous["fraction"], following["fraction"]
        if prev_arc == next_arc and next_pos >= prev_pos:
            return float((next_pos - prev_pos) * self.lengths[prev_arc])
        source, target = int(self.targets[prev_arc]), int(self.sources[next_arc])
        distances, _ = self.distances_from(source, cutoff)
        middle = distances.get(target)
        if middle is None:
            return math.inf
        return float((1.0 - prev_pos) * self.lengths[prev_arc] + middle + next_pos * self.lengths[next_arc])

    def path_arcs(self, source: int, target: int, cutoff: float) -> list[int] | None:
        distances, parents = self.distances_from(source, cutoff)
        if target not in distances:
            return None
        result = []
        node = target
        while node != source:
            if node not in parents:
                return None
            parent, arc_id = parents[node]
            result.append(arc_id)
            node = parent
        return list(reversed(result))


def _load_emission_model():
    for path in MODEL_PATHS:
        if not path.exists():
            continue
        model = joblib.load(path)
        names = getattr(model, "feature_names_in_", None)
        if names is None and hasattr(model, "named_steps"):
            names = getattr(model.named_steps.get("imputer"), "feature_names_in_", None)
        if names is not None and list(names) != FEATURES:
            raise ValueError(f"Model feature order mismatch at {path}: {list(names)}")
        return model, path
    return None, None


def _candidate_states(trip: dict, generator: CandidateGenerator, k: int) -> list[list[dict]]:
    track = trip["track"]
    xy = project_coordinates(track.longitude.to_numpy(), track.latitude.to_numpy(), generator.lat0)
    timestamps = track.timestamp.to_numpy(dtype=float)
    state_rows = []
    total = len(generator.arcs)
    if not 1 <= k <= total:
        raise ValueError("k must be between 1 and arc count")
    shortlist = min(max(k * 20, 100), total)
    for obs_id, point in enumerate(xy):
        candidate_ids = None
        candidate_distances = None
        current_shortlist = shortlist
        while True:
            mid_distances, indices = generator.tree.query(point, k=current_shortlist)
            indices = np.atleast_1d(indices).astype(int)
            mid_distances = np.atleast_1d(mid_distances)
            starts = generator.starts[indices]
            vectors = generator.ends[indices] - starts
            lengths_sq = np.einsum("ij,ij->i", vectors, vectors)
            numerators = np.einsum("ij,ij->i", point - starts, vectors)
            fractions = np.divide(numerators, lengths_sq, out=np.zeros_like(numerators), where=lengths_sq > 0)
            fractions = np.clip(fractions, 0.0, 1.0)
            projections = starts + fractions[:, None] * vectors
            distances = np.linalg.norm(projections - point, axis=1)
            order = np.lexsort((indices, distances))
            if current_shortlist == total or mid_distances[-1] - generator.max_half_length > distances[order[k-1]]:
                order = order[:k]
                candidate_ids = indices[order]
                candidate_distances = distances[order]
                break
            current_shortlist = min(current_shortlist * 2, total)

        if obs_id == 0:
            gap = displacement = speed = gps_heading = np.nan
        else:
            gap = timestamps[obs_id] - timestamps[obs_id - 1]
            displacement = float(np.linalg.norm(xy[obs_id] - xy[obs_id - 1]))
            speed = displacement / gap if gap > 0 else np.nan
            gps_heading = ((np.degrees(np.arctan2(*(xy[obs_id] - xy[obs_id - 1]))) + 360) % 360) if displacement > 0 else np.nan
        states = []
        for rank, (arc_id, distance) in enumerate(zip(candidate_ids, candidate_distances), start=1):
            arc_id = int(arc_id)
            vector = generator.ends[arc_id] - generator.starts[arc_id]
            length_sq = float(vector @ vector)
            fraction = 0.0 if length_sq == 0 else float(np.clip((point-generator.starts[arc_id]) @ vector / length_sq, 0, 1))
            road_heading = (np.degrees(np.arctan2(vector[0], vector[1])) + 360) % 360
            heading_difference = (abs((gps_heading - road_heading + 180) % 360 - 180)
                                  if np.isfinite(gps_heading) else np.nan)
            states.append({"trip_id": str(trip["trip_id"]), "observation_id": int(obs_id),
                           "arc_id": arc_id, "fraction": fraction,
                           "distance_m": float(distance), "heading_difference_deg": heading_difference,
                           "gps_speed_mps": speed, "time_gap_s": gap,
                           "gps_displacement_m": displacement, "candidate_rank": rank})
        state_rows.append(states)
    return state_rows


def _emissions(state_rows: list[list[dict]], model, temperature: float,
               distance_scale_m: float) -> list[np.ndarray]:
    # Batch the forest call across every candidate in the trip. Per-observation
    # predict_proba calls add substantial overhead for hundreds/thousands of
    # GPS points while producing the same independent candidate scores.
    output: list[np.ndarray] = []
    if model is not None:
        flat_states = [state for states in state_rows for state in states]
        X = pd.DataFrame([{key: state[key] for key in FEATURES} for state in flat_states], columns=FEATURES)
        class_idx = list(model.classes_).index(1)
        flat_probs = np.asarray(model.predict_proba(X)[:, class_idx], dtype=float)
    else:
        flat_probs = np.empty(sum(map(len, state_rows)), dtype=float)
        cursor = 0
        for states in state_rows:
            count = len(states)
            if count:
                scores = np.array([-float(state["distance_m"]) / distance_scale_m for state in states])
                scores -= np.max(scores)
                flat_probs[cursor:cursor+count] = np.exp(scores)
            cursor += count

    cursor = 0
    for states in state_rows:
        count = len(states)
        probs = flat_probs[cursor:cursor+count].copy()
        cursor += count
        if not count:
            output.append(np.array([], dtype=float))
            continue
        probs = np.clip(probs, 1e-9, 1.0) ** (1.0 / max(temperature, 1e-6))
        probs /= max(float(probs.sum()), 1e-300)
        output.append(np.log(np.clip(probs, 1e-300, 1.0)))
    return output


def decode_trip(trip: dict, model=None, k: int = 10, temperature: float = 1.0,
                distance_scale_m: float = 25.0, transition_weight: float = 1.0,
                speed_tolerance_mps: float = 20.0) -> tuple[pd.DataFrame, list[int], dict]:
    """Decode one trip, returning per-observation matches and reconstructed route."""
    started = time.perf_counter()
    generator = CandidateGenerator(trip["nodes"], trip["arcs"])
    graph = DirectedRoadGraph(trip["nodes"], trip["arcs"], generator.lat0)
    states = _candidate_states(trip, generator, k)
    emissions = _emissions(states, model, temperature, distance_scale_m)
    n = len(states)
    scores = [None] * n
    back = [None] * n
    track = trip["track"]
    xy = project_coordinates(track.longitude.to_numpy(), track.latitude.to_numpy(), generator.lat0)
    timestamps = track.timestamp.to_numpy(dtype=float)
    for i in range(n):
        if not states[i]:
            scores[i] = np.array([], dtype=float); back[i] = np.array([], dtype=int)
            continue
        if i == 0 or not states[i-1]:
            scores[i] = emissions[i].copy(); back[i] = np.full(len(states[i]), -1, dtype=int)
            continue
        gap = timestamps[i] - timestamps[i-1]
        displacement = float(np.linalg.norm(xy[i] - xy[i-1]))
        if not np.isfinite(gap) or gap <= 0:
            gap = 1.0
        allowed = max(250.0, displacement + speed_tolerance_mps * min(gap, 120.0) + 100.0)
        decay = 1.0 if gap <= 120 else max(0.15, 120.0 / gap)
        transition = np.full((len(states[i-1]), len(states[i])), -50.0)
        for p, previous in enumerate(states[i-1]):
            for q, following in enumerate(states[i]):
                road_distance = graph.route_distance(previous, following, allowed * 3)
                if math.isfinite(road_distance):
                    error = abs(road_distance - displacement)
                    sigma = max(20.0, displacement * 0.5, speed_tolerance_mps * min(gap, 30.0))
                    penalty = min(40.0, error / sigma)
                    speed = road_distance / gap
                    if speed > speed_tolerance_mps * 2:
                        penalty += min(30.0, (speed - speed_tolerance_mps * 2) / speed_tolerance_mps)
                    # U-turns are possible but receive an additional penalty.
                    if graph.sources[previous["arc_id"]] == graph.targets[following["arc_id"]] and previous["arc_id"] != following["arc_id"]:
                        penalty += 3.0
                else:
                    penalty = 18.0 + min(20.0, displacement / 100.0)
                transition[p, q] = -transition_weight * decay * penalty
        cumulative = scores[i-1][:, None] + transition
        back[i] = cumulative.argmax(axis=0)
        scores[i] = emissions[i] + cumulative.max(axis=0)

    chosen = [None] * n
    nonempty = [i for i, layer in enumerate(scores) if len(layer)]
    if nonempty:
        idx = int(scores[nonempty[-1]].argmax())
        for i in reversed(range(n)):
            if not len(scores[i]):
                chosen[i] = None
                continue
            chosen[i] = idx
            if i > 0 and len(back[i]) and idx >= 0:
                idx = int(back[i][idx])
            elif i > 0 and len(scores[i-1]):
                idx = int(scores[i-1].argmax())

    matched = []
    for i, state_idx in enumerate(chosen):
        state = states[i][state_idx] if state_idx is not None and states[i] else None
        matched.append({"trip_id": str(trip["trip_id"]), "observation_id": i,
                        "timestamp": float(timestamps[i]), "matched_arc_id": state["arc_id"] if state else None,
                        "candidate_count": len(states[i]), "candidate_covered": int(bool(states[i])),
                        "selected_distance_m": float(state["distance_m"]) if state else np.nan,
                        "selected_candidate_rank": int(state["candidate_rank"]) if state else np.nan,
                        "candidate_arc_ids": ";".join(str(item["arc_id"]) for item in states[i])})
    matches = pd.DataFrame(matched)

    # Route reconstruction preserves observation matches and fills directed connecting paths.
    route = []
    valid = [states[i][idx] for i, idx in enumerate(chosen) if idx is not None and states[i]]
    for pos, state in enumerate(valid):
        arc = state["arc_id"]
        if not route or route[-1] != arc:
            route.append(arc)
        if pos + 1 < len(valid):
            nxt = valid[pos+1]
            if arc != nxt["arc_id"]:
                mid = graph.path_arcs(int(graph.targets[arc]), int(graph.sources[nxt["arc_id"]]), 5000.0)
                if mid:
                    route.extend(a for a in mid if route[-1] != a)
    stats = {"trip_id": str(trip["trip_id"]), "observations": n,
             "candidate_covered": int(matches.candidate_covered.sum()),
             "candidate_coverage": float(matches.candidate_covered.mean()) if n else 0.0,
             "runtime_seconds": time.perf_counter() - started,
             "decoded_observations": int(matches.matched_arc_id.notna().sum())}
    return matches, route, stats


def evaluate_trip(trip: dict, matches: pd.DataFrame, aligned: pd.DataFrame) -> dict:
    """Score against existing route-derived provisional alignment labels."""
    reference = aligned[["observation_id", "aligned_arc_id"]]
    scored = matches.merge(reference, on="observation_id", how="left", validate="one_to_one")
    valid = scored.aligned_arc_id.notna()
    candidate_valid = valid & scored.apply(
        lambda row: str(int(row.aligned_arc_id)) in row.candidate_arc_ids.split(";") if row.candidate_arc_ids else False,
        axis=1,
    )
    covered = valid & scored.matched_arc_id.notna()
    denominator = int(candidate_valid.sum())
    all_n = len(scored)
    correct = int((scored.loc[candidate_valid, "matched_arc_id"] == scored.loc[candidate_valid, "aligned_arc_id"]).sum())
    all_correct = int((scored.loc[covered, "matched_arc_id"] == scored.loc[covered, "aligned_arc_id"]).sum())
    return {"trip_id": str(trip["trip_id"]), "valid_reference_observations": denominator,
            "all_gps_observations": all_n, "candidate_covered_observations": int(covered.sum()),
            "accuracy_valid_reference": correct / denominator if denominator else np.nan,
            "accuracy_all_gps": all_correct / all_n if all_n else np.nan,
            "reference_candidate_coverage": denominator / all_n if all_n else 0.0}


def run(root: Path = ROOT, trip_ids: list[str] | None = None, k: int = 10,
        temperature: float = 1.0) -> None:
    output = root / "results/hmm"
    output.mkdir(parents=True, exist_ok=True)
    model, model_path = _load_emission_model()
    split = json.loads((root / "results/models/trip_split.json").read_text())
    validation_ids = list(map(str, split["val_trips"]))
    selected_ids = list(map(str, trip_ids)) if trip_ids else validation_ids
    is_validation = set(selected_ids).issubset(set(validation_ids))
    base = root / "data/train"
    from align_route import align_trip
    all_matches, all_routes, accuracies, runtimes = [], [], [], []
    for idx, trip_id in enumerate(selected_ids, 1):
        trip = load_trip(base / trip_id)
        matches, route, stats = decode_trip(trip, model=model, k=k, temperature=temperature)
        stats["scoring_method"] = "random_forest" if model is not None else "geometric_distance_fallback"
        matches["scoring_method"] = stats["scoring_method"]
        aligned = align_trip(trip)
        metrics = evaluate_trip(trip, matches, aligned)
        metrics.update(stats)
        metrics["alignment_labels_are_provisional"] = True
        metrics["split"] = "validation" if is_validation else "training_smoke_or_development"
        all_matches.append(matches)
        all_routes.append({"trip_id": trip_id, "arc_ids": route})
        accuracies.append(metrics)
        runtimes.append(stats["runtime_seconds"])
        print(f"[{idx}/{len(selected_ids)}] trip {trip_id}: provisional Top-1={metrics['accuracy_valid_reference']:.3%}; {stats['runtime_seconds']:.2f}s", flush=True)
    pd.concat(all_matches, ignore_index=True).to_csv(output / "observation_matches.csv", index=False)
    pd.DataFrame(accuracies).to_csv(output / "per_trip_metrics.csv", index=False)
    (output / "reconstructed_routes.json").write_text(json.dumps(all_routes, indent=2))
    metric_df = pd.DataFrame(accuracies)
    summary = {
        "split": "validation" if is_validation else "training_smoke_or_development",
        "trip_count": len(selected_ids), "scoring_method": "random_forest" if model is not None else "geometric_distance_fallback",
        "model_path": str(model_path) if model_path else None,
        "reference_labels": "provisional route alignment; not independently verified",
        "accuracy_valid_reference": float(metric_df.accuracy_valid_reference.mean()),
        "accuracy_all_gps": float(metric_df.accuracy_all_gps.mean()),
        "candidate_coverage": float(metric_df.reference_candidate_coverage.mean()),
        "mean_runtime_seconds_per_trip": float(np.mean(runtimes)),
        "k_candidates": k, "emission_temperature": temperature,
    }
    (output / "aggregate_metrics.json").write_text(json.dumps(summary, indent=2))
    (output / "config.json").write_text(json.dumps({**summary, "transition_weight": 1.0,
        "speed_tolerance_mps": 20.0, "distance_fallback_scale_m": 25.0,
        "candidate_generator": "existing CandidateGenerator", "decode": "log-space Viterbi"}, indent=2))
    baseline_path = root / "results/models/bayesian_optimization/random_forest_optimization_comparison.csv"
    comparison = [{"model": "HMM + Viterbi", "top1_accuracy": summary["accuracy_valid_reference"],
                   "candidate_coverage": summary["candidate_coverage"], "runtime_seconds_per_trip": summary["mean_runtime_seconds_per_trip"]}]
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        comparison.extend(baseline.to_dict("records"))
    pd.DataFrame(comparison).to_csv(output / "baseline_vs_hmm.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trip", help="Run one trip (training smoke test)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--validation", action="store_true", help="Evaluate persisted held-out validation trips")
    args = parser.parse_args()
    ids = None if args.validation else ([args.trip] if args.trip else ["00000001"])
    run(k=args.k, temperature=args.temperature, trip_ids=ids)
