"""Decode one PathSnap trip from candidate scores or a compatible fitted model."""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from decoder import RoadNetwork, decode

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src" / folder) for folder in ("features", "preprocessing")]
from align_route import align_trip
from candidates import project_coordinates
from features import extract_features
from load_data import read_raw_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trip", type=Path, help="Trip directory, such as student_test/00000007")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scores", type=Path, help="CSV with observation_id, arc_id, score")
    source.add_argument("--model", type=Path, help="Trusted fitted joblib pipeline")
    source.add_argument("--geometry-demo", action="store_true",
                        help="Use distance-based scores to test the decoder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, help="Number of candidates generated per observation")
    parser.add_argument("--extended-features", action="store_true",
                        help="Use six emission and twelve transition features")
    parser.add_argument("--beta", type=float, default=25.0)
    parser.add_argument("--max-speed", type=float, default=55.0)
    parser.add_argument("--max-gap", type=float, default=30.0)
    parser.add_argument("--gps-slack", type=float, default=30.0)
    parser.add_argument("--evaluate", action="store_true",
                        help="Compare to aligned reference only after decoding")
    args = parser.parse_args()
    if args.k is not None and args.k < 1:
        parser.error("--k must be positive.")

    trip_id = args.trip.name
    trip = {"trip_id": trip_id}
    for kind in ("track", "nodes", "arcs"):
        trip[kind] = read_raw_file(args.trip / f"{trip_id}.{kind}", kind)
    track, nodes, arcs = (trip[kind] for kind in ("track", "nodes", "arcs"))
    lat0 = nodes["latitude"].mean()
    node_xy = project_coordinates(nodes["longitude"], nodes["latitude"], lat0)
    point_xy = project_coordinates(track["longitude"], track["latitude"], lat0)
    network = RoadNetwork(node_xy, arcs.to_numpy(dtype=int))

    if args.scores:
        scored = pd.read_csv(args.scores, dtype={"trip_id": str})
        if "trip_id" in scored:
            trip_numbers = pd.to_numeric(scored["trip_id"])
            scored = scored.loc[trip_numbers == int(trip_id)]
            if scored.empty:
                parser.error(f"The scores CSV has no rows for trip {trip_id}.")
        scored = scored.rename(columns={"candidate_arc_id": "arc_id"})
    else:
        model = joblib.load(args.model) if args.model else None
        k = args.k or getattr(model, "pathsnap_candidate_count_", 5)
        valid = np.isfinite(track.to_numpy()).all(axis=1)
        indices = np.flatnonzero(valid)
        clean_trip = {**trip, "track": track.loc[valid].reset_index(drop=True)}
        if len(indices):
            scored = extract_features(clean_trip, k=k)
            scored["observation_id"] = indices[scored["observation_id"]]
        else:
            scored = pd.DataFrame(columns=["observation_id", "arc_id", "distance_m"])
        if model is not None and not scored.empty:
            columns = getattr(model, "feature_names_in_", None)
            if columns is None or not set(columns).issubset(scored.columns):
                parser.error("Model feature schema differs. Export candidate probabilities with your model's original feature generator and pass --scores.")
            positive = list(model.classes_).index(1)
            scored["score"] = model.predict_proba(scored[list(columns)])[:, positive]
        else:
            scored["score"] = np.exp(-scored["distance_m"].astype(float) / 20)

    result, fragments = decode(
        point_xy, track["timestamp"], network, scored, beta=args.beta,
        max_speed=args.max_speed, max_gap=args.max_gap, gps_slack=args.gps_slack,
        extended_features=args.extended_features,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    result = pd.concat([track, result], axis=1)
    result.to_csv(args.out / "matched_observations.csv", index=False)
    features = []
    for fragment_id, route in enumerate(fragments):
        node_ids = [network.arcs[route[0], 0], *network.arcs[route, 1]]
        coordinates = nodes.iloc[node_ids][["longitude", "latitude"]].values.tolist()
        features.append({
            "type": "Feature",
            "properties": {"fragment_id": fragment_id, "arc_ids": route},
            "geometry": {"type": "LineString", "coordinates": coordinates},
        })
    (args.out / "route.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
    )

    summary = {"trip_id": trip_id, "observations": len(result),
               "matched": int(result["viterbi_arc_id"].count()), "fragments": len(fragments)}
    if args.extended_features:
        summary["hmm_scoring_features"] = 18
    if args.evaluate:
        trip["route"] = read_raw_file(args.trip / f"{trip_id}.route", "route")
        reference = align_trip(trip)["aligned_arc_id"]
        for column in ("classifier_arc_id", "viterbi_arc_id"):
            summary[f"{column}_accuracy_vs_aligned_reference"] = float(
                result[column].eq(reference).fillna(False).mean()
            )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
