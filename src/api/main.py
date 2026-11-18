"""FastAPI service exposing PathSnap trip listing and HMM map matching."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "hmm"))
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))

from load_data import read_raw_file  # noqa: E402
from viterbi import _load_emission_model, decode_trip  # noqa: E402

app = FastAPI(
    title="PathSnap Map Matching API",
    description="Road-network-aware GPS map matching with HMM and Viterbi decoding.",
    version="1.0.0",
)

DatasetName = Literal["train", "student_test"]


class MatchRequest(BaseModel):
    trip_id: str = Field(pattern=r"^\d{8}$", description="Eight-digit PathSnap trip ID")
    dataset: DatasetName = "train"
    candidate_count: int = Field(default=10, ge=1, le=50)
    emission_temperature: float = Field(default=1.0, gt=0, le=20)


def _dataset_dir(dataset: DatasetName) -> Path:
    return ROOT / "data" / dataset


def _load_inference_trip(dataset: DatasetName, trip_id: str) -> dict:
    """Load only inference inputs. In particular, never read the .route file."""
    trip_dir = _dataset_dir(dataset) / trip_id
    if not trip_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} was not found in {dataset}")
    trip = {"trip_id": trip_id}
    for file_type in ("track", "nodes", "arcs"):
        path = trip_dir / f"{trip_id}.{file_type}"
        if not path.is_file():
            raise HTTPException(status_code=422, detail=f"Trip is missing its {file_type} file")
        trip[file_type] = read_raw_file(path, file_type)
    trip["nodes"].index.name = "node_id"
    trip["arcs"].index.name = "arc_id"
    return trip


@lru_cache(maxsize=1)
def _emission_model():
    return _load_emission_model()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@app.get("/health")
def health():
    model, model_path = _emission_model()
    return {
        "status": "ok",
        "emission_method": "random_forest" if model is not None else "geometric_distance_fallback",
        "model_path": str(model_path) if model_path else None,
    }


@app.get("/trips")
def list_trips(dataset: DatasetName = "train"):
    """List trips available for inference; labels/routes are not opened."""
    base = _dataset_dir(dataset)
    if not base.is_dir():
        raise HTTPException(status_code=404, detail=f"Dataset {dataset} was not found")
    ids = sorted(path.name for path in base.iterdir()
                 if path.is_dir() and (path / f"{path.name}.track").is_file())
    return {"dataset": dataset, "count": len(ids), "trip_ids": ids}


@app.post("/match")
def match_trip(request: MatchRequest):
    """Return observation matches and the reconstructed directed arc route."""
    trip = _load_inference_trip(request.dataset, request.trip_id)
    model, model_path = _emission_model()
    try:
        matches, route, stats = decode_trip(
            trip,
            model=model,
            k=request.candidate_count,
            temperature=request.emission_temperature,
        )
    except (ValueError, IndexError, KeyError) as error:
        raise HTTPException(status_code=422, detail=f"Could not match trip: {error}") from error

    matches = matches.drop(columns=["candidate_arc_ids"], errors="ignore")
    stats["scoring_method"] = "random_forest" if model is not None else "geometric_distance_fallback"
    stats["model_path"] = str(model_path) if model_path else None
    stats["dataset"] = request.dataset
    stats["ground_truth_used"] = False
    return _json_safe({
        "trip_id": request.trip_id,
        "observation_matches": matches.to_dict(orient="records"),
        "reconstructed_route_arc_ids": route,
        "metrics": stats,
        "config": {
            "candidate_count": request.candidate_count,
            "emission_temperature": request.emission_temperature,
            "decoder": "log-space Viterbi",
        },
    })
