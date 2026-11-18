
from pathlib import Path

from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="PathSnap API",
    description="GPS Map Matching Backend",
    version="1.0.0",
)

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Student evaluation trips only
STUDENT_TEST_DIR = PROJECT_ROOT / "data" / "student_test"


@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "service": "PathSnap Backend",
    }


@app.get("/api/v1/trips")
def get_trips():
    if not STUDENT_TEST_DIR.exists():
        raise HTTPException(
            status_code=500,
            detail="Student evaluation dataset not found",
        )

    trip_ids = sorted(
        folder.name
        for folder in STUDENT_TEST_DIR.iterdir()
        if folder.is_dir()
        and folder.name.isdigit()
        and len(folder.name) == 8
        and (folder / f"{folder.name}.track").exists()
    )

    return {
        "trips": trip_ids,
        "count": len(trip_ids),
    }
