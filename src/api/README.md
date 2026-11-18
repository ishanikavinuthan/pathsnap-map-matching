# PathSnap FastAPI backend

Install the HTTP server dependencies in the project environment:

```bash
.venv/bin/pip install -r requirements-api.txt
```

Start the API from the repository root:

```bash
.venv/bin/uvicorn src.api.main:app --reload
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Endpoints

- `GET /health` reports whether the Random Forest artifact was loaded or the
  geometric emission fallback is active.
- `GET /trips?dataset=train` lists inference-ready trip IDs. `dataset` may be
  `train` or `student_test`.
- `POST /match` runs HMM/Viterbi and returns one match per GPS observation,
  a reconstructed directed arc route, and runtime/coverage metadata.

Example request:

```json
{
  "trip_id": "00000002",
  "dataset": "train",
  "candidate_count": 10,
  "emission_temperature": 1.0
}
```

The inference loader reads only `.track`, `.nodes`, and `.arcs`; it does not
open `.route` or use labels. The optimized Random Forest is preferred when its
artifact exists, then the baseline artifact, then geometric distance scoring.
The API does not retrain or tune models. `student_test` is available only for
inference and is not used for tuning or evaluation by this service.
