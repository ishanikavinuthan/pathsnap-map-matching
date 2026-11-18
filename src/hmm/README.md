# PathSnap HMM map matching

Run one training-trip smoke example:

```bash
.venv/bin/python src/hmm/viterbi.py --trip 00000002
```

Run the held-out trips listed in `results/models/trip_split.json`:

```bash
.venv/bin/python src/hmm/viterbi.py --validation
```

The decoder generates up to 10 candidate directed arcs at each GPS sample,
scores them with the optimized Random Forest when its artifact is present,
then uses a distance-based emission fallback otherwise. It combines emission
scores with directed shortest-path transition distances and runs Viterbi in
log space. Candidate projection fractions account for the positions along the
arcs. The `.route` file is accessed only after decoding to score against the
project's existing route alignment; those labels are provisional.

Outputs are written to `results/hmm/`: per-observation matches, reconstructed
routes, per-trip metrics, aggregate metrics, a configuration record, and a
baseline comparison. `accuracy_valid_reference` only counts observations for
which the aligned reference arc appears in the candidate set.
`accuracy_all_gps` counts every GPS sample and treats uncovered samples as
incorrect. The project's route alignment is not independently verified ground
truth, so these metrics should be described as provisional.

The current repository has no FastAPI application. A backend can call
`load_trip()` and `decode_trip()` with a trip's track and network tables, then
return the observation matches and reconstructed arc list as response data.
