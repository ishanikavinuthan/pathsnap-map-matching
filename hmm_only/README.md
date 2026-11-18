# PathSnap standalone HMM

The standalone run generates ten candidate directed roads per GPS observation
and uses a score of exp(-distance_m / 20) in place of the classifier
probability. The other emission and transition features are unchanged. The
decoder uses a 10 m tolerance for backward GPS jitter on the same road.

Run from this folder in PowerShell with numpy, pandas, scipy and joblib
installed. Set the trip path to your extracted student-test directory.

```powershell
$trip = "C:\path\to\Student folder\student_test\00000007"
python .\route_reconstruction\reconstruct.py "$trip" --geometry-demo --k 10 --extended-features --out ".\results\hmm_only_trip_07" --evaluate
```

The summary field named classifier_arc_id_accuracy_vs_aligned_reference
represents nearest-road accuracy in geometry mode. The Viterbi field
represents standalone HMM accuracy. The --evaluate option reads the .route
file after decoding to compare against the aligned reference. Omit it for
inference without reference data.
