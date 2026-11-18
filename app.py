from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="PathSnap - Model Evaluation",
    page_icon="🗺️",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "results" / "models"

API_URL = os.getenv(
    "PATHSNAP_API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

REQUEST_TIMEOUT = 120


# --------------------------------------------------
# LOAD MODEL EVALUATION RESULTS
# --------------------------------------------------

@st.cache_data
def load_metrics() -> pd.DataFrame:
    """Load evaluation metrics for the five ML models."""

    combined_file = MODELS_DIR / "all_five_models.csv"
    member3_file = MODELS_DIR / "member3_metrics.csv"
    member4_file = MODELS_DIR / "member4_metrics.csv"

    if combined_file.exists():
        return pd.read_csv(combined_file)

    dataframes = []

    if member3_file.exists():
        dataframes.append(pd.read_csv(member3_file))

    if member4_file.exists():
        dataframes.append(pd.read_csv(member4_file))

    if not dataframes:
        return pd.DataFrame()

    metrics = pd.concat(dataframes, ignore_index=True)

    if "model" in metrics.columns:
        metrics = metrics.drop_duplicates(
            subset=["model"],
            keep="last"
        )

    return metrics


# --------------------------------------------------
# FASTAPI HELPERS
# --------------------------------------------------

def check_backend():
    """Check backend health using GET /health."""

    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )

        response.raise_for_status()

        return True, response.json(), None

    except requests.RequestException as error:
        return False, None, str(error)

    except ValueError as error:
        return False, None, f"Invalid backend JSON: {error}"


def get_trips(dataset: str) -> list[str]:
    """Fetch trip IDs using GET /trips."""

    response = requests.get(
        f"{API_URL}/trips",
        params={"dataset": dataset},
        timeout=10
    )

    response.raise_for_status()

    data = response.json()
    trip_ids = data.get("trip_ids", [])

    if not isinstance(trip_ids, list):
        raise ValueError(
            "Backend response field 'trip_ids' is not a list."
        )

    return [str(trip_id) for trip_id in trip_ids]


def match_trip(
    trip_id: str,
    dataset: str,
    candidate_count: int,
    emission_temperature: float
) -> dict:
    """Call POST /match for HMM/Viterbi map matching."""

    payload = {
        "trip_id": trip_id,
        "dataset": dataset,
        "candidate_count": candidate_count,
        "emission_temperature": emission_temperature
    }

    response = requests.post(
        f"{API_URL}/match",
        json=payload,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    result = response.json()

    if not isinstance(result, dict):
        raise ValueError(
            "Backend returned an unexpected response format."
        )

    return result


# --------------------------------------------------
# MAIN DASHBOARD
# --------------------------------------------------

st.title("🗺️ PathSnap")

st.subheader(
    "GPS Map Matching and Route Reconstruction"
)

st.markdown(
    """
    This dashboard presents the evaluation results of five
    machine learning models used for GPS map matching.
    """
)

st.divider()

st.header("Model Evaluation Results")


if st.button("🔄 Refresh Evaluation Results"):

    load_metrics.clear()
    st.rerun()


metrics_df = load_metrics()


if not metrics_df.empty:

    required_columns = {
        "model",
        "accuracy",
        "f1",
        "top1_accuracy"
    }

    missing_columns = required_columns - set(metrics_df.columns)

    if missing_columns:

        st.warning(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    else:

        metrics_df = metrics_df.copy()

        numeric_columns = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "macro_f1",
            "top1_accuracy",
            "time_seconds"
        ]

        for column in numeric_columns:

            if column in metrics_df.columns:

                metrics_df[column] = pd.to_numeric(
                    metrics_df[column],
                    errors="coerce"
                )

        metrics_df = metrics_df.sort_values(
            by="f1",
            ascending=False
        )

        # --------------------------------------------------
        # SUMMARY METRICS
        # --------------------------------------------------

        best_accuracy = metrics_df.loc[
            metrics_df["accuracy"].idxmax()
        ]

        best_f1 = metrics_df.loc[
            metrics_df["f1"].idxmax()
        ]

        best_top1 = metrics_df.loc[
            metrics_df["top1_accuracy"].idxmax()
        ]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Best Accuracy",
            f"{best_accuracy['accuracy']:.2%}",
            str(best_accuracy["model"])
        )

        col2.metric(
            "Best F1 Score",
            f"{best_f1['f1']:.2%}",
            str(best_f1["model"])
        )

        col3.metric(
            "Best Top-1 Accuracy",
            f"{best_top1['top1_accuracy']:.2%}",
            str(best_top1["model"])
        )

        st.divider()

        # --------------------------------------------------
        # MODEL COMPARISON TABLE
        # --------------------------------------------------

        st.subheader("Model Comparison")

        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------
        # PERFORMANCE COMPARISON
        # --------------------------------------------------

        st.subheader("Performance Comparison")

        metric_options = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "macro_f1",
            "top1_accuracy"
        ]

        available_metrics = [
            metric
            for metric in metric_options
            if metric in metrics_df.columns
            and metrics_df[metric].notna().any()
        ]

        if available_metrics:

            selected_metric = st.selectbox(
                "Select evaluation metric",
                available_metrics
            )

            chart_data = metrics_df.set_index("model")[
                selected_metric
            ]

            st.bar_chart(chart_data)

        else:

            st.info(
                "No numeric performance metrics are available."
            )

        # --------------------------------------------------
        # EXECUTION TIME
        # --------------------------------------------------

        if "time_seconds" in metrics_df.columns:

            time_data = metrics_df.set_index("model")[
                "time_seconds"
            ].dropna()

            if not time_data.empty:

                st.subheader("Model Execution Time")

                st.bar_chart(time_data)

                st.caption(
                    "Execution time is measured in seconds."
                )

        # --------------------------------------------------
        # DOWNLOAD RESULTS
        # --------------------------------------------------

        st.divider()

        st.subheader("Download Evaluation Results")

        csv_data = metrics_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Model Metrics CSV",
            data=csv_data,
            file_name="pathsnap_model_evaluation.csv",
            mime="text/csv"
        )

else:

    st.warning(
        "Model evaluation files were not found. "
        "Please check the results/models folder."
    )


# --------------------------------------------------
# EVALUATION NOTES
# --------------------------------------------------

st.divider()

st.subheader("Evaluation Notes")

st.markdown(
    """
    - Five machine learning models are compared when their
      evaluation files are available.
    - Accuracy, precision, recall, F1-score, macro-F1
      and Top-1 accuracy are used for evaluation.
    - Training labels are derived from route alignment.
    - These metrics measure agreement with alignment-derived
      labels, not independently verified real-world
      map-matching accuracy.
    - Top-1 accuracy measures whether the highest-scoring
      candidate matches the reference label for an observation.
    - Candidate coverage and model accuracy are different measures.
    """
)


# --------------------------------------------------
# FASTAPI BACKEND CONNECTION
# --------------------------------------------------

backend_ok, health_data, health_error = check_backend()


# --------------------------------------------------
# GPS TRIP MAP MATCHING
# --------------------------------------------------

st.divider()

st.header("GPS Trip Map Matching")

st.markdown(
    """
    Select a dataset and GPS trip, configure the candidate
    generation parameters, and run map matching using the
    FastAPI backend with HMM and Viterbi decoding.

    The backend uses GPS track, nodes and arcs files for
    inference. It does not read the reference route file.
    """
)


if backend_ok:

    st.success("FastAPI backend connected")

    if health_data:

        scoring_method = health_data.get(
            "emission_method",
            "Not reported"
        )

        st.caption(
            f"Emission scoring method: {scoring_method}"
        )

    # --------------------------------------------------
    # DATASET SELECTION
    # --------------------------------------------------

    dataset_choice = st.selectbox(
        "Select dataset",
        options=[
            "student_test",
            "train"
        ],
        index=0,
        format_func=lambda value: (
            "Student Test (unseen evaluation trips)"
            if value == "student_test"
            else "Training"
        )
    )

    try:

        trips = get_trips(dataset_choice)

        if trips:

            selected_trip = st.selectbox(
                "Select GPS trip",
                trips,
                key=f"trip_select_{dataset_choice}"
            )

            col1, col2 = st.columns(2)

            # --------------------------------------------------
            # CANDIDATE COUNT
            # --------------------------------------------------

            with col1:

                candidate_count = st.slider(
                    "Candidate road segments (k)",
                    min_value=1,
                    max_value=50,
                    value=10,
                    step=1
                )

            # --------------------------------------------------
            # EMISSION TEMPERATURE
            # --------------------------------------------------

            with col2:

                emission_temperature = st.number_input(
                    "Emission temperature",
                    min_value=0.01,
                    max_value=20.0,
                    value=1.0,
                    step=0.1
                )

            st.info(
                f"Selected trip: {selected_trip}"
            )

            # --------------------------------------------------
            # RUN MAP MATCHING
            # --------------------------------------------------

            if st.button(
                "🚀 Run Map Matching",
                type="primary"
            ):

                with st.spinner(
                    "Matching GPS observations and reconstructing route..."
                ):

                    try:

                        result = match_trip(
                            trip_id=selected_trip,
                            dataset=dataset_choice,
                            candidate_count=candidate_count,
                            emission_temperature=emission_temperature
                        )

                        st.session_state[
                            "pathsnap_match_result"
                        ] = result

                        st.session_state[
                            "pathsnap_match_trip"
                        ] = selected_trip

                        st.session_state[
                            "pathsnap_match_dataset"
                        ] = dataset_choice

                    except requests.HTTPError as error:

                        detail = (
                            error.response.text
                            if error.response is not None
                            else str(error)
                        )

                        st.error(
                            f"Map-matching request failed: {detail}"
                        )

                    except (
                        requests.RequestException,
                        ValueError
                    ) as error:

                        st.error(
                            f"Could not complete map matching: {error}"
                        )

            # --------------------------------------------------
            # DISPLAY MATCHING RESULTS
            # --------------------------------------------------

            result = st.session_state.get(
                "pathsnap_match_result"
            )

            result_trip = st.session_state.get(
                "pathsnap_match_trip"
            )

            result_dataset = st.session_state.get(
                "pathsnap_match_dataset"
            )

            if (
                result
                and result_trip == selected_trip
                and result_dataset == dataset_choice
            ):

                st.divider()

                st.subheader(
                    f"Map Matching Result — {selected_trip}"
                )

                # --------------------------------------------------
                # MATCHING METRICS
                # --------------------------------------------------

                metrics = result.get(
                    "metrics",
                    {}
                )

                if metrics:

                    st.markdown(
                        "**Backend Matching Metrics**"
                    )

                    st.json(metrics)

                # --------------------------------------------------
                # RECONSTRUCTED ROUTE
                # --------------------------------------------------

                route_arc_ids = result.get(
                    "reconstructed_route_arc_ids",
                    []
                )

                st.markdown(
                    "**Reconstructed Directed Route**"
                )

                st.write(
                    f"Number of route arcs: {len(route_arc_ids)}"
                )

                if route_arc_ids:

                    st.code(
                        ", ".join(
                            map(str, route_arc_ids)
                        )
                    )

                else:

                    st.info(
                        "No route arcs were returned."
                    )

                # --------------------------------------------------
                # OBSERVATION MATCHES
                # --------------------------------------------------

                observation_matches = result.get(
                    "observation_matches",
                    []
                )

                st.markdown(
                    "**Observation-Level Matches**"
                )

                st.write(
                    f"Number of matched observations: "
                    f"{len(observation_matches)}"
                )

                if observation_matches:

                    matches_df = pd.DataFrame(
                        observation_matches
                    )

                    st.dataframe(
                        matches_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    matches_csv = matches_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Observation Matches CSV",
                        data=matches_csv,
                        file_name=(
                            f"pathsnap_{selected_trip}_matches.csv"
                        ),
                        mime="text/csv"
                    )

                else:

                    st.info(
                        "No observation-level matches were returned."
                    )

                # --------------------------------------------------
                # REQUEST CONFIGURATION
                # --------------------------------------------------

                st.markdown(
                    "**Request Configuration**"
                )

                st.json(
                    result.get("config", {})
                )

        else:

            st.warning(
                f"No trips were returned for dataset "
                f"'{dataset_choice}'."
            )

    except requests.HTTPError as error:

        detail = (
            error.response.text
            if error.response is not None
            else str(error)
        )

        st.error(
            f"Could not load trips from backend: {detail}"
        )

    except (
        requests.RequestException,
        ValueError
    ) as error:

        st.error(
            f"Could not load evaluation trips: {error}"
        )

else:

    st.warning(
        "FastAPI backend is not running or is unreachable. "
        "The model evaluation dashboard is still available."
    )

    if health_error:

        st.caption(
            f"Connection detail: {health_error}"
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "PathSnap | GPS Map Matching and Route Reconstruction"
)