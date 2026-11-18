
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
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "results" / "models"

API_URL = "http://127.0.0.1:8000"


# --------------------------------------------------
# LOAD MODEL EVALUATION RESULTS
# --------------------------------------------------

def load_metrics():
    """
    Load evaluation metrics for all five ML models.
    """

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

    if dataframes:
        return pd.concat(dataframes, ignore_index=True)

    return pd.DataFrame()


# --------------------------------------------------
# MAIN DASHBOARD
# --------------------------------------------------

st.title("🗺️ PathSnap")
st.subheader("GPS Map Matching and Route Reconstruction")

st.markdown(
    """
    This dashboard presents the evaluation results of five
    machine learning models used for GPS map matching.
    """
)

st.divider()

st.header("Model Evaluation Results")

if st.button("🔄 Refresh Evaluation Results"):
    st.rerun()

metrics_df = load_metrics()

if not metrics_df.empty:

    # Sort models by F1 score
    if "f1" in metrics_df.columns:
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
        metric for metric in metric_options
        if metric in metrics_df.columns
    ]

    selected_metric = st.selectbox(
        "Select evaluation metric",
        available_metrics
    )

    chart_data = metrics_df.set_index("model")[
        selected_metric
    ]

    st.bar_chart(chart_data)

    # --------------------------------------------------
    # EXECUTION TIME
    # --------------------------------------------------

    if "time_seconds" in metrics_df.columns:

        st.subheader("Model Execution Time")

        time_data = metrics_df.set_index("model")[
            "time_seconds"
        ]

        st.bar_chart(time_data)

        st.caption(
            "Execution time is measured in seconds."
        )

    # --------------------------------------------------
    # DOWNLOAD RESULTS
    # --------------------------------------------------

    st.divider()

    st.subheader("Download Evaluation Results")

    csv_data = metrics_df.to_csv(index=False).encode("utf-8")

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
    - Five machine learning models are compared.
    - Accuracy, precision, recall, F1-score, macro-F1
      and Top-1 accuracy are used for evaluation.
    - The training labels are derived from route alignment.
    - Therefore, these metrics measure agreement with
      alignment-derived labels, not independently verified
      real-world map-matching accuracy.
    - Top-1 accuracy measures whether the highest-scoring
      candidate matches the reference label for an observation.
    - Candidate coverage and model accuracy are different measures.
    """
)


# --------------------------------------------------
# FASTAPI BACKEND CONNECTION
# --------------------------------------------------

def check_backend():
    try:
        response = requests.get(
            f"{API_URL}/api/v1/health",
            timeout=5
        )

        response.raise_for_status()
        return True

    except requests.RequestException:
        return False


def get_evaluation_trips():
    response = requests.get(
        f"{API_URL}/api/v1/trips",
        timeout=10
    )

    response.raise_for_status()

    data = response.json()
    return data.get("trips", [])


# --------------------------------------------------
# STUDENT TRIP EVALUATION
# --------------------------------------------------

st.divider()

st.header("Student Trip Evaluation")

st.markdown(
    """
    Connect to the FastAPI backend and select a student
    evaluation trip from the available dataset.
    """
)

if check_backend():

    st.success("FastAPI backend connected")

    try:
        trips = get_evaluation_trips()

        if trips:

            selected_trip = st.selectbox(
                "Select evaluation trip",
                trips
            )

            st.info(f"Selected trip: {selected_trip}")

            st.caption(
                "Trip prediction will be available when "
                "the backend evaluation endpoint is ready."
            )

        else:
            st.warning(
                "No evaluation trips were returned by the backend."
            )

    except requests.RequestException as error:

        st.error(
            f"Could not fetch evaluation trips: {error}"
        )

else:

    st.warning(
        "FastAPI backend is not running or is unreachable. "
        "The model evaluation dashboard is still available."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "PathSnap | GPS Map Matching and Route Reconstruction"
)
