# streamlit_scrap_dashboard.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import numpy as np
import pandas as pd
import streamlit as st

from dateutil.relativedelta import relativedelta
from scipy.stats import wilcoxon

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, accuracy_score

# -----------------------------
# Page / constants
# -----------------------------
st.set_page_config(
    page_title="Simulated Foundry Scrap Risk Dashboard - Actionable Insights",
    layout="wide"
)

RANDOM_STATE = 42
DEFAULT_ESTIMATORS = 180
MIN_SAMPLES_LEAF = 2

S_GRID = np.linspace(0.6, 1.2, 13)
GAMMA_GRID = np.linspace(0.5, 1.2, 15)
TOP_K_PARETO = 8 
USE_RATE_COLS_PERMANENT = True 

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Data Source")
csv_path = st.sidebar.text_input("Path to CSV", value="anonymized_parts_simulated_success.csv")

st.sidebar.header("Risk Definition")
thr_label = st.sidebar.slider("Scrap % Threshold", 1.0, 15.0, 6.50, 0.5)

st.sidebar.header("Validation Settings")
run_validation = st.sidebar.checkbox("Run rolling validation", value=True)

# Validate CSV
if not os.path.exists(csv_path):
    st.error("CSV not found at path: " + csv_path)
    st.stop()

# Load Data
@st.cache_data(show_spinner=False)
def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace("(", "", regex=False)
                  .str.replace(")", "", regex=False)
                  .str.replace("#", "num", regex=False)
    )
    needed = ["part_id", "week_ending", "scrap%", "order_quantity", "piece_weight_lbs"]
    df = df.dropna(subset=needed).copy()
    df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df = df.dropna(subset=["week_ending"])
    df = df.sort_values("week_ending").reset_index(drop=True)
    return df

# Load the dataset
df = load_and_clean(csv_path)

# -----------------------------
# Main Title and Tabs
# -----------------------------
st.title("Foundry Scrap Risk Dashboard - Actionable Insights")
st.caption("RF + calibrated probs, Validation-tuned (s, gamma), exceedance scaling, MTTFscrap & reliability")

if "validation_results" not in st.session_state:
    st.session_state.validation_results = {
        "is_complete": False,
        "s_median": 1.0,
        "gamma_median": 0.5,
        "n_estimators": DEFAULT_ESTIMATORS,
        "results_df": pd.DataFrame(),
    }

tabs = st.tabs(["Predict", "Validation"])

# -----------------------------
# TAB 1: Predict
# -----------------------------
with tabs[0]:
    st.subheader("Predict")
    part_ids = sorted(df["part_id"].unique())
    selected_part = st.selectbox("Select Part ID to Predict", part_ids)

    st.info(f"Prediction for Part ID: {selected_part} — Placeholder Model")

    # Historical section
    st.markdown("---")
    st.subheader("Historical Data for Selected Part")

    filtered_df = df[df["part_id"] == selected_part]
    if filtered_df.empty:
        st.warning("No historical data found for the selected part.")
    else:
        st.dataframe(filtered_df)
        st.write("Historical records:", len(filtered_df))

# -----------------------------
# TAB 2: Validation
# -----------------------------
with tabs[1]:
    st.subheader("Validation")
    st.write("Validation logic placeholder...")
