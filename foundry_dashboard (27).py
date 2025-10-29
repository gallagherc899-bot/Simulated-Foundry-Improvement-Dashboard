# streamlit_app.py
# Run with: streamlit run streamlit_app.py

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

# Page setup
st.set_page_config(
    page_title="Foundry Scrap Risk Dashboard",
    layout="wide"
)

# Debug version print
st.sidebar.write("Streamlit version:", st.__version__)

# Global constants
RANDOM_STATE = 42
DEFAULT_ESTIMATORS = 180
MIN_SAMPLES_LEAF = 2
S_GRID = np.linspace(0.6, 1.2, 13)
GAMMA_GRID = np.linspace(0.5, 1.2, 15)
USE_RATE_COLS_PERMANENT = True

# Sidebar controls
st.sidebar.header("Data Source")
csv_path = st.sidebar.text_input("Path to CSV", value="anonymized_parts.csv")

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

st.title("Foundry Scrap Risk Dashboard")
st.caption("Includes Predict, Validation, and Historical analysis tabs.")

# Initialize session state
df_empty = df.empty
if "validation_results" not in st.session_state:
    st.session_state.validation_results = {
        "is_complete": False,
        "s_median": 1.0,
        "gamma_median": 0.5,
        "n_estimators": DEFAULT_ESTIMATORS,
        "results_df": pd.DataFrame(),
    }

# Tabs setup (emoji-free for compatibility)
tabs = st.tabs(["Predict", "Validation", "History"])

# Debug check for tab rendering
st.write(f"Tabs loaded: {len(tabs)}")

with tabs[0]:
    st.subheader("Predict Tab")
    st.write("Predict Tab Loaded")

with tabs[1]:
    st.subheader("Validation Tab")
    st.write("Validation Tab Loaded")

with tabs[2]:
    st.subheader("History Tab")
    st.write("History Tab Loaded")
    
    if df_empty:
        st.warning("No data loaded to display history.")
    else:
        part_ids = sorted(df["part_id"].unique())
        selected_part = st.selectbox("Select Part ID", part_ids)
        filtered_df = df[df["part_id"] == selected_part]
        st.dataframe(filtered_df)

        st.write("Historical data available:", len(filtered_df))
