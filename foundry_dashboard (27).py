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

# -----------------------------
# Page / constants
# -----------------------------
st.set_page_config(
    page_title="Foundry Scrap Risk Dashboard - Actionable Insights",
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
# Load and Model Prep (Runs once at app startup)
# -----------------------------
# ... (content unchanged) ...

# -----------------------------
# Sidebar
# -----------------------------
# ... (content unchanged) ...

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

# Use ASCII-safe labels only
tabs = st.tabs(["Predict", "Validation", "History"])
st.write(f"✅ Tabs loaded: {len(tabs)}")

# -----------------------------
# TAB 1: Predict
# -----------------------------
with tabs[0]:
    st.write("✅ Predict Tab Loaded")
    # ... your Predict logic ...

# -----------------------------
# TAB 2: Validation
# -----------------------------
with tabs[1]:
    st.write("✅ Validation Tab Loaded")
    # ... your Validation logic ...

# -----------------------------
# TAB 3: History
# -----------------------------
if len(tabs) > 2:
    with tabs[2]:
        st.write("✅ History Tab Loaded")
        # ... your History tab logic ...
else:
    st.error("❌ History tab failed to render. Try removing emojis from title and tab labels.")
