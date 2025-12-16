import streamlit as st
import numpy as np
import joblib
import math

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Process → Property Prediction",
    layout="wide"
)

# -------------------------------
# Custom CSS (Dashboard Styling)
# -------------------------------
st.markdown("""
<style>
/* Background */
.stApp {
    background-color: #f4f6f9;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1f2937;
}
[data-testid="stSidebar"] * {
    color: #ffffff;
}

/* Title */
h1 {
    color: #111827;
    font-weight: 700;
}

/* Subheader */
h3 {
    color: #374151;
}

/* Buttons */
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 0.7em 1.5em;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #1e40af;
}

/* Metric cards */
.metric-card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    text-align: center;
}
.metric-title {
    font-size: 18px;
    color: #6b7280;
}
.metric-value {
    font-size: 32px;
    font-weight: 700;
    color: #111827;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load trained models
# -------------------------------
hardness_model = joblib.load("microhardness_model.pkl")
uts_model = joblib.load("uts_model.pkl")

# -------------------------------
# Header
# -------------------------------
st.title("Process → Property Prediction Dashboard")
st.write(
    "A physics-informed machine learning tool for predicting "
    "**Vickers Microhardness** and **Ultimate Tensile Strength**."
)

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Input Parameters")

pw_choice = st.sidebar.selectbox(
    "Process Type",
    ("NW (0)", "PW (1)")
)
pw_nw = 1 if pw_choice == "PW (1)" else 0

current = st.sidebar.number_input(
    "Current (A)",
    min_value=100.0,
    step=1.0
)

scan_speed = st.sidebar.number_input(
    "Scan Speed (mm/s)",
    min_value=0.1,
    step=0.1
)

if scan_speed < 0.2:
    st.sidebar.warning("Very low scan speed may reduce prediction reliability.")

# -------------------------------
# Prediction
# -------------------------------
if st.sidebar.button("Predict Properties"):

    # Feature engineering
    heat_input = (14 * current) / scan_speed
    log_heat = math.log(heat_input)

    input_data = np.array([[
        pw_nw,
        current,
        scan_speed,
        heat_input,
        log_heat
    ]])

    hardness_pred = hardness_model.predict(input_data)[0]
    uts_pred = uts_model.predict(input_data)[0]

    # -------------------------------
    # Results Section
    # -------------------------------
    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Vickers Microhardness</div>
            <div class="metric-value">{hardness_pred:.2f} HV</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Ultimate Tensile Strength</div>
            <div class="metric-value">{uts_pred:.2f} MPa</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.caption(
        "Heat input calculated using an average voltage of 14 V. "
        "Predictions are generated using gradient boosting regression models."
    )

else:
    st.info("Enter process parameters in the sidebar and click **Predict Properties**.")
