import streamlit as st
import numpy as np
import joblib
import math

# -------------------------------
# Load trained models
# -------------------------------
hardness_model = joblib.load("microhardness_model.pkl")
uts_model = joblib.load("uts_model.pkl")

# -------------------------------
# App Title
# -------------------------------
st.title("Process Parameter → Property Prediction")
st.subheader("Microhardness & Ultimate Tensile Strength")

st.write(
    "This tool predicts **Vickers Microhardness (HV)** and "
    "**Ultimate Tensile Strength (MPa)** based on process parameters."
)

# -------------------------------
# User Inputs
# -------------------------------
pw_choice = st.selectbox(
    "Process Type (PW / NW)",
    ("NW (0)", "PW (1)")
)

pw_nw = 1 if pw_choice == "PW (1)" else 0

current = st.number_input(
    "Current (A)",
    min_value=0.1,
    step=0.1
)

scan_speed = st.number_input(
    "Scan Speed (mm/s)",
    min_value=0.1,
    step=1.0
)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict Properties"):

    # ---- Feature engineering (MUST match training) ----
    heat_input = (14 * current) / scan_speed
    log_heat = math.log(heat_input)

    input_data = np.array([[
        pw_nw,
        current,
        scan_speed,
        heat_input,
        log_heat
    ]])

    # ---- Predictions ----
    hardness_pred = hardness_model.predict(input_data)[0]
    uts_pred = uts_model.predict(input_data)[0]

    # -------------------------------
    # Display Results
    # -------------------------------
    st.success("Prediction Successful")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Vickers Microhardness",
            value=f"{hardness_pred:.2f} HV"
        )

    with col2:
        st.metric(
            label="Ultimate Tensile Strength",
            value=f"{uts_pred:.2f} MPa"
        )

    # -------------------------------
    # Optional explanation
    # -------------------------------
    st.caption(
        "Heat input calculated using an average voltage of 14 V. "
        "Predictions are based on data-driven regression models."
    )
