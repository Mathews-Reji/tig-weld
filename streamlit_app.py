import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("microhardness_model.pkl")

st.title("Microhardness Prediction Model")

st.write("Predict Vickers Microhardness based on process parameters")

# User inputs
pw_nw = st.number_input("PW/NW", min_value=0.0, step=0.01)
current = st.number_input("Current (A)", min_value=0.0, step=0.1)
scan_speed = st.number_input("Scan Speed (mm/s)", min_value=0.0, step=1.0)

if st.button("Predict Microhardness"):
    input_data = np.array([[pw_nw, current, scan_speed]])
    prediction = model.predict(input_data)

    st.success(f"Predicted Microhardness: {prediction[0]:.2f} HV")
