import streamlit as st
import pandas as pd
import pickle

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Vickers Microhardness Predictor",
    layout="wide"
)

st.title("Vickers Microhardness Prediction App")
st.markdown("---")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    with open("random_forest_regressor_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("Input Parameters")

current = st.sidebar.slider(
    "Current [A]", 100.0, 120.0, 100.0, step=1.0
)

scan_speed = st.sidebar.slider(
    "Scan Speed [mm/s]", 0.7, 3.0, 1.0, step=0.1
)

pw_nw = st.sidebar.selectbox(
    "PW/NW",
    [
        "PW",
        "NW",
        "NW- NORMAL (WITHOUT POWDER)",
        "PW- POWDER APPLIED"
    ]
)

# ---------------- Prepare Input ----------------
input_df = pd.DataFrame({
    "Current [A]": [current],
    "Scan Speed [mm/s]": [scan_speed],
    "PW/NW": [pw_nw]
})

# ---------------- Prediction ----------------
if st.sidebar.button("Predict Vickers Microhardness"):
    prediction = model.predict(input_df)
    st.metric(
        label="Predicted Vickers Microhardness [HV 0.05]",
        value=f"{prediction[0]:.2f}"
    )

st.markdown("---")
st.write("### About")
st.write(
    "This app predicts **Vickers Microhardness** using a pre-trained model. "
    "Only inference logic is included; all training and preprocessing were done offline."
)
