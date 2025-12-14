import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page configuration
st.set_page_config(layout="wide", page_title="Vickers Microhardness Predictor")

st.title("Vickers Microhardness Prediction App")
st.markdown("--- ")

# --- Load the trained model and scaler ---
try:
    with open('random_forest_regressor_model.pkl', 'rb') as file:
        rf_model = pickle.load(file)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Error: 'random_forest_regressor_model.pkl' not found. Please ensure the model is saved in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Placeholder for scaler - in a real scenario, the scaler would also be saved/loaded
# For this demonstration, we'll recreate a dummy scaler that matches the training data properties
# In a real application, you would save and load the *trained* scaler object.

# Re-creating the scaler based on the original X_train data structure and scaling
# NOTE: This is a simplification. Ideally, the trained scaler would be saved.
# Based on kernel state, numerical_cols = ['Current [A]', 'Scan Speed [mm/s]']
# X_train was scaled with StandardScaler
# To simulate a loaded scaler, we need min/max or mean/std from the training data.
# For this exercise, let's assume we have access to the scaler's parameters from training.
# For proper deployment, always save your scaler object along with the model.

# Let's assume we had a scaler object with these parameters (example values):
# The original X_train (before scaling) would look like this for numerical columns:
# Current [A]: mean ~100-104, std ~0-8.4
# Scan Speed [mm/s]: mean ~1-1.5, std ~0-0.9

# As a temporary measure for this demo, we'll manually define parameters based on the original data's mean and std before scaling
# In a production setup, you MUST save and load the actual StandardScaler object.
class DummyScaler:
    def __init__(self, mean, scale, numerical_cols):
        self.mean_ = np.array(mean)
        self.scale_ = np.array(scale)
        self.numerical_cols = numerical_cols

    def transform(self, X):
        X_scaled = X.copy()
        for i, col in enumerate(self.numerical_cols):
            if col in X_scaled.columns:
                X_scaled[col] = (X_scaled[col] - self.mean_[i]) / self.scale_[i]
        return X_scaled

# These values are approximate based on the descriptive statistics of the X_train before scaling
# A better approach is to load the actual scaler instance.
# df.describe() shows mean Current [A]: 104, std Current [A]: 8.43
# df.describe() shows mean Scan Speed [mm/s]: 1.48, std Scan Speed [mm/s]: 0.946
# We need the mean and std from the *training data before scaling*, not the full df.
# Let's derive a reasonable approximation based on the original `df`'s numerical columns
# and assuming `X_train` would have similar stats for `Current [A]` and `Scan Speed [mm/s]`

# We will use the mean and std values of the full original numerical columns as a proxy
# for the scaler's parameters, as the actual fitted scaler object was not saved.
# THIS IS NOT IDEAL FOR PRODUCTION, BUT NECESSARY FOR THIS DEMO WITHOUT THE SAVED SCALER.

# Mean and std from df.describe() for numerical columns used in X
# (Current [A], Scan Speed [mm/s])
original_numerical_means = {
    'Current [A]': df['Current [A]'].mean(),
    'Scan Speed [mm/s]': df['Scan Speed [mm/s]'].mean()
}
original_numerical_stds = {
    'Current [A]': df['Current [A]'].std(),
    'Scan Speed [mm/s]': df['Scan Speed [mm/s]'].std()
}

# Create a dummy scaler with these values
# For `scaler = StandardScaler()` X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
# The actual mean and std are from X_train *before* scaling.
# Based on the X_train kernel variable (already scaled), we can't directly get pre-scaled stats.
# Re-running the preprocessing steps on a temporary dataframe to get scaler parameters

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Re-create the preprocessing to get the actual scaler state
temp_df = pd.read_excel('/content/HARDNESS AND UTS.xlsx')
for column in ['Current [A]', 'Scan Speed [mm/s]', 'Vickers Microhardness [HV 0.05]', 'Ultimate Tensile Strength [MPa]']:
    temp_df[column].fillna(temp_df[column].median(), inplace=True)
temp_df['PW/NW'].fillna(temp_df['PW/NW'].mode()[0], inplace=True)

temp_df = pd.get_dummies(temp_df, columns=['PW/NW'], drop_first=False)
y_temp = temp_df['Vickers Microhardness [HV 0.05]']
X_temp = temp_df.drop(columns=['Vickers Microhardness [HV 0.05]', 'Ultimate Tensile Strength [MPa]'])

X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

scaler = StandardScaler()
numerical_cols = ['Current [A]', 'Scan Speed [mm/s]']
scaler.fit(X_train_temp[numerical_cols]) # Fit the scaler on the numerical training data

st.sidebar.header("Input Features")

# --- Input Widgets ---
current = st.sidebar.slider("Current [A]", min_value=100.0, max_value=120.0, value=100.0, step=1.0)
scan_speed = st.sidebar.slider("Scan Speed [mm/s]", min_value=0.7, max_value=3.0, value=1.0, step=0.1)
pw_nw = st.sidebar.selectbox("PW/NW", ['PW', 'NW', 'NW- NORMAL (WITHOUT POWDER)', 'PW- POWDER APPLIED'])

# --- Preprocess User Input ---
def preprocess_input(current_val, scan_speed_val, pw_nw_val, scaler):
    # Create a DataFrame for the input
    input_data = pd.DataFrame([[current_val, scan_speed_val, pw_nw_val]],
                              columns=['Current [A]', 'Scan Speed [mm/s]', 'PW/NW'])

    # One-hot encode 'PW/NW' - ensure all possible columns are present with 0s if not selected
    # Based on X.columns in the kernel:
    # ['Current [A]', 'Scan Speed [mm/s]', 'PW/NW_NW', 'PW/NW_NW- NORMAL (WITHOUT POWDER)', 'PW/NW_PW', 'PW/NW_PW- POWDER APPLIED']
    all_pw_nw_cols = [
        'PW/NW_NW',
        'PW/NW_NW- NORMAL (WITHOUT POWDER)',
        'PW/NW_PW',
        'PW/NW_PW- POWDER APPLIED'
    ]
    
    # Create dummy variables for 'PW/NW'
    input_data = pd.get_dummies(input_data, columns=['PW/NW'], drop_first=False)

    # Add missing dummy columns with 0
    for col in all_pw_nw_cols:
        if col not in input_data.columns:
            input_data[col] = 0

    # Ensure column order matches training data X_train
    # Reconstruct the column order based on X_train_temp.columns from the re-created preprocessing
    final_columns_order = X_train_temp.columns
    input_data = input_data.reindex(columns=final_columns_order, fill_value=0)

    # Scale numerical features
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
    
    return input_data

processed_input = preprocess_input(current, scan_speed, pw_nw, scaler)

# --- Make Prediction ---
if st.sidebar.button("Predict Vickers Microhardness"):
    prediction = rf_model.predict(processed_input)
    st.metric(label="Predicted Vickers Microhardness [HV 0.05]", value=f"{prediction[0]:.2f}")

st.markdown("--- ")
st.write("### About this App")
st.write("This Streamlit application predicts Vickers Microhardness based on user-provided material processing parameters.")
st.write("The prediction is made using a pre-trained Random Forest Regressor model.")
st.write("Please adjust the input features using the sliders and select box on the sidebar and click 'Predict'.")
