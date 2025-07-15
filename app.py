import streamlit as st
import numpy as np
import pandas as pd
import os
from joblib import load
from sklearn.preprocessing import StandardScaler

# Page layout
st.set_page_config(page_title="Earthquake Death Predictor", layout="centered")

# Custom styling
st.markdown("""
<style>
/* App background */
.stApp {
    background-color: #fef2f2;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #fdecea;
    color: #7f1d1d;
}

/* Header */
header[data-testid="stHeader"] {
    background-color: #fee2e2;
}

/* Number input, text input */
input[type="number"], input[type="text"] {
    background-color: #fff5f5 !important;
    border: 1px solid #fca5a5 !important;
    color: #7f1d1d !important;
    border-radius: 6px;
    padding: 0.4rem;
}

/* Fix for selectbox input display */
div[data-baseweb="select"] > div {
    background-color: #fff5f5 !important;
    border: 1px solid #fca5a5 !important;
    color: #7f1d1d !important;
    border-radius: 6px;
}

/* Fix for selectbox options dropdown */
ul[role="listbox"] {
    background-color: #fff5f5 !important;
    border: 1px solid #fca5a5 !important;
}
ul[role="listbox"] > li {
    color: #7f1d1d !important;
    background-color: #fff5f5 !important;
}
ul[role="listbox"] > li:hover {
    background-color: #fca5a5 !important;
}

/* Buttons */
.stButton > button {
    background-color: #dc2626;
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 5px;
    padding: 8px 20px;
}
.stButton > button:hover {
    background-color: #b91c1c;
}

/* General text */
h1, h2, h3, h4, h5, h6, p, label, div {
    color: #7f1d1d;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("Enter earthquake parameters to estimate deaths using ML models.")

# Sidebar model selector
model_options = [
    "Random Forest",
    "Decision Tree",
    "Linear Regression",
    "Gradient Boosting",
    "KNN"
]
selected_model = st.sidebar.selectbox("Choose Model", model_options)

# Input fields
magnitude = st.number_input("Magnitude", min_value=1.0, value=6.5, step=0.5)
depth = st.number_input("Depth (km)", min_value=0.0, max_value=700.0, value=10.0, step=10.0)
pop_density = st.number_input("Population Density (people/km²)", min_value=0.0, value=200.0, step=100.0)
urban_rate = st.number_input("Urbanization Rate (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)

# Model filename mapping
model_map = {
    "Random Forest": "random_forest_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "Linear Regression": "linear_regression_model.pkl",
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "KNN": "knn_model.pkl"
}

# Load scaler and correlation weights from dataset
@st.cache_resource
def load_scaler_and_correlations():
    df = pd.read_csv("earthquake_dataset.csv")
    df = df[df["Year"] >= 1960]
    df = df[["Magnitude", "Depth_km", "Population_Density", "Urbanization_Rate", "Deaths"]].dropna()

    X = df[["Magnitude", "Depth_km", "Population_Density", "Urbanization_Rate"]]
    scaler = StandardScaler().fit(X)
    correlations = df.corr(numeric_only=True)["Deaths"].drop("Deaths").abs().values  # shape (4,)
    return scaler, correlations

# Preprocess user inputs
def preprocess_inputs(mag, depth, pop, urban, scaler, correlations):
    raw_input = np.array([[mag, depth, pop, urban]])
    scaled = scaler.transform(raw_input)
    weighted = scaled * correlations
    return weighted

# Load model
def load_model(model_key):
    model_file = model_map.get(model_key)
    if not model_file or not os.path.exists(model_file):
        st.error(f"❌ Model file '{model_file}' not found.")
        return None
    return load(model_file)

# Prediction block
if st.button("Predict"):
    model = load_model(selected_model)
    scaler, correlations = load_scaler_and_correlations()

    if model is not None:
        try:
            processed_input = preprocess_inputs(magnitude, depth, pop_density, urban_rate, scaler, correlations)
            prediction = model.predict(processed_input)
            st.markdown(
                f"<h3 style='color:#dc2626;'>Predicted Deaths: {int(prediction[0]):,}</h3>",
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
