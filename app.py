import streamlit as st
import numpy as np
import pickle

# Page settings
st.set_page_config(page_title="Earthquake Death Predictor", layout="centered")

# Custom Styling (unchanged)
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

# UI Header
st.markdown("Enter earthquake parameters to estimate deaths using ML models.")

# Model selection
model_options = [
    "Random Forest",
    "Decision Tree",
    "Linear Regression",
    "Gradient Boosting",
    "KNN"
]
selected_model = st.sidebar.selectbox("Choose Model", model_options)

# User inputs
magnitude = st.number_input("Magnitude", min_value=1.0, value=6.5, step=0.5)
depth = st.number_input("Depth (km)", min_value=0.0, max_value=700.0, value=10.0, step=10.0)
pop_density = st.number_input("Population Density (people/km²)", min_value=0.0, value=200.0, step=100.0)
urban_rate = st.number_input("Urbanization Rate (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)

# Map UI selection to file names
model_map = {
    "Random Forest": "random_forest",
    "Decision Tree": "decision_tree",
    "Linear Regression": "linear_regression",
    "Gradient Boosting": "gradient_boosting",
    "KNN": "knn"
}

# Load the corresponding model
def load_model(model_key):
    model_file = f"{model_key}.pkl"
    try:
        with open(model_file, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"❌ Model file '{model_file}' not found.")
        return None

# Predict button
if st.button("Predict"):
    model_key = model_map[selected_model]
    model = load_model(model_key)

    if model:
        try:
            features = np.array([[magnitude, depth, pop_density, urban_rate]])
            prediction = model.predict(features)
            st.markdown(
                f"<h3 style='color:#dc2626;'>Predicted Deaths: {int(prediction[0]):,}</h3>",
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
