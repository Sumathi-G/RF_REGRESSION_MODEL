import streamlit as st
import numpy as np
import os

# App title
st.set_page_config(page_title="Regression Predictor", layout="centered")
st.title("🔮 Regression Prediction App")

# Load model
from joblib import load

@st.cache_resource
def load_model():
    return load("model.joblib")

model = load_model()

st.success("Model loaded successfully!")

# ---- INPUT FEATURES ----
st.header("Enter Input Features")

# ⚠️ Update these feature names according to your trained model
feature_1 = st.number_input("Feature 1", value=0.0)
feature_2 = st.number_input("Feature 2", value=0.0)
feature_3 = st.number_input("Feature 3", value=0.0)
feature_4 = st.number_input("Feature 4", value=0.0)

# Combine features into array
input_data = np.array([[feature_1, feature_2, feature_3, feature_4]])

# ---- PREDICTION ----
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        st.subheader("📈 Prediction Result")
        st.write(f"**Predicted Value:** {prediction[0]:.4f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---- FOOTER ----
st.markdown("---")
st.caption("Built with Streamlit & Scikit-learn")
