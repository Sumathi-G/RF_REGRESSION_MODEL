import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Regression Prediction App")

st.title("🔮 Regression Prediction App")

@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ model file not found")
        st.stop()
    return joblib.load(model_path)

model = load_model()

f1 = st.number_input("Feature 1", 0.0)
f2 = st.number_input("Feature 2", 0.0)
f3 = st.number_input("Feature 3", 0.0)

if st.button("Predict"):
    X = np.array([[f1, f2, f3]])
    pred = model.predict(X)
    st.success(f"Prediction: {pred[0]}")
