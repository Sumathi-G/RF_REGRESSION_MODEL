import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(page_title="Regression Prediction App", layout="centered")

st.title("🔮 Regression Prediction App")

# Load model safely
@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        st.error("❌ model.pkl not found. Please upload it to the GitHub repo.")
        st.stop()
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.write("Enter feature values below:")

# Example: 3 input features (change if needed)
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

if st.button("Predict"):
    input_data = np.array([[feature1, feature2, feature3]])
    prediction = model.predict(input_data)
    st.success(f"✅ Prediction: {prediction[0]}")
