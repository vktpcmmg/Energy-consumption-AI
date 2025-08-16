import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

# ===============================
# Load Logo (PNG file should be in same folder)
# ===============================
def get_image_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_base64 = get_image_base64("tata_logo.png")

st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{logo_base64}" width="100">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h1 style='text-align: center; color: #0072C6;'>🔌 Electricity Consumption Predictor </h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center; color: gray;'>🔷 Designed by <span style='color: #0072C6;'>Tata Power - MMG</span></h4>",
    unsafe_allow_html=True
)

# ===============================
# Load Saved Models and Encoders
# ===============================
@st.cache_resource
def load_models():
    with open("trained_models.pkl", "rb") as f:
        models = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    return models, label_encoders

models, label_encoders = load_models()

# ===============================
# App Description
# ===============================
st.markdown("*_Note: This model is trained on ~5.8 lakh smart meter records from FY 24–25._*")
st.write("Enter details to predict monthly electricity usage (kWh/KVAh).")

# ===============================
# User Inputs
# ===============================
connected_load = st.number_input("Connected Load (kW/KVA)", min_value=0.0, value=10.0)

zone = st.selectbox("Select Zone", label_encoders['Zone'].classes_)

category = st.selectbox("Select Category", label_encoders['Category'].classes_)

month = st.selectbox("Select Month", list(models.keys()))

# ===============================
# Prediction
# ===============================
if connected_load <= 0:
    st.error("⚠️ Please enter a valid load. Zero or negative load does not exist.")
else:
    if st.button("🔍 Predict Consumption"):
        zone_enc = label_encoders['Zone'].transform([zone])[0]
        category_enc = label_encoders['Category'].transform([category])[0]

        input_data = np.array([[category_enc, connected_load, zone_enc]])
        prediction = models[month].predict(input_data)[0]

        st.success(f"📊 Predicted electricity consumption for **{month}**: **{prediction:.2f} kWh**")
