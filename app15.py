import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import base64

# =============================
# --- Logo Loader Function ---
# =============================
def get_image_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load Tata Power logo
logo_base64 = get_image_base64("tata_logo.png")

# =============================
# --- Page Header ---
# =============================
st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{logo_base64}" width="110">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h1 style='text-align: center; color: #0072C6;'>🔌 Electricity Consumption Predictor</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center; color: gray;'>🔷 Designed by <span style='color: #0072C6;'>Tata Power - MMG</span></h4>",
    unsafe_allow_html=True
)

# =============================
# --- Load & Merge Data ---
# =============================
@st.cache_data
def load_data():
    df1 = pd.read_csv("consumptionai1.csv")
    df2 = pd.read_csv("consumptionai2.csv")
    df3 = pd.read_csv("consumptionai3.csv")

    df = pd.concat([df1, df2, df3], ignore_index=True)

    # Clean column names
    df.columns = df.columns.str.strip()
    df.rename(columns={'Connected  Load': 'Connected Load'}, inplace=True)
    return df

# =============================
# --- Train Models ---
# =============================
@st.cache_resource
def train_models(df):
    label_encoders = {}
    for col in ['Zone', 'Category']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    input_features = ['Connected Load', 'Zone', 'Category']
    months = ['May', 'Jun', 'Jul', 'August', 'Sept', 'Oct', 'Nov', 'Dec', 
              'Jan', 'Feb', 'Mar', 'Apr']
    models = {}

    for month in months:
        X = df[input_features]
        y = df[month]
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        models[month] = model

    return models, label_encoders

# =============================
# --- Load Data & Train ---
# =============================
st.markdown("*_Note: This is based on around 100K smart meter data from FY 24–25._*")
st.write("Enter details to predict monthly electricity usage (kWh/KVAh).")

df = load_data()
models, label_encoders = train_models(df)

# =============================
# --- User Inputs ---
# =============================
connected_load = st.number_input("Connected Load (kW/KVA)", min_value=0.0, value=10.0, step=0.5)
zone = st.selectbox("Select Zone", label_encoders['Zone'].classes_)
category = st.selectbox("Select Category", label_encoders['Category'].classes_)
month = st.selectbox("Select Month", 
                     ['May','Jun','Jul','August','Sept','Oct','Nov','Dec',
                      'Jan','Feb','Mar','Apr'])

# Encode categorical inputs
zone_enc = label_encoders['Zone'].transform([zone])[0]
category_enc = label_encoders['Category'].transform([category])[0]

# =============================
# --- Prediction ---
# =============================
if connected_load <= 0:
    st.error("⚠️ Please enter a valid load. Zero or negative load does not exist.")
else:
    if st.button("🔍 Predict Consumption"):
        input_data = np.array([[connected_load, zone_enc, category_enc]])
        prediction = models[month].predict(input_data)[0]
        st.success(f"📊 Predicted electricity consumption for **{month}**: **{prediction:.2f} kWh**")
