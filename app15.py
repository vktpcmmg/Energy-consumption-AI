import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import base64

# Load logo
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

# Title and subtitle
st.markdown("<h1 style='text-align: center; color: #0072C6;'>🔌 Electricity Consumption Predictor </h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>🔷 Designed by <span style='color: #0072C6;'>Tata Power - MMG</span></h4>", unsafe_allow_html=True)

st.markdown("*_Note: This is based on around 100K smart meter data from FY 24–25._*")
st.write("Enter details to predict monthly electricity usage (kWh/KVAh).")

# Load CSV
@st.cache_data
def load_data():
    df = pd.read_csv("consumptionai2.csv")
    df.columns = df.columns.str.strip()
    df.rename(columns={'Connected  Load': 'Connected Load'}, inplace=True)

    # Trim strings
    for col in ['Zone', 'Category']:
        df[col] = df[col].astype(str).str.strip()

    # Numeric conversion
    df['Connected Load'] = pd.to_numeric(df['Connected Load'], errors='coerce')
    months = [col for col in df.columns if col not in ['Connected Load', 'Zone', 'Category']]
    df[months] = df[months].apply(pd.to_numeric, errors='coerce').fillna(0)
    df = df.dropna(subset=['Connected Load', 'Zone', 'Category'])
    return df, months

# Train models
@st.cache_resource
def train_models(df, months):
    label_encoders = {}
    for col in ['Zone', 'Category']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    input_features = ['Connected Load', 'Zone', 'Category']
    models = {}
    for month in months:
        X = df[input_features]
        y = df[month].astype(float)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        models[month] = model

    return models, label_encoders

# Load data and models
df, months = load_data()
models, label_encoders = train_models(df, months)

# User inputs
connected_load = st.number_input("Connected Load (kW/KVA)", min_value=0.0, value=10.0)
zone = st.selectbox("Select Zone", label_encoders['Zone'].classes_)
category = st.selectbox("Select Category", label_encoders['Category'].classes_)

# Encode categorical inputs
zone_enc = label_encoders['Zone'].transform([zone])[0]
category_enc = label_encoders['Category'].transform([category])[0]

# Predict for all months
if connected_load <= 0:
    st.error("⚠️ Please enter a valid load. Zero or negative load does not exist.")
else:
    if st.button("🔍 Predict Yearly Consumption"):
        input_data = np.array([[connected_load, zone_enc, category_enc]])
        predictions = {month: models[month].predict(input_data)[0] for month in months}
        
        # Display results
        st.success("📊 Predicted electricity consumption (kWh) for all months:")
        st.table(pd.DataFrame(predictions.items(), columns=["Month", "Predicted Consumption"]))

        # Bar chart
        st.bar_chart(pd.DataFrame(predictions, index=[0]).T.rename(columns={0: "Consumption"}))
