import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pickle

# ---------------- TRAIN MODELS ----------------
def train_models(df):
    models = {}
    label_encoders = {}

    # Encode categorical columns
    categorical_cols = ["meter_number", "Category", "Zone"]
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Features
    feature_cols = ["meter_number", "Category", "Connected  Load", "Zone"]

    # Targets (monthly consumption)
    target_cols = ["May","Jun","Jul","August","Sept","Oct","Nov","Dec",
                   "Jan","Feb","Mar","Apr"]

    for target in target_cols:
        X = df[feature_cols]
        y = df[target]

        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
        y = pd.to_numeric(y, errors="coerce").fillna(0)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        models[target] = model

    return models, label_encoders


# ---------------- PREDICT FUNCTION ----------------
def predict_consumption(models, label_encoders, input_data, month):
    # Encode categorical inputs
    for col in ["meter_number", "Category", "Zone"]:
        le = label_encoders[col]
        input_data[col] = le.transform([input_data[col]])[0]

    # Make sure correct order of features
    X_new = pd.DataFrame([input_data])[["meter_number", "Category", "Connected  Load", "Zone"]]

    # Predict from correct month's model
    prediction = models[month].predict(X_new)[0]
    return prediction


# ---------------- STREAMLIT APP ----------------
st.title("Energy Consumption Predictor")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("Data uploaded successfully!")

    models, label_encoders = train_models(df)

    # User input
    meter_number = st.text_input("Enter Meter Number")
    category = st.text_input("Enter Category")
    connected_load = st.number_input("Enter Connected Load (kW)", min_value=0.0)
    zone = st.selectbox("Select Zone", ["MC", "SZ", "CZ", "NZ", "EZ", "SC"])

    month = st.selectbox("Select Month to Predict", 
                         ["May","Jun","Jul","August","Sept","Oct","Nov","Dec","Jan","Feb","Mar","Apr"])

    if st.button("Predict"):
        input_data = {
            "meter_number": meter_number,
            "Category": category,
            "Connected  Load": connected_load,
            "Zone": zone
        }

        try:
            result = predict_consumption(models, label_encoders, input_data, month)
            st.success(f"Predicted consumption for {month}: {result:.2f} units")
        except Exception as e:
            st.error(f"Error: {e}")
