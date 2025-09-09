import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import requests
from pymongo import MongoClient

# Local MongoDB connect
client = MongoClient("mongodb://localhost:27017/")

# Database & collection
db = client["house_app"]
collection = db["predictions"]




# Load trained model
model_path = os.path.join( "house_price_model.pkl")
if os.path.exists('house_price_model.pkl'):
    model = joblib.load(model_path)
else:
    st.error("Model not found. Please train the model first!")
    st.stop()

st.set_page_config(page_title="🏠 House Price Prediction", layout="centered")

# Title
st.title("🏠 House Price Prediction Web App")
st.write("Enter house details below and get an estimated price prediction.")

# Sidebar inputs
st.sidebar.header("Input Features")

area = st.sidebar.number_input("Area (sq ft)", min_value=500, max_value=10000, value=1500, step=100)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 2)
stories = st.sidebar.slider("Stories", 1, 5, 2)
parking = st.sidebar.slider("Parking Spaces", 0, 5, 1)
year_built = st.sidebar.slider("Year Built", 1950, 2025, 2010)
city = st.sidebar.selectbox("City", ["Delhi", "Mumbai", "Bangalore", "Chennai", "Pune", "Kolkata"])

# Derived features (same as training)
house_age = 2025 - year_built
area_per_bedroom = area / bedrooms if bedrooms > 0 else area

# Prediction button
if st.sidebar.button("🔮 Predict Price"):
    # Prepare input data
    input_data = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,
        "year_built": year_built,
        "city": city,
        "house_age": house_age,
        "area_per_bedroom": area_per_bedroom
    }])
    
    # Predict
    prediction = model.predict(input_data)[0]
    (st.write(input_data.shape))
    
    # after computing pred (float)
    record = {
        "area": int(area),
        "bedrooms": int(bedrooms),
        "bathrooms": int(bathrooms),
        "stories": int(stories),
        "parking": int(parking),
        "year_built": int(year_built),
        "city": city,
        "house_age": int(house_age),
        "area_per_bedroom": float(area_per_bedroom),
        "predicted_price": float(prediction)
    }
    response = requests.post("http://127.0.0.1:8000/predict", json=record)
    result = response.json()
    prediction = result["predicted_price"]
    collection.insert_one(record)
    st.success(f"💰 Predicted House Price: {prediction:,.2f}")

   
