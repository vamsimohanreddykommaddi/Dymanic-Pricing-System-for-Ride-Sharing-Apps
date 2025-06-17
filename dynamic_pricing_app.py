import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.graph_objects as go
from PIL import Image

# Load the trained model and preprocessing pipeline
model = pickle.load(open(r"C:\Users\vamshi\best_dynamic_pricing_model.pkl", 'rb'))
preprocessor = joblib.load(r"C:\Users\vamshi\preprocess_dynamic")


# Streamlit App UI
st.set_page_config(page_title="Dynamic Ride Pricing", layout="centered")
st.title("🚕 Dynamic Ride Pricing Predictor")

# Input Section
st.subheader("Enter Ride Details")
riders = st.number_input("Number of Riders", min_value=1, value=10)
drivers = st.number_input("Number of Drivers", min_value=1, value=5)
duration = st.number_input("Expected Ride Duration (minutes)", min_value=1, value=30)
vehicle_type = st.selectbox("Vehicle Type", ["Economy", "Premium"])

# DataFrame for model
input_df = pd.DataFrame({
    'Number_of_Riders': [riders],
    'Number_of_Drivers': [drivers],
    'Expected_Ride_Duration': [duration],
    'Vehicle_Type': [vehicle_type]
})

# Prediction
if st.button("Predict Ride Price 💰"):
    input_processed = preprocessor.transform(input_df)
    prediction = model.predict(input_processed)
    st.success(f"Predicted Ride Cost: ₹{prediction[0]:.2f}")


# Display pre-generated image of donut chart
st.subheader("📊 Profitability of Dynamic Pricing (Based on Historical Data)")

# Load and display the image
image = Image.open(r"D:\Dynamic Pricing strategy\profit-chart.png")
st.image(image, caption='Profitable vs Loss Rides', use_column_width=True)
