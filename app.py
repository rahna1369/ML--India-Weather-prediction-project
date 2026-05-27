import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved models
temperature_model = joblib.load("temperature_model.pkl")
rainfall_model = joblib.load("rainfall_model.pkl")
rain_classifier = joblib.load("rain_classifier_model.pkl")
encoder = joblib.load("city_encoder.pkl")

# Load dataset
df = pd.read_csv("cleaned_weather_dataset.csv")

# Create year and month if not available
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

if "month" not in df.columns and "date" in df.columns:
    df["month"] = df["date"].dt.month

if "year" not in df.columns and "date" in df.columns:
    df["year"] = df["date"].dt.year

# Create prcp_log if not available
if "prcp_log" not in df.columns:
    df["prcp_log"] = np.log1p(df["prcp"])

# Streamlit app
st.set_page_config(page_title="Weather Prediction App", layout="centered")

st.title("Weather Prediction App")
st.write("Predict average temperature, rainfall amount, and rain/no-rain status.")

# User input
city_list = sorted(df["city"].unique())

city = st.selectbox("Select City", city_list)
month = st.selectbox("Select Month", list(range(1, 13)))
year = st.number_input("Enter Year", min_value=2026, max_value=2035, value=2026)

# Filter selected city and month
city_month_data = df[
    (df["city"] == city) &
    (df["month"] == month)
]

if not city_month_data.empty:
    tavg = city_month_data["tavg"].mean()
    tmin = city_month_data["tmin"].mean()
    tmax = city_month_data["tmax"].mean()
    prcp = city_month_data["prcp"].mean()
    pres = city_month_data["pres"].mean()
else:
    tavg = df["tavg"].mean()
    tmin = df["tmin"].mean()
    tmax = df["tmax"].mean()
    prcp = df["prcp"].mean()
    pres = df["pres"].mean()

# Feature preparation
city_encoded = encoder.transform([city])[0]
prcp_log = np.log1p(prcp)

if st.button("Predict Weather"):

    # Temperature input
    temp_input = pd.DataFrame({
        "tmin": [tmin],
        "tmax": [tmax],
        "prcp_log": [prcp_log],
        "pres": [pres],
        "city_encoded": [city_encoded],
        "year": [year],
        "month": [month]
    })

    predicted_temp = temperature_model.predict(temp_input)[0]

    # Rainfall input
    rain_input = pd.DataFrame({
        "tavg": [tavg],
        "tmin": [tmin],
        "tmax": [tmax],
        "pres": [pres],
        "city_encoded": [city_encoded],
        "year": [year],
        "month": [month]
    })

    predicted_rain_log = rainfall_model.predict(rain_input)[0]
    predicted_rainfall = np.expm1(predicted_rain_log)

    # Rain / No Rain classification
    rain_class = rain_classifier.predict(rain_input)[0]
    rain_status = "Rain" if rain_class == 1 else "No Rain"

    # Output
    st.subheader("Prediction Result")
    st.write("City:", city)
    st.write("Month:", month)
    st.write("Year:", year)

    st.success(f"Predicted Average Temperature: {predicted_temp:.2f} °C")
    st.info(f"Predicted Rainfall: {predicted_rainfall:.2f} mm")
    st.warning(f"Rain Status: {rain_status}")
