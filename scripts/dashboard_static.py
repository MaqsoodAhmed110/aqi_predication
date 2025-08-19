import requests
import pandas as pd
import plotly.express as px
import xgboost as xgb
import joblib
from datetime import datetime

# -------------------------
# Load model
# -------------------------
model = xgb.XGBRegressor()
model.load_model("models/aqi_model.json")

features = ['pm10','carbon_monoxide','sulphur_dioxide','dust','aerosol_optical_depth','ozone']

# -------------------------
# Load local historical data (last 24 hrs)
# -------------------------
df = pd.read_csv("data/hourly_features.csv")
df[features] = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
df["predicted_aqi"] = model.predict(df[features])

# -------------------------
# Chart for last 24 hours
# -------------------------
fig1 = px.line(df.tail(24), x='timestamp', y='predicted_aqi',
               title="Predicted AQI - Last 24 Hours")

# -------------------------
# Fetch future data (72-hour forecast from API)
# -------------------------
def fetch_forecast(lat=24.6844, lon=67.0479):  # default: Islamabad, Pakistan
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=pm10,carbon_monoxide,sulphur_dioxide,aerosol_optical_depth,ozone,dust"
        f"&forecast_days=3"
    )

    response = requests.get(url)
    if response.status_code != 200:
        return pd.DataFrame()

    data = response.json()
    if "hourly" not in data:
        return pd.DataFrame()

    hourly = data["hourly"]
    forecast_df = pd.DataFrame(hourly)

    # Ensure features exist
    if not all(f in forecast_df.columns for f in features):
        return pd.DataFrame()

    # Convert to numeric & fill NA
    forecast_df[features] = forecast_df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    forecast_df["timestamp"] = pd.to_datetime(forecast_df["time"])

    return forecast_df

# Get forecast data
forecast_df = fetch_forecast()

if not forecast_df.empty:
    # Predict AQI for forecast data
    forecast_df["predicted_aqi"] = model.predict(forecast_df[features])

    # Keep only next 72 hours
    forecast_df = forecast_df.head(72)

    # Forecast chart
    fig2 = px.line(forecast_df, x='timestamp', y='predicted_aqi',
                   title="Predicted AQI - Next 72 Hours")
else:
    fig2 = px.line(title="No Forecast Data Available")

# -------------------------
# Save as HTML dashboard
# -------------------------
with open("public/index.html", "w") as f:
    f.write("<h1>Air Quality Dashboard</h1>")
    f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write("<hr>")
    f.write(fig2.to_html(full_html=False, include_plotlyjs=False))
