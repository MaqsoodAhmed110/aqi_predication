import pandas as pd
import plotly.express as px
import xgboost as xgb
import joblib
from datetime import datetime, timedelta

# Load model
model = xgb.XGBRegressor()
model.load_model("models/aqi_model.json")
features = ['pm10','carbon_monoxide','sulphur_dioxide','dust','aerosol_optical_depth','ozone']

# Load data
df = pd.read_csv("data/hourly_features.csv")
df[features] = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
df["predicted_aqi"] = model.predict(df[features])

# Last 24 hours chart
fig1 = px.line(df.tail(24), x='timestamp', y='predicted_aqi',
               title="Predicted AQI - Last 24 Hours")

# --- Replace mock forecast with actual model predictions ---

# Create timestamps for next 72 hours
future_hours = [datetime.now() + timedelta(hours=i) for i in range(1, 73)]

# For demonstration, let's assume future features are the same as the last known row
# (In practice, you’d load or predict actual future values for pollutants)
last_row = df[features].iloc[-1]
future_features = pd.DataFrame([last_row.values] * 72, columns=features)

# Predict AQI for the next 72 hours using the trained model
future_predictions = model.predict(future_features)

# Build forecast dataframe
forecast_data = pd.DataFrame({
    "hour": future_hours,
    "predicted_aqi": future_predictions
})

# Create forecast chart
fig2 = px.line(forecast_data, x='hour', y='predicted_aqi',
               title="Predicted AQI - Next 72 Hours")

# Save as HTML
with open("public/index.html", "w") as f:
    f.write("<h1>Air Quality Dashboard</h1>")
    f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write("<hr>")
    f.write(fig2.to_html(full_html=False, include_plotlyjs=False))
