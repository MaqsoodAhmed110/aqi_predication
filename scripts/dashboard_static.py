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

# 72-hour forecast mock
forecast_data = pd.DataFrame({
    'hour': [datetime.now() + timedelta(hours=i) for i in range(72)],
    'predicted_aqi': [120 + (i % 20) for i in range(72)]
})
fig2 = px.line(forecast_data, x='hour', y='predicted_aqi',
               title="Predicted AQI - Next 72 Hours")

# Save as HTML
with open("public/index.html", "w") as f:
    f.write("<h1>Air Quality Dashboard</h1>")
    f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write("<hr>")
    f.write(fig2.to_html(full_html=False, include_plotlyjs=False))
