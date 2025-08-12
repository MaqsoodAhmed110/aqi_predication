import pandas as pd
import xgboost as xgb
import joblib
import plotly.express as px
from dash import Dash, dcc, html
from datetime import datetime, timedelta

# Load latest model and features
model = xgb.XGBRegressor()
model.load_model("models/aqi_model.json")
features = joblib.load("models/features.pkl")

# Load latest features
df = pd.read_csv("data/hourly_features.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Predict future AQI (simple example: use last known values and forecast)
forecast_hours = 24
future_times = [df['timestamp'].max() + timedelta(hours=i) for i in range(1, forecast_hours+1)]
future_df = pd.DataFrame([df[features].iloc[-1].values] * forecast_hours, columns=features)
future_aqi = model.predict(future_df)

# Create dashboard app
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Real-Time & Forecasted AQI Dashboard"),
    html.P(f"Last Updated: {datetime.now()}"),

    html.H2("Latest AQI Trends"),
    dcc.Graph(figure=px.line(df, x="timestamp", y="us_aqi", title="Observed AQI")),

    html.H2("Forecasted AQI (Next 24 Hours)"),
    dcc.Graph(figure=px.line(x=future_times, y=future_aqi, labels={'x': 'Time', 'y': 'Predicted AQI'}, title="Forecasted AQI"))
])

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8080)
