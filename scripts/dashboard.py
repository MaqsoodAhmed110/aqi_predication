import pandas as pd
import joblib
from dash import Dash, dcc, html, dash_table
import dash_bootstrap_components as dbc
import xgboost as xgb
import os
# --------------------
# Paths & Config
# --------------------
MODEL_PATH = "models/aqi_model.json"   # Adjust if needed
DATA_PATH = "data/hourly_features.csv"
OUTPUT_PATH = "data/dashboard_predictions.csv"

FEATURES = [
    'pm10',
    'carbon_monoxide',
    'sulphur_dioxide',
    'dust',
    'aerosol_optical_depth',
    'ozone'
]

# --------------------
# --- Load model ---
print(f"Loading XGBoost model from {MODEL_PATH}...")
model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

# --------------------
# Load and Clean Data
# --------------------
print(f"Loading data from {DATA_PATH}...")
df_raw = pd.read_csv(DATA_PATH)

missing_features = [f for f in FEATURES if f not in df_raw.columns]
if missing_features:
    raise ValueError(f"Missing features in input data: {missing_features}")

df = df_raw[FEATURES].copy()
df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

# --------------------
# Predict
# --------------------
print("Making predictions...")
df_raw["predicted_aqi"] = model.predict(df)

# Save predictions to CSV
df_raw.to_csv(OUTPUT_PATH, index=False)
print(f"Predictions saved to {OUTPUT_PATH}")

# --------------------
# Create Dashboard
# --------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "AQI Prediction Dashboard"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Air Quality Index Predictions"), className="mb-4")
    ]),
    dbc.Row([
        dbc.Col(dash_table.DataTable(
            id='aqi-table',
            columns=[{"name": col, "id": col} for col in df_raw.columns],
            data=df_raw.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center', 'minWidth': '100px', 'maxWidth': '180px'},
            page_size=15
        ))
    ]),
    dbc.Row([
        dbc.Col(html.A(
            "Download Predictions CSV",
            href=f"/download/{OUTPUT_PATH}",
            target="_blank",
            className="btn btn-primary mt-3"
        ))
    ])
], fluid=True)

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
