import pandas as pd
import xgboost as xgb
import dash
from dash import dcc, html, dash_table

# Paths
MODEL_PATH = "models/aqi_model.json"
DATA_PATH = "data/hourly_features.csv"
OUTPUT_PATH = "data/dashboard_predictions.csv"

# Features used in training
FEATURES = [
    'pm10',
    'carbon_monoxide',
    'sulphur_dioxide',
    'dust',
    'aerosol_optical_depth',
    'ozone'
]

# --- Load model ---
print(f"Loading XGBoost model from {MODEL_PATH}...")
model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

# --- Load feature data ---
print(f"Loading data from {DATA_PATH}...")
df_raw = pd.read_csv(DATA_PATH)

# Ensure all features exist
missing_features = [f for f in FEATURES if f not in df_raw.columns]
if missing_features:
    raise ValueError(f"Missing features in input data: {missing_features}")

# Select and clean features
df = df_raw[FEATURES].copy()
df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

# --- Make predictions ---
print("Making predictions...")
df_raw["predicted_aqi"] = model.predict(df)

# Save results
df_raw.to_csv(OUTPUT_PATH, index=False)
print(f"Predictions saved to {OUTPUT_PATH}")

# --- Build dashboard ---
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Air Quality Index Predictions"),
    dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in df_raw.columns],
        data=df_raw.to_dict("records"),
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center"},
    ),
    html.Br(),
    dcc.Graph(
        figure={
            "data": [
                {
                    "x": df_raw.index,
                    "y": df_raw["predicted_aqi"],
                    "type": "line",
                    "name": "Predicted AQI"
                }
            ],
            "layout": {"title": "Predicted AQI Over Records"}
        }
    )
])

# Run the app (updated for Dash >= 2.16)
if __name__ == "__main__":
    try:
        app.run(debug=True, host="0.0.0.0", port=8050)
    except AttributeError:
        # Fallback for old versions of Dash
        app.run_server(debug=True, host="0.0.0.0", port=8050)
