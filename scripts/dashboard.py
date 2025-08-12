import pandas as pd
import joblib
import xgboost as xgb
import plotly.express as px
from dash import Dash, html, dcc

# Load data & model
df = pd.read_csv("data/hourly_features.csv")

model = xgb.XGBRegressor()
model.load_model("models/aqi_model.json")

# Make predictions
if "us_aqi" in df.columns:
    df["predicted_aqi"] = model.predict(df.drop(columns=["us_aqi"], errors="ignore"))
else:
    df["predicted_aqi"] = model.predict(df)

# Plot
fig = px.line(df, x="timestamp", y=["us_aqi", "predicted_aqi"],
              labels={"value": "AQI", "variable": "Type"},
              title="Actual vs Predicted AQI")

# Dash layout
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Air Quality Dashboard"),
    dcc.Graph(figure=fig)
])

# Save static HTML for GitHub Pages
fig.write_html("docs/index.html", include_plotlyjs="cdn")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
