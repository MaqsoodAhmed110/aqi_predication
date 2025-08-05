import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
from datetime import datetime

def train_model():
    # Load data
    df = pd.read_csv("data/hourly_features.csv")
    
    # Feature selection
    features = ['pm10', 'carbon_monoxide', 'sulphur_dioxide', 'dust', 
               'aerosol_optical_depth', 'ozone']
    
    # Prepare data
    X = df[features]
    y = df['us_aqi']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train XGBoost model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model trained at {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Save model
    model.save_model("models/aqi_model.json")
    joblib.dump(features, "models/features.pkl")

if __name__ == "__main__":
    train_model()
