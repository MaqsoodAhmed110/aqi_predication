import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from datetime import datetime

def train_model():
    # Load collected features
    df = pd.read_csv("data/hourly_features.csv")
    
    # Preprocess data
    # ... (your preprocessing code)
    
    # Train model
    X = df[['pm10', 'carbon_monoxide', ...]]  # Your features
    y = df['aqi']  # Your target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    
    # Save model
    model.save_model("models/aqi_model.json")

if __name__ == "__main__":
    train_model()
