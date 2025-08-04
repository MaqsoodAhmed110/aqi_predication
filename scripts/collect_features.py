import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
import numpy as np
from datetime import datetime

def collect_features():
    # Setup API client
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Fetch air quality data (example for Karachi)
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": 24.8608,
        "longitude": 67.0104,
        "hourly": ["pm10", "carbon_monoxide", "sulphur_dioxide", "ozone", 
                  "aerosol_optical_depth", "dust"],
        "timezone": "auto"
    }
    
    response = openmeteo.weather_api(url, params=params)[0]
    hourly = response.Hourly()
    
    # Process data
    data = {
        "timestamp": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "pm10": hourly.Variables(0).ValuesAsNumpy(),
        "carbon_monoxide": hourly.Variables(1).ValuesAsNumpy(),
        "sulphur_dioxide": hourly.Variables(2).ValuesAsNumpy(),
        "ozone": hourly.Variables(3).ValuesAsNumpy(),
        "aerosol_optical_depth": hourly.Variables(4).ValuesAsNumpy(),
        "dust": hourly.Variables(5).ValuesAsNumpy(),
    }
    
    df = pd.DataFrame(data)
    df.to_csv("data/hourly_features.csv", index=False)

if __name__ == "__main__":
    collect_features()
