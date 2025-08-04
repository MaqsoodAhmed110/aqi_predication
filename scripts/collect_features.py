import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime

def calculate_pm25_aqi(pm25):
    """Calculate AQI based on PM2.5 concentration"""
    if pm25 <= 12.0:
        return ((50 - 0) / (12.0 - 0.0)) * (pm25 - 0.0) + 0
    elif pm25 <= 35.4:
        return ((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51
    elif pm25 <= 55.4:
        return ((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101
    elif pm25 <= 150.4:
        return ((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151
    elif pm25 <= 250.4:
        return ((300 - 201) / (250.4 - 150.5)) * (pm25 - 150.5) + 201
    elif pm25 <= 350.4:
        return ((400 - 301) / (350.4 - 250.5)) * (pm25 - 250.5) + 301
    elif pm25 <= 500.4:
        return ((500 - 401) / (500.4 - 350.5)) * (pm25 - 350.5) + 401
    else:
        return 500

def calculate_pm10_aqi(pm10):
    """Calculate AQI based on PM10 concentration"""
    if pm10 <= 54:
        return ((50 - 0) / (54 - 0)) * (pm10 - 0) + 0
    elif pm10 <= 154:
        return ((100 - 51) / (154 - 55)) * (pm10 - 55) + 51
    elif pm10 <= 254:
        return ((150 - 101) / (254 - 155)) * (pm10 - 155) + 101
    elif pm10 <= 354:
        return ((200 - 151) / (354 - 255)) * (pm10 - 255) + 151
    elif pm10 <= 424:
        return ((300 - 201) / (424 - 355)) * (pm10 - 355) + 201
    elif pm10 <= 504:
        return ((400 - 301) / (504 - 425)) * (pm10 - 425) + 301
    elif pm10 <= 604:
        return ((500 - 401) / (604 - 505)) * (pm10 - 505) + 401
    else:
        return 500

def calculate_aqi(row):
    """Calculate overall US AQI"""
    pm25_aqi = calculate_pm25_aqi(row['pm2_5'])
    pm10_aqi = calculate_pm10_aqi(row['pm10'])
    return max(pm25_aqi, pm10_aqi)

def collect_features():
    # Setup API client
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # API parameters
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": 24.8608,
        "longitude": 67.0104,
        "hourly": [
            "pm10", "pm2_5", "carbon_monoxide", "carbon_dioxide",
            "nitrogen_dioxide", "sulphur_dioxide", "ozone",
            "aerosol_optical_depth", "dust", "uv_index",
            "uv_index_clear_sky", "ammonia", "methane"
        ],
        "timezone": "auto",
        "past_days": 1,
        "forecast_days": 1
    }
    
    # Get data
    response = openmeteo.weather_api(url, params=params)[0]
    hourly = response.Hourly()
    
    # Process data
    hourly_data = {
        "timestamp": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "pm10": hourly.Variables(0).ValuesAsNumpy(),
        "pm2_5": hourly.Variables(1).ValuesAsNumpy(),
        "carbon_monoxide": hourly.Variables(2).ValuesAsNumpy(),
        "carbon_dioxide": hourly.Variables(3).ValuesAsNumpy(),
        "nitrogen_dioxide": hourly.Variables(4).ValuesAsNumpy(),
        "sulphur_dioxide": hourly.Variables(5).ValuesAsNumpy(),
        "ozone": hourly.Variables(6).ValuesAsNumpy(),
        "aerosol_optical_depth": hourly.Variables(7).ValuesAsNumpy(),
        "dust": hourly.Variables(8).ValuesAsNumpy(),
        "uv_index": hourly.Variables(9).ValuesAsNumpy(),
        "uv_index_clear_sky": hourly.Variables(10).ValuesAsNumpy(),
        "ammonia": hourly.Variables(11).ValuesAsNumpy(),
        "methane": hourly.Variables(12).ValuesAsNumpy()
    }
    
    df = pd.DataFrame(hourly_data)
    
    # Calculate AQI
    df['us_aqi'] = df.apply(calculate_aqi, axis=1)
    df['us_aqi'] = df['us_aqi'].round().astype(int)
    
    # Save data
    df.to_csv("data/hourly_features.csv", index=False)
    print(f"Data collected at {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Records collected: {len(df)}")
    print(f"AQI range: {df['us_aqi'].min()} to {df['us_aqi'].max()}")

if __name__ == "__main__":
    collect_features()
