# 🌍 AQI Prediction Dashboard  

This repository provides a **Machine Learning-based Air Quality Prediction System** using **XGBoost** and **Gradio**.  
It fetches **real-time air quality data** from the [Open-Meteo Air Quality API](https://open-meteo.com/) and predicts the **US AQI (Air Quality Index)** for the next **3 days** based on pollutant concentrations.  

---

## Overview
This repository contains an end-to-end Air Quality Index (AQI) prediction system featuring:
- Automated CI/CD pipeline for continuous data collection and model updates
- Local development environment with Jupyter notebooks
- Interactive Gradio and Streamlit dashboards
- XGBoost-based machine learning models
---

## 📂 Project Structure
- ├── .github/workflows/ # CI/CD pipeline definitions
- │ ├── features.yml # Hourly data collection
- │ ├── training.yml # Model retraining
- │ ├── aqi-dashboard.yml # Dashboard deployment
- │ └── static.yml # Fallback static site
- ├── data/ # Processed AQI datasets
- ├── models/ # Trained model binaries
- ├── scripts/ # Core pipeline scripts
- │ ├── collect_features.py # Data ingestion
- │ ├── train_model.py # ML training
- │ ├── dashboard.py # Interactive dashboard
- │ └── dashboard_static.py # Static version
- ├── AQI_predication_1.ipynb # Exploratory analysis and ML models
- ├── gradio_aqi_code.ipynb # Gradio UI prototype
- ├── requirements.txt # Python dependencies
- └── Air Quality Index.pdf # Project documentation
