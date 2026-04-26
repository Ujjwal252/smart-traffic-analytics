# Smart Traffic Analytics

A real-time traffic analytics and prediction system that processes traffic data, identifies congestion hotspots, and provides predictive insights using machine learning.

## What the System Does

- **Traffic Data Processing**: Analyzes large-scale traffic flow data to extract meaningful patterns
- **Congestion Hotspot Detection**: Identifies areas with recurring traffic congestion
- **Predictive Analytics**: Uses XGBoost machine learning models to predict traffic conditions
- **Interactive Visualizations**: Provides interactive maps and charts via Streamlit dashboard
- **Weather Integration**: Incorporates weather data to improve predictions

## Tech Stack

- **Python 3.12** - Core programming language
- **Pandas & NumPy** - Data manipulation and numerical computing
- **Scikit-learn** - Machine learning utilities
- **XGBoost** - Gradient boosting for traffic prediction
- **Folium & Streamlit-Folium** - Interactive map visualizations
- **Plotly** - Interactive charts
- **Streamlit** - Web dashboard framework
- **Open-Meteo API** - Weather data integration

## How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard
```bash
streamlit run app/main.py
```

### 3. Access the App
Open your browser and navigate to `http://localhost:8501`

## Folder Structure

```
smart-traffic-analytics/
├── app/
│   ├── __init__.py
│   ├── config.py           # Configuration settings
│   ├── main.py             # Streamlit app entry point
│   ├── tabs/               # Dashboard tab modules
│   │   ├── __init__.py
│   │   ├── overview.py
│   │   ├── hotspots.py
│   │   ├── prediction.py
│   │   └── weather.py
│   └── utils/              # Utility functions
│       ├── __init__.py
│       ├── data_loader.py
│       ├── preprocessing.py
│       └── visualization.py
├── data/                   # Data files (CSV, Parquet, JSON)
│   └── charts/             # Generated chart exports
├── models/                 # Trained ML models (pickle files)
├── notebooks/              # Jupyter notebooks for analysis
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```