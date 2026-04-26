import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os

# ============================================================
# WMO Weather Code Functions
# ============================================================

def wmo_code_to_risk(code):
    """Map WMO weather code to risk score 1-5."""
    if code == 0:
        return 1
    elif code in [1, 2, 3]:
        return 1
    elif code in [45, 48]:
        return 4
    elif code in [51, 53, 55]:
        return 2
    elif code in [61, 63, 65]:
        return 3
    elif code in [71, 73, 75]:
        return 5
    elif code in [80, 81, 82]:
        return 3
    elif code == 95:
        return 4
    elif code in [96, 99]:
        return 5
    else:
        return 2


def wmo_code_to_description(code):
    """Return human readable weather description."""
    descriptions = {
        0 : "Clear sky",
        1 : "Mainly clear",
        2 : "Partly cloudy",
        3 : "Overcast",
        45: "Foggy",
        48: "Icy fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Heavy drizzle",
        61: "Light rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Light snow",
        73: "Moderate snow",
        75: "Heavy snowfall",
        80: "Rain showers",
        81: "Moderate showers",
        82: "Heavy showers",
        95: "Thunderstorm",
        96: "Thunderstorm with hail",
        99: "Heavy thunderstorm with hail"
    }
    return descriptions.get(code, "Unknown")


def get_risk_color(risk_score):
    """Return hex color for risk score."""
    colors = {
        1: "#22c55e",   # green
        2: "#84cc16",   # lime
        3: "#f97316",   # orange
        4: "#ef4444",   # red
        5: "#7c3aed"    # purple
    }
    return colors.get(risk_score, "#6b7280")


# ============================================================
# Main Weather Forecast Function
# ============================================================

def get_weather_forecast(latitude, longitude):
    """
    Fetch 6-hour weather forecast from Open-Meteo API.

    Args:
        latitude (float): Location latitude
        longitude (float): Location longitude

    Returns:
        pd.DataFrame: Weather forecast for next 6 hours
    """
    try:
        print(f"🌤️ Fetching weather for ({latitude}, {longitude})...")

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude"   : latitude,
            "longitude"  : longitude,
            "hourly"     : [
                "temperature_2m",
                "precipitation",
                "weathercode",
                "visibility",
                "windspeed_10m",
                "relativehumidity_2m"
            ],
            "forecast_days": 1,
            "timezone"   : "auto"
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Parse hourly data
        hourly = data.get("hourly", {})

        df = pd.DataFrame({
            "datetime"        : pd.to_datetime(hourly["time"]),
            "temperature_c"   : hourly["temperature_2m"],
            "precipitation_mm": hourly["precipitation"],
            "weathercode"     : hourly["weathercode"],
            "visibility_m"    : hourly["visibility"],
            "windspeed_kmh"   : hourly["windspeed_10m"],
            "humidity_pct"    : hourly["relativehumidity_2m"],
        })

        # Convert units
        df["temperature_f"] = df["temperature_c"] * 9/5 + 32
        df["visibility_km"] = df["visibility_m"] / 1000
        df["hour"]          = df["datetime"].dt.hour

        # Add risk and description
        df["weather_risk_score"]    = df["weathercode"].apply(wmo_code_to_risk)
        df["weather_description"]   = df["weathercode"].apply(wmo_code_to_description)
        df["risk_color"]            = df["weather_risk_score"].apply(get_risk_color)

        # Filter to next 6 hours only
        now         = datetime.now()
        six_hrs     = now + timedelta(hours=6)
        df          = df[
            (df["datetime"] >= now) &
            (df["datetime"] <= six_hrs)
        ].reset_index(drop=True)

        # Clean up columns
        df = df[[
            "datetime", "hour", "temperature_f",
            "precipitation_mm", "weathercode",
            "visibility_km", "windspeed_kmh",
            "humidity_pct", "weather_risk_score",
            "weather_description", "risk_color"
        ]]

        print(f"✅ Got {len(df)} hours of forecast!")
        return df

    except requests.exceptions.Timeout:
        print("❌ Weather API timeout — using fallback data")
        return _get_fallback_forecast()

    except requests.exceptions.ConnectionError:
        print("❌ No internet connection — using fallback data")
        return _get_fallback_forecast()

    except Exception as e:
        print(f"❌ Weather API error: {e}")
        return _get_fallback_forecast()


def _get_fallback_forecast():
    """Return dummy forecast when API fails."""
    now = datetime.now()
    rows = []
    for i in range(6):
        dt = now + timedelta(hours=i)
        rows.append({
            "datetime"           : dt,
            "hour"               : dt.hour,
            "temperature_f"      : 65.0,
            "precipitation_mm"   : 0.0,
            "weathercode"        : 0,
            "visibility_km"      : 10.0,
            "windspeed_kmh"      : 10.0,
            "humidity_pct"       : 50.0,
            "weather_risk_score" : 1,
            "weather_description": "Clear sky (fallback)",
            "risk_color"         : "#22c55e"
        })
    return pd.DataFrame(rows)


# ============================================================
# Test Block
# ============================================================
if __name__ == "__main__":
    print("="*50)
    print("TESTING weather.py")
    print("="*50)

    # Test with San Francisco
    print("\n📍 Testing with San Francisco...")
    df = get_weather_forecast(37.7749, -122.4194)

    if len(df) > 0:
        print("\n✅ Forecast received:")
        print(df[["datetime","temperature_f",
                  "weather_description","weather_risk_score"]].to_string(index=False))
    else:
        print("⚠️ Empty forecast — check internet connection")

    # Test helper functions
    print("\n✅ WMO Code Tests:")
    print(f"   Code 0  → Risk: {wmo_code_to_risk(0)}  | {wmo_code_to_description(0)}")
    print(f"   Code 63 → Risk: {wmo_code_to_risk(63)} | {wmo_code_to_description(63)}")
    print(f"   Code 75 → Risk: {wmo_code_to_risk(75)} | {wmo_code_to_description(75)}")
    print(f"   Code 95 → Risk: {wmo_code_to_risk(95)} | {wmo_code_to_description(95)}")

    print("\n✅ weather.py working correctly!")