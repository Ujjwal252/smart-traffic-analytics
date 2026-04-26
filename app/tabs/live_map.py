"""
Live Risk Map Tab - Display accident hotspots on an interactive Folium map
with weather data and risk visualization.
"""
import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import sys
import os

# Path fix
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.weather import get_weather_forecast
from utils.optimizer import load_hotspots

# ============================================================
# US Cities
# ============================================================
CITIES = {
    "Los Angeles, CA"   : {"lat": 34.0522,  "lng": -118.2437},
    "San Francisco, CA" : {"lat": 37.7749,  "lng": -122.4194},
    "New York, NY"      : {"lat": 40.7128,  "lng": -74.0060},
    "Chicago, IL"       : {"lat": 41.8781,  "lng": -87.6298},
    "Houston, TX"       : {"lat": 29.7604,  "lng": -95.3698},
    "Phoenix, AZ"       : {"lat": 33.4484,  "lng": -112.0740},
    "Philadelphia, PA"  : {"lat": 39.9526,  "lng": -75.1652},
    "San Antonio, TX"   : {"lat": 29.4241,  "lng": -98.4936},
    "San Diego, CA"     : {"lat": 32.7157,  "lng": -117.1611},
    "Dallas, TX"        : {"lat": 32.7767,  "lng": -96.7970},
}


def get_lat_lng_cols(df):
    """Auto detect lat/lng column names."""
    lat_col = None
    lng_col = None

    for col in df.columns:
        if col.lower() in ['centroid_lat', 'lat', 'latitude', 'start_lat']:
            lat_col = col
        if col.lower() in ['centroid_lng', 'lng', 'longitude', 'start_lng']:
            lng_col = col

    return lat_col, lng_col


def render_live_map_tab():
    """Render the Live Risk Map tab."""
    st.header("🗺️ Live Risk Map")
    st.markdown("View real-time accident hotspots and weather-based risk assessment.")

    # City selector
    city_name   = st.selectbox("Select City", options=list(CITIES.keys()), index=0)
    city_coords = CITIES[city_name]
    lat         = city_coords["lat"]
    lng         = city_coords["lng"]

    current_risk = 2  # default

    # ============================================================
    # Weather Forecast
    # ============================================================
    st.subheader("🌤️ Weather Forecast (Next 6 Hours)")

    try:
        with st.spinner("Fetching weather data..."):
            weather_df = get_weather_forecast(lat, lng)

        if weather_df is not None and not weather_df.empty:
            # Show only available columns
            display_cols = [
                'datetime', 'temperature_f', 'weather_description',
                'weather_risk_score', 'humidity_pct', 'windspeed_kmh'
            ]
            available_cols = [c for c in display_cols if c in weather_df.columns]
            st.dataframe(
                weather_df[available_cols].head(6),
                use_container_width=True,
                hide_index=True
            )

            # Current risk
            current_risk = int(weather_df.iloc[0].get('weather_risk_score', 2))

            # Risk badge
            risk_colors = {
                1: "#22c55e", 2: "#84cc16",
                3: "#f97316", 4: "#ef4444", 5: "#7c3aed"
            }
            risk_labels = {
                1: "Low", 2: "Low-Medium",
                3: "Medium", 4: "High", 5: "Extreme"
            }
            color = risk_colors.get(current_risk, "#6b7280")
            label = risk_labels.get(current_risk, "Unknown")

            st.markdown(
                f"<div style='background:{color}; padding:10px; border-radius:8px;"
                f"color:white; text-align:center; font-weight:bold; margin:10px 0;'>"
                f"Current Weather Risk: {label} ({current_risk}/5)"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.warning("⚠️ Weather data unavailable — using default risk level.")

    except Exception as e:
        st.warning(f"⚠️ Weather fetch failed: {str(e)} — using default risk level.")

    # ============================================================
    # Load Hotspots
    # ============================================================
    try:
        with st.spinner("Loading hotspot data..."):
            hotspots_df = load_hotspots()

        if hotspots_df is None or hotspots_df.empty:
            st.warning("⚠️ No hotspot data available. Run notebook 04_clustering.ipynb first.")
            return

        # Auto detect lat/lng columns
        lat_col, lng_col = get_lat_lng_cols(hotspots_df)

        if lat_col is None or lng_col is None:
            st.error(f"❌ Could not find lat/lng columns. Available: {hotspots_df.columns.tolist()}")
            return

        # Filter nearby hotspots
        nearby = hotspots_df[
            (hotspots_df[lat_col].between(lat - 3, lat + 3)) &
            (hotspots_df[lng_col].between(lng - 3, lng + 3))
        ].copy()

        if nearby.empty:
            st.info(f"ℹ️ No hotspots found near {city_name}. Try another city.")
            # Still show empty map
            m = folium.Map(location=[lat, lng], zoom_start=10, tiles="cartodbpositron")
            folium.Marker(
                location=[lat, lng],
                popup=f"<b>{city_name}</b>",
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(m)
            folium_static(m, width=900, height=500)
            return

        # ============================================================
        # Build Folium Map
        # ============================================================
        m = folium.Map(
            location=[lat, lng],
            zoom_start=10,
            tiles="cartodbpositron"
        )

        # Heatmap layer
        if 'risk_score' in nearby.columns:
            heat_data = [
                [row[lat_col], row[lng_col], row['risk_score']]
                for _, row in nearby.iterrows()
            ]
            HeatMap(heat_data, radius=20, blur=15).add_to(m)

        # Circle markers
        for _, row in nearby.iterrows():
            risk           = float(row.get('risk_score', 0.5))
            accident_count = int(row.get('accident_count', 0))
            avg_severity   = float(row.get('avg_severity', 0))

            color  = "red" if risk >= 0.7 else "orange" if risk >= 0.4 else "green"
            radius = 8 + risk * 15

            popup_html = f"""
            <b>Accident Hotspot</b><br>
            Accidents: {accident_count:,}<br>
            Avg Severity: {avg_severity:.2f}<br>
            Risk Score: {risk:.3f}
            """

            folium.CircleMarker(
                location=[row[lat_col], row[lng_col]],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=folium.Popup(popup_html, max_width=200)
            ).add_to(m)

        # City center marker
        folium.Marker(
            location=[lat, lng],
            popup=f"<b>{city_name}</b><br>City Center",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)

        # Legend
        legend_html = """
        <div style="position:fixed; bottom:50px; left:50px; width:160px;
                    border:2px solid grey; z-index:9999; font-size:13px;
                    background-color:white; padding:10px; border-radius:8px;">
        <b>Risk Legend</b><br>
        <i style="background:red;width:12px;height:12px;
                  display:inline-block;border-radius:50%;"></i> High Risk<br>
        <i style="background:orange;width:12px;height:12px;
                  display:inline-block;border-radius:50%;"></i> Medium Risk<br>
        <i style="background:green;width:12px;height:12px;
                  display:inline-block;border-radius:50%;"></i> Low Risk
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        # Render map
        st.subheader("🗺️ Accident Hotspot Map")
        folium_static(m, width=900, height=500)

        # ============================================================
        # Summary Metrics
        # ============================================================
        st.subheader("📊 Summary")
        col1, col2, col3, col4 = st.columns(4)

        high_risk_count = len(nearby[nearby['risk_score'] >= 0.7]) \
                          if 'risk_score' in nearby.columns else 0

        risk_labels = {
            1: "Low", 2: "Low-Medium",
            3: "Medium", 4: "High", 5: "Extreme"
        }

        with col1:
            st.metric("Zones Monitored", len(nearby))
        with col2:
            st.metric("High-Risk Zones", high_risk_count)
        with col3:
            st.metric("Weather Risk", risk_labels.get(current_risk, "Unknown"))
        with col4:
            readiness = "Critical" if current_risk >= 4 else \
                        "High" if current_risk >= 3 else \
                        "Medium" if current_risk >= 2 else "Low"
            st.metric("Deployment Readiness", readiness)

        st.info("ℹ️ Map updates every time you select a new city.")

    except Exception as e:
        st.error(f"❌ Error loading hotspot data: {str(e)}")
        st.info("Make sure hotspots.json exists in the data/ folder.")