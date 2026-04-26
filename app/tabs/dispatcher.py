"""
Dispatcher Panel Tab - Emergency unit deployment optimization.
"""
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from datetime import datetime
import sys
import os

# Path fix
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.weather import get_weather_forecast
from utils.optimizer import load_hotspots, get_deployment_plan, estimate_coverage

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


def _get_col(df, candidates):
    """Helper to find first matching column name."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def render_dispatcher_tab():
    """Render the Emergency Dispatcher Panel tab."""
    st.header("🚑 Emergency Dispatcher Panel")
    st.markdown("Optimize emergency unit deployment based on real-time risk assessment.")

    current_time = datetime.now()

    # ============================================================
    # Controls
    # ============================================================
    col1, col2 = st.columns([1, 1])

    with col1:
        n_units = st.slider(
            "Number of Units",
            min_value=1, max_value=20, value=5,
            key="dispatcher_units"          # unique key fix
        )
        city_name = st.selectbox(
            "Select City",
            options=list(CITIES.keys()),
            index=0,
            key="dispatcher_city"           # unique key fix
        )
        generate_btn = st.button(
            "🎯 Generate Deployment Plan",
            type="primary",
            key="dispatcher_generate"
        )

    with col2:
        st.metric("Current Time", current_time.strftime("%H:%M"))

        # Weather risk
        try:
            city_coords      = CITIES[city_name]
            weather_data     = get_weather_forecast(city_coords["lat"], city_coords["lng"])
            if weather_data is not None and not weather_data.empty:
                current_weather_risk = int(weather_data.iloc[0].get('weather_risk_score', 2))
                st.metric("Weather Risk", f"{current_weather_risk}/5")
            else:
                current_weather_risk = 2
                st.metric("Weather Risk", "N/A")
        except Exception:
            current_weather_risk = 2
            st.metric("Weather Risk", "N/A")

        st.metric("Population at Risk", f"{n_units * 12500:,}")

    # ============================================================
    # Generate Plan
    # ============================================================
    if generate_btn:
        with st.spinner("⏳ Generating optimized deployment plan..."):
            try:
                city_coords  = CITIES[city_name]
                lat          = city_coords["lat"]
                lng          = city_coords["lng"]
                current_hour = current_time.hour

                # Fetch weather
                try:
                    weather_data = get_weather_forecast(lat, lng)
                except Exception:
                    weather_data = None

                # Load hotspots
                hotspots_df = load_hotspots()

                if hotspots_df is None or hotspots_df.empty:
                    st.error("❌ No hotspot data available. Run notebook 04_clustering.ipynb first.")
                    return

                # Auto detect lat/lng columns
                lat_col = _get_col(hotspots_df, ['centroid_lat', 'lat', 'latitude'])
                lng_col = _get_col(hotspots_df, ['centroid_lng', 'lng', 'longitude'])

                if lat_col is None or lng_col is None:
                    st.error(f"❌ Cannot find lat/lng columns. Found: {hotspots_df.columns.tolist()}")
                    return

                # Filter to city region
                city_hotspots = hotspots_df[
                    (hotspots_df[lat_col].between(lat - 3, lat + 3)) &
                    (hotspots_df[lng_col].between(lng - 3, lng + 3))
                ].copy()

                if city_hotspots.empty:
                    st.warning(f"⚠️ No hotspots found near {city_name}. Try another city.")
                    return

                # Get deployment plan
                deployment_plan = get_deployment_plan(
                    n_units      = n_units,
                    hotspots_df  = city_hotspots,
                    weather_df   = weather_data,
                    current_hour = current_hour
                )

                if not deployment_plan:
                    st.error("❌ Could not generate deployment plan.")
                    return

                # Coverage
                coverage = estimate_coverage(deployment_plan, city_hotspots)

                # ============================================================
                # Results
                # ============================================================
                st.success(f"✅ Deployment plan ready! {n_units} units covering {len(city_hotspots)} zones.")

                # Metrics row
                m1, m2, m3 = st.columns(3)
                m1.metric("Units Deployed",      n_units)
                m2.metric("Risk Coverage",        f"{coverage['coverage_pct']:.1f}%")
                m3.metric("Zones Covered",
                          f"{coverage['covered_zones']}/{coverage['total_zones']}")

                # Deployment table
                st.subheader("📋 Deployment Table")
                df_plan = pd.DataFrame(deployment_plan)
                df_plan = df_plan.rename(columns={
                    'unit_id'             : 'Unit',
                    'deploy_lat'          : 'Latitude',
                    'deploy_lng'          : 'Longitude',
                    'zone_risk'           : 'Risk Score',
                    'zone_accident_count' : 'Accidents',
                    'zone_avg_severity'   : 'Avg Severity',
                    'recommended_position': 'Position'
                })
                # Round floats
                df_plan['Risk Score']   = df_plan['Risk Score'].round(3)
                df_plan['Avg Severity'] = df_plan['Avg Severity'].round(2)
                st.dataframe(df_plan, use_container_width=True, hide_index=True)

                # ============================================================
                # Deployment Map
                # ============================================================
                st.subheader("🗺️ Deployment Map")
                m = folium.Map(
                    location=[lat, lng],
                    zoom_start=10,
                    tiles="cartodbpositron"
                )

                # Red circles = high risk zones
                for _, zone in city_hotspots.iterrows():
                    risk = float(zone.get('risk_score', 0.5))
                    if risk >= 0.4:
                        folium.Circle(
                            location=[zone[lat_col], zone[lng_col]],
                            radius=500,
                            color="red",
                            fill=True,
                            fill_color="red",
                            fill_opacity=0.3,
                            popup=f"Risk Zone<br>Accidents: {zone.get('accident_count',0)}"
                        ).add_to(m)

                # Blue markers = unit positions
                for unit in deployment_plan:
                    folium.Marker(
                        location=[unit['deploy_lat'], unit['deploy_lng']],
                        popup=(
                            f"<b>Unit {unit['unit_id']}</b><br>"
                            f"Risk: {unit['zone_risk']:.3f}<br>"
                            f"Accidents: {unit['zone_accident_count']}"
                        ),
                        icon=folium.Icon(color="blue", icon="plus-sign")
                    ).add_to(m)

                folium_static(m, width=900, height=400)

                # ============================================================
                # Time-based Advice
                # ============================================================
                st.subheader("⏰ Time-Based Recommendations")
                hour = current_time.hour

                if 6 <= hour < 9:
                    advice = "🌅 Morning rush (6-9 AM): High traffic. Position units near highways."
                elif 9 <= hour < 12:
                    advice = "🏙️ Late morning: Moderate risk. Focus on commercial districts."
                elif 12 <= hour < 14:
                    advice = "🍽️ Lunch hour: Increased pedestrian activity near restaurants."
                elif 14 <= hour < 17:
                    advice = "🚗 Afternoon: Risk building. Pre-position along major corridors."
                elif 17 <= hour < 20:
                    advice = "🔴 Evening rush (5-8 PM): PEAK RISK. Maximum coverage needed!"
                elif 20 <= hour < 23:
                    advice = "🌆 Evening: Focus on entertainment districts."
                else:
                    advice = "🌙 Night: Lower traffic but higher severity. Cover highway corridors."

                st.info(advice)

                # ============================================================
                # Export
                # ============================================================
                csv = pd.DataFrame(deployment_plan).to_csv(index=False)
                st.download_button(
                    label="📥 Export Plan as CSV",
                    data=csv,
                    file_name=f"deployment_{city_name.replace(', ','_')}.csv",
                    mime="text/csv",
                    key="dispatcher_download"
                )

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    else:
        st.info("👆 Select city and units, then click 'Generate Deployment Plan'")