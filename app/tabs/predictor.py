"""
Risk Predictor Tab - ML-based accident risk prediction.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import build_input_features, predict_risk


def render_predictor_tab():
    """Render the Accident Risk Predictor tab."""
    st.header("🔮 Accident Risk Predictor")
    st.markdown("Predict accident risk based on time, weather, and road conditions.")
    
    # Input layout - two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Time & Road Features")
        hour = st.slider("Hour of Day", min_value=0, max_value=23, value=8)
        day_of_week = st.selectbox(
            "Day of Week",
            options=[
                (0, "Monday"), (1, "Tuesday"), (2, "Wednesday"), (3, "Thursday"),
                (4, "Friday"), (5, "Saturday"), (6, "Sunday")
            ],
            format_func=lambda x: x[1],
            index=0
        )
        day_of_week = day_of_week[0]
        
        month = st.selectbox(
            "Month",
            options=[
                (1, "January"), (2, "February"), (3, "March"), (4, "April"),
                (5, "May"), (6, "June"), (7, "July"), (8, "August"),
                (9, "September"), (10, "October"), (11, "November"), (12, "December")
            ],
            format_func=lambda x: x[1],
            index=3
        )
        month = month[0]
        
        road_features = st.multiselect(
            "Road Features",
            options=["Junction", "Traffic Signal", "Crossing", "Railway"],
            default=["Traffic Signal"]
        )
        
    with col2:
        st.subheader("Weather & Environment")
        weather_risk = st.selectbox(
            "Weather Condition",
            options=[
                (1, "Clear/Cloudy"), (2, "Light Precipitation"), (3, "Moderate Rain"),
                (4, "Heavy Rain/Fog"), (5, "Storm/Snow")
            ],
            format_func=lambda x: x[1],
            index=0
        )
        weather_risk = weather_risk[0]
        
        visibility = st.select_slider(
            "Visibility",
            options=["Low", "Medium", "High"],
            value="Medium"
        )
        visibility_bucket = {"Low": 0, "Medium": 1, "High": 2}[visibility]
        
        temperature = st.select_slider(
            "Temperature",
            options=["Freezing (<0°C)", "Cool (0-15°C)", "Warm (15-25°C)", "Hot (>25°C)"],
            value="Cool (0-15°C)"
        )
        temp_bucket = {"Freezing (<0°C)": 0, "Cool (0-15°C)": 1, "Warm (15-25°C)": 2, "Hot (>25°C)": 3}[temperature]
    
    # Predict button
    predict_btn = st.button("Predict Risk", type="primary")
    
    if predict_btn:
        with st.spinner("Analyzing risk factors..."):
            try:
                # Build input features
                is_junction = "Junction" in road_features
                is_traffic_signal = "Traffic Signal" in road_features
                is_crossing = "Crossing" in road_features
                
                input_features = build_input_features(
                    hour=hour,
                    day_of_week=day_of_week,
                    month=month,
                    weather_risk_score=weather_risk,
                    visibility_bucket=visibility_bucket,
                    temp_bucket=temp_bucket,
                    is_junction=is_junction,
                    is_traffic_signal=is_traffic_signal,
                    is_crossing=is_crossing
                )
                
                # Get prediction
                result = predict_risk(input_features)
                
                if result:
                    risk_prob = result.get('risk_probability', 0.5)
                    risk_label = result.get('risk_label', 'Medium')
                    
                    # Determine color and label
                    if risk_prob >= 0.7:
                        color = "#dc2626"  # red
                        emoji = "🔴"
                        label = "HIGH RISK"
                    elif risk_prob >= 0.4:
                        color = "#f59e0b"  # orange
                        emoji = "🟠"
                        label = "MODERATE RISK"
                    else:
                        color = "#16a34a"  # green
                        emoji = "🟢"
                        label = "LOW RISK"
                    
                    # Large colored badge
                    st.markdown(
                        f"<div style='background-color:{color}; padding:20px; border-radius:10px; "
                        f"color:white; text-align:center; font-size:24px; font-weight:bold; margin:20px 0;'>"
                        f"{emoji} {label} - {risk_prob*100:.1f}% Probability"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    
                    # Risk probability metric
                    st.metric("Risk Probability", f"{risk_prob*100:.1f}%")
                    
                    # Plotly gauge chart
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = risk_prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Accident Risk Score"},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': color},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 40], 'color': "#dcfce7"},
                                {'range': [40, 70], 'color': "#fef3c7"},
                                {'range': [70, 100], 'color': "#fee2e2"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': risk_prob * 100
                            }
                        }
                    ))
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(l=20, r=20, t=50, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Human readable explanation
                    st.subheader("📋 Analysis")
                    
                    # Build explanation
                    factors = []
                    if hour in [7, 8, 9, 17, 18, 19]:
                        factors.append("peak commute hours")
                    if hour < 6 or hour >= 22:
                        factors.append("nighttime driving")
                    if weather_risk >= 3:
                        factors.append("adverse weather conditions")
                    if visibility == "Low":
                        factors.append("poor visibility")
                    if "Junction" in road_features:
                        factors.append("junction location")
                    if "Traffic Signal" in road_features:
                        factors.append("traffic signal intersection")
                    if "Crossing" in road_features:
                        factors.append("pedestrian crossing")
                    if day_of_week >= 5:
                        factors.append("weekend traffic")
                    
                    if factors:
                        explanation = f"Risk factors identified: {', '.join(factors)}."
                    else:
                        explanation = "Standard driving conditions with no major risk factors."
                    
                    st.write(explanation)
                    
                    # Recommendation
                    st.subheader("💡 Recommendation")
                    if risk_prob >= 0.7:
                        recommendation = """
                        **High Risk Alert**: Consider postponing non-essential travel. 
                        If travel is necessary, exercise extreme caution, reduce speed, 
                        and maintain increased following distance. Emergency services 
                        should be on standby.
                        """
                    elif risk_prob >= 0.4:
                        recommendation = """
                        **Moderate Risk**: Drive with caution. Be aware of increased 
                        traffic and potential weather impacts. Consider allowing 
                        extra travel time.
                        """
                    else:
                        recommendation = """
                        **Low Risk**: Normal driving conditions. Continue to follow 
                        safe driving practices and stay alert.
                        """
                    
                    st.info(recommendation)
                else:
                    st.error("Prediction failed. Please try again.")
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    
    # Show placeholder when not predicted
    if not predict_btn:
        st.info("👆 Adjust the parameters above and click 'Predict Risk' to get an accident risk assessment.")
