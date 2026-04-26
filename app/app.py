"""
Smart Traffic Analytics - Main Streamlit Application
"""
import streamlit as st
import sys
import os

# Path fix
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Page config — FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Smart Traffic Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Sidebar
    with st.sidebar:
        st.title("🚦 Smart Traffic Analytics")
        st.markdown("""
        Real-time accident risk prediction and emergency
        dispatch optimization powered by ML.
        """)
        st.info("🚨 Pre-positions emergency units based on predicted accident risk")
        st.markdown("---")
        st.markdown("**Built with:**")
        st.markdown("- Pandas + XGBoost")
        st.markdown("- Folium + Streamlit")
        st.markdown("---")
        st.caption("Data: US Accidents Dataset (2016-2023)")

    # Main Tabs
    tabs = st.tabs([
        "🗺️ Live Risk Map",
        "🚑 Dispatcher Panel",
        "🔮 Risk Predictor",
        "📊 Insights"
    ])

    # Tab 1 — Live Map
    with tabs[0]:
        try:
            from tabs.live_map import render_live_map_tab
            render_live_map_tab()
        except Exception as e:
            st.error(f"❌ Live Map error: {e}")
            st.info("Make sure hotspots.json exists in data/ folder")

    # Tab 2 — Dispatcher
    with tabs[1]:
        try:
            from tabs.dispatcher import render_dispatcher_tab
            render_dispatcher_tab()
        except Exception as e:
            st.error(f"❌ Dispatcher error: {e}")

    # Tab 3 — Predictor
    with tabs[2]:
        try:
            from tabs.predictor import render_predictor_tab
            render_predictor_tab()
        except Exception as e:
            st.error(f"❌ Predictor error: {e}")
            st.info("Make sure models are trained and saved in models/ folder")

    # Tab 4 — Insights
    with tabs[3]:
        try:
            from tabs.insights import render_insights_tab
            render_insights_tab()
        except Exception as e:
            st.error(f"❌ Insights error: {e}")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:gray; padding:10px'>"
        "📊 US Accidents Dataset | 🤖 XGBoost | 🚦 Smart Traffic Analytics"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()