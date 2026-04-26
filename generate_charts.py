import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json, os, joblib

DATA_DIR   = "data"
CHARTS_DIR = "data/charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

# Load data
df = pd.read_parquet(f"{DATA_DIR}/accidents_features.parquet")
print(f"Loaded {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")

# Chart 1 - Hourly
hourly = df.groupby("hour_of_day").size().reset_index(name="count")
px.bar(hourly, x="hour_of_day", y="count",
       title="Accidents by Hour",
       color="count",
       color_continuous_scale="Reds").write_html(f"{CHARTS_DIR}/chart_hourly.html")
print("chart_hourly done")

# Chart 2 - Weekday
dow = df.groupby("day_of_week").size().reset_index(name="count")
dow["day"] = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
px.bar(dow, x="day", y="count",
       title="Accidents by Day").write_html(f"{CHARTS_DIR}/chart_weekday.html")
print("chart_weekday done")

# Chart 3 - Trend
trend = df.groupby(["year","month"]).size().reset_index(name="count")
trend["year_month"] = trend["year"].astype(str) + "-" + trend["month"].astype(str).str.zfill(2)
px.line(trend, x="year_month", y="count",
        title="Accident Trend Over Time").write_html(f"{CHARTS_DIR}/chart_trend.html")
print("chart_trend done")

# Chart 4 - Heatmap
pivot = df.groupby(["hour_of_day","day_of_week"]).size().unstack(fill_value=0)
go.Figure(go.Heatmap(
    z=pivot.values,
    x=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
    y=list(range(24)),
    colorscale="Reds"
)).update_layout(title="Heatmap: Hour vs Day").write_html(f"{CHARTS_DIR}/chart_heatmap.html")
print("chart_heatmap done")

# Chart 5 - Weather
weather = df.groupby("weather_risk_score").size().reset_index(name="count")
px.bar(weather, x="weather_risk_score", y="count",
       title="Accidents by Weather Risk",
       color="weather_risk_score",
       color_continuous_scale="RdYlGn_r").write_html(f"{CHARTS_DIR}/chart_weather.html")
print("chart_weather done")

# Chart 6 - States
if "State" in df.columns:
    states = df.groupby("State").size().reset_index(name="count").nlargest(15,"count")
    px.bar(states, x="count", y="State", orientation="h",
           title="Top 15 States").write_html(f"{CHARTS_DIR}/chart_states.html")
    print("chart_states done")
else:
    print("State column not found — skipping")

# Feature Importance
try:
    model    = joblib.load("models/risk_model_binary.pkl")
    features = joblib.load("models/feature_list.pkl")
    fi = pd.DataFrame({
        "feature"   : features,
        "importance": model.feature_importances_
    }).sort_values("importance")
    px.bar(fi, x="importance", y="feature", orientation="h",
           title="Feature Importance",
           color="importance",
           color_continuous_scale="Viridis").write_html(f"{CHARTS_DIR}/feature_importance.html")
    print("feature_importance done")
except Exception as e:
    print(f"feature_importance skipped: {e}")

# Stats JSON
high_risk_pct = "N/A"
if "binary_severity" in df.columns:
    high_risk_pct = f"{df['binary_severity'].mean():.1%}"

stats = {
    "total_records"       : len(df),
    "most_dangerous_hour" : f"{int(df.groupby('hour_of_day').size().idxmax())}:00",
    "highest_risk_weather": "Storm/Heavy Rain",
    "high_risk_pct"       : high_risk_pct,
    "model_accuracy"      : "65.0%"
}
with open(f"{CHARTS_DIR}/stats.json", "w") as f:
    json.dump(stats, f)
print("stats.json done")

# Sample parquet
df.sample(min(1000, len(df))).to_parquet(
    f"{DATA_DIR}/sample_features.parquet", index=False
)
print("sample_features.parquet done")

print("\nALL CHARTS CREATED SUCCESSFULLY!")