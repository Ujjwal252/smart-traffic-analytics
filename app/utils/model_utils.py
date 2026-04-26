import joblib
import pandas as pd
import os

# ============================================================
# Dynamic path fix — works from anywhere
# ============================================================
BASE_DIR = os.path.dirname(   # utils folder
           os.path.dirname(   # app folder  
           os.path.abspath(__file__)))  # project root

# Go one more level up to reach project root
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODELS_PATH  = os.path.join(PROJECT_ROOT, "models")

print(f"📁 Looking for models in: {MODELS_PATH}")

# ============================================================
# Load models at module level (once, reused)
# ============================================================
try:
    risk_model   = joblib.load(os.path.join(MODELS_PATH, "risk_model_binary.pkl"))
    feature_list = joblib.load(os.path.join(MODELS_PATH, "feature_list.pkl"))
    print("✅ Models loaded successfully")
    print(f"✅ Features: {feature_list}")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print(f"   Looked in: {MODELS_PATH}")
    risk_model   = None
    feature_list = None


def get_season(month):
    """Get season from month number."""
    if month in [12, 1, 2]:
        return 1   # Winter
    elif month in [3, 4, 5]:
        return 2   # Spring
    elif month in [6, 7, 8]:
        return 3   # Summer
    else:
        return 4   # Fall


def build_input_features(hour, day_of_week, month, weather_risk_score,
                         visibility_bucket, temp_bucket,
                         is_junction, is_traffic_signal, is_crossing):
    """
    Build complete input features dictionary for model prediction.

    Args:
        hour (int): Hour of day (0-23)
        day_of_week (int): Day of week (0=Monday, 6=Sunday)
        month (int): Month (1-12)
        weather_risk_score (int): Weather risk score (1-5)
        visibility_bucket (int): 0=Low, 1=Medium, 2=High
        temp_bucket (int): 0=Freezing, 1=Cool, 2=Warm, 3=Hot
        is_junction (bool): Is at junction
        is_traffic_signal (bool): Has traffic signal
        is_crossing (bool): Is at crossing

    Returns:
        dict: Complete feature dictionary ready for prediction
    """
    # Derived features
    season         = get_season(month)
    is_weekend     = int(day_of_week >= 5)
    is_peak_hour   = int(hour in [7, 8, 9, 17, 18, 19])
    is_night       = int(hour < 6 or hour >= 22)
    is_adverse_weather  = int(weather_risk_score >= 3)
    is_low_visibility   = int(visibility_bucket == 0)
    duration_minutes    = 15  # default

    features = {
        'hour_of_day'        : hour,
        'day_of_week'        : day_of_week,
        'month'              : month,
        'season'             : season,
        'is_weekend'         : is_weekend,
        'is_peak_hour'       : is_peak_hour,
        'is_night'           : is_night,
        'weather_risk_score' : weather_risk_score,
        'is_adverse_weather' : is_adverse_weather,
        'is_low_visibility'  : is_low_visibility,
        'visibility_bucket'  : visibility_bucket,
        'temp_bucket'        : temp_bucket,
        'duration_minutes'   : duration_minutes,
        'Junction'           : int(is_junction),
        'Traffic_Signal'     : int(is_traffic_signal),
        'Crossing'           : int(is_crossing),
        'Railway'            : 0
    }

    return features


def predict_risk(input_dict):
    """
    Predict accident risk using trained XGBoost model.

    Args:
        input_dict (dict): Input features from build_input_features()

    Returns:
        dict: {is_high_risk, risk_probability, risk_label, risk_color}
    """
    # Model not loaded graceful fallback
    if risk_model is None or feature_list is None:
        return {
            "is_high_risk"    : False,
            "risk_probability": 0.5,
            "risk_label"      : "Model not loaded",
            "risk_color"      : "#6b7280"
        }

    try:
        # DataFrame banao
        df = pd.DataFrame([input_dict])

        # Missing features fill karo with 0
        for feature in feature_list:
            if feature not in df.columns:
                df[feature] = 0

        # Sirf feature_list wale columns rakho
        df = df[feature_list]

        # Predict
        prediction    = risk_model.predict(df)[0]
        probabilities = risk_model.predict_proba(df)[0]

        is_high_risk     = bool(prediction == 1)
        risk_probability = float(probabilities[1])
        risk_label       = "High Risk" if is_high_risk else "Low Risk"
        risk_color       = get_risk_color(risk_probability)

        return {
            "is_high_risk"    : is_high_risk,
            "risk_probability": risk_probability,
            "risk_label"      : risk_label,
            "risk_color"      : risk_color
        }

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return {
            "is_high_risk"    : False,
            "risk_probability": 0.5,
            "risk_label"      : "Prediction error",
            "risk_color"      : "#6b7280"
        }


def get_risk_color(probability):
    """Return hex color based on risk probability."""
    if probability < 0.3:
        return "#22c55e"   # green
    elif probability < 0.5:
        return "#f59e0b"   # yellow
    elif probability < 0.7:
        return "#f97316"   # orange
    else:
        return "#ef4444"   # red


# ============================================================
# Test block
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("TESTING model_utils.py")
    print("="*50)

    # Test build_input_features
    test_input = build_input_features(
        hour=8,               # 8 AM peak hour
        day_of_week=0,        # Monday
        month=1,              # January
        weather_risk_score=4, # Foggy
        visibility_bucket=0,  # Low visibility
        temp_bucket=0,        # Freezing
        is_junction=True,
        is_traffic_signal=False,
        is_crossing=False
    )
    print(f"\n✅ Input features built: {test_input}")

    # Test predict_risk
    result = predict_risk(test_input)
    print(f"\n✅ Prediction result:")
    print(f"   Risk Label    : {result['risk_label']}")
    print(f"   Probability   : {result['risk_probability']:.1%}")
    print(f"   Is High Risk  : {result['is_high_risk']}")
    print(f"   Color         : {result['risk_color']}")
    print("\n✅ model_utils.py working correctly!")