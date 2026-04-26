import pandas as pd
import json
import math
import os

# ============================================================
# Dynamic path — works from anywhere
# ============================================================
BASE_DIR = os.path.dirname(       # utils/
           os.path.dirname(       # app/
           os.path.dirname(       # project root
           os.path.abspath(__file__))))

HOTSPOTS_PATH = os.path.join(BASE_DIR, "data", "hotspots.json")


def load_hotspots():
    """
    Load hotspots data from JSON file and normalize risk_score.

    Returns:
        pd.DataFrame: Hotspots data with normalized risk_score
    """
    try:
        print(f"📁 Loading hotspots from: {HOTSPOTS_PATH}")

        with open(HOTSPOTS_PATH, "r") as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        print(f"✅ Loaded {len(df)} hotspots")
        print(f"   Columns: {df.columns.tolist()}")

        # Normalize risk_score to 0-1
        if 'risk_score' in df.columns:
            min_risk = df['risk_score'].min()
            max_risk = df['risk_score'].max()
            if max_risk > min_risk:
                df['risk_score'] = (df['risk_score'] - min_risk) / (max_risk - min_risk)
            else:
                df['risk_score'] = 0.5
        else:
            df['risk_score'] = 0.5

        # Also keep normalized_risk for backward compatibility
        df['normalized_risk'] = df['risk_score']

        return df

    except FileNotFoundError:
        print(f"❌ hotspots.json not found at: {HOTSPOTS_PATH}")
        return pd.DataFrame()

    except Exception as e:
        print(f"❌ Error loading hotspots: {e}")
        return pd.DataFrame()


def compute_zone_risks(hotspots_df, weather_df, current_hour):
    """
    Compute dynamic risk scores for zones based on weather and time.

    Args:
        hotspots_df (pd.DataFrame): Hotspots data
        weather_df (pd.DataFrame): Weather forecast data
        current_hour (int): Current hour of day

    Returns:
        pd.DataFrame: Hotspots with dynamic_risk_score column
    """
    df = hotspots_df.copy()

    # Base risk
    df['dynamic_risk_score'] = df['risk_score']

    # Weather multiplier
    if weather_df is not None and not weather_df.empty:
        current_weather = weather_df.iloc[0]
        weather_risk    = int(current_weather.get('weather_risk_score', 2))

        weather_multipliers = {
            1: 1.0, 2: 1.2, 3: 1.5, 4: 1.8, 5: 2.0
        }
        weather_mult = weather_multipliers.get(weather_risk, 1.2)
        df['dynamic_risk_score'] *= weather_mult

    # Hour multiplier
    if current_hour is not None:
        if current_hour in [7, 8, 9, 17, 18, 19]:
            hour_mult = 1.5   # peak hours
        elif current_hour < 6 or current_hour >= 22:
            hour_mult = 1.2   # night
        else:
            hour_mult = 1.0

        df['dynamic_risk_score'] *= hour_mult

    # Normalize dynamic risk to 0-1
    max_dynamic = df['dynamic_risk_score'].max()
    if max_dynamic > 0:
        df['dynamic_risk_score'] = df['dynamic_risk_score'] / max_dynamic

    return df


def get_deployment_plan(n_units, hotspots_df, weather_df=None, current_hour=None):
    """
    Generate deployment plan using greedy algorithm.

    Args:
        n_units (int): Number of units to deploy
        hotspots_df (pd.DataFrame): Hotspots data
        weather_df (pd.DataFrame): Weather data (optional)
        current_hour (int): Current hour (optional)

    Returns:
        list: List of deployment dictionaries
    """
    if hotspots_df is None or hotspots_df.empty:
        print("❌ No hotspot data for deployment plan")
        return []

    # Compute dynamic risks
    zones_df = compute_zone_risks(hotspots_df, weather_df, current_hour)

    # Sort by dynamic risk descending
    zones_df = zones_df.sort_values('dynamic_risk_score', ascending=False)

    # Handle case where n_units > available zones
    available = len(zones_df)
    if n_units > available:
        print(f"⚠️ Requested {n_units} units but only {available} zones available")
        # Repeat top zones if needed
        zones_df = pd.concat(
            [zones_df] * (n_units // available + 1)
        ).head(n_units)

    top_zones = zones_df.head(n_units)

    # Auto detect lat/lng column names
    lat_col = _get_col(zones_df, ['centroid_lat', 'lat', 'latitude', 'start_lat'])
    lng_col = _get_col(zones_df, ['centroid_lng', 'lng', 'longitude', 'start_lng'])

    deployment_plan = []
    for i, (_, zone) in enumerate(top_zones.iterrows()):
        deployment_plan.append({
            "unit_id"             : i + 1,
            "deploy_lat"          : float(zone.get(lat_col, 0)) if lat_col else 0,
            "deploy_lng"          : float(zone.get(lng_col, 0)) if lng_col else 0,
            "zone_risk"           : float(zone['dynamic_risk_score']),
            "zone_accident_count" : int(zone.get('accident_count', 0)),
            "zone_avg_severity"   : float(zone.get('avg_severity', 0)),
            "recommended_position": f"Zone #{i+1} — Risk: {zone['dynamic_risk_score']:.2f}"
        })

    print(f"✅ Deployment plan created for {len(deployment_plan)} units")
    return deployment_plan


def _get_col(df, candidates):
    """Helper to find first matching column name."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great circle distance between two points in km.
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a    = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c    = 2 * math.asin(math.sqrt(a))

    return 6371 * c  # Earth radius in km


def estimate_coverage(deployment_plan, all_hotspots_df):
    """
    Estimate coverage percentage based on deployment plan.
    A hotspot is covered if any unit is within 15km.

    Args:
        deployment_plan (list): List of deployment dicts
        all_hotspots_df (pd.DataFrame): All hotspots data

    Returns:
        dict: {coverage_pct, covered_zones, total_zones}
    """
    if not deployment_plan or all_hotspots_df.empty:
        return {"coverage_pct": 0, "covered_zones": 0, "total_zones": 0}

    lat_col = _get_col(all_hotspots_df, ['centroid_lat', 'lat', 'latitude'])
    lng_col = _get_col(all_hotspots_df, ['centroid_lng', 'lng', 'longitude'])

    if lat_col is None or lng_col is None:
        return {"coverage_pct": 0, "covered_zones": 0, "total_zones": len(all_hotspots_df)}

    covered_zones = 0
    total_zones   = len(all_hotspots_df)

    for _, hotspot in all_hotspots_df.iterrows():
        h_lat = float(hotspot.get(lat_col, 0))
        h_lng = float(hotspot.get(lng_col, 0))

        for unit in deployment_plan:
            dist = haversine_distance(h_lat, h_lng,
                                      unit['deploy_lat'],
                                      unit['deploy_lng'])
            if dist <= 15:
                covered_zones += 1
                break

    coverage_pct = (covered_zones / total_zones * 100) if total_zones > 0 else 0

    return {
        "coverage_pct" : round(coverage_pct, 1),
        "covered_zones": covered_zones,
        "total_zones"  : total_zones
    }


# ============================================================
# Test block
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("TESTING optimizer.py")
    print("=" * 50)

    df = load_hotspots()

    if not df.empty:
        print(f"\n✅ Hotspots loaded: {len(df)} zones")
        print(f"   Columns: {df.columns.tolist()}")

        plan = get_deployment_plan(5, df)
        print(f"\n✅ Deployment plan (5 units):")
        for unit in plan:
            print(f"   Unit {unit['unit_id']}: "
                  f"({unit['deploy_lat']:.3f}, {unit['deploy_lng']:.3f}) "
                  f"Risk={unit['zone_risk']:.3f}")

        coverage = estimate_coverage(plan, df)
        print(f"\n✅ Coverage: {coverage['coverage_pct']:.1f}% "
              f"({coverage['covered_zones']}/{coverage['total_zones']} zones)")
    else:
        print("❌ No hotspots loaded!")