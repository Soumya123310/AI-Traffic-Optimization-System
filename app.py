import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Green AI Traffic Optimizer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("traffic_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# -----------------------------
# LOCATION DATABASE
# -----------------------------
LOCATIONS = {
    "Bengaluru City Center": {
        "km_from_start": 0,
        "hour": 7, "month": 6, "day_of_week": 1,
        "temp": 295.0, "rain_1h": 0.0,
        "snow_1h": 0.0, "clouds_all": 40,
        "scenario": "Morning Rush Hour — Heavy office traffic",
    },
    "Kengeri": {
        "km_from_start": 18,
        "hour": 8, "month": 6, "day_of_week": 1,
        "temp": 294.0, "rain_1h": 2.0,
        "snow_1h": 0.0, "clouds_all": 70,
        "scenario": "Light Rain — Visibility reduced",
    },
    "Bidadi": {
        "km_from_start": 40,
        "hour": 9, "month": 6, "day_of_week": 1,
        "temp": 292.0, "rain_1h": 8.0,
        "snow_1h": 0.0, "clouds_all": 90,
        "scenario": "Heavy Rain — Peak congestion",
    },
    "Ramanagara": {
        "km_from_start": 50,
        "hour": 10, "month": 6, "day_of_week": 1,
        "temp": 294.0, "rain_1h": 0.0,
        "snow_1h": 0.0, "clouds_all": 35,
        "scenario": "Mid-morning — Silk town traffic",
    },
    "Channapatna": {
        "km_from_start": 60,
        "hour": 10, "month": 6, "day_of_week": 1,
        "temp": 293.0, "rain_1h": 3.0,
        "snow_1h": 0.0, "clouds_all": 60,
        "scenario": "Rain Clearing — Traffic moderating",
    },
    "Tumkur": {
        "km_from_start": 70,
        "hour": 9, "month": 6, "day_of_week": 1,
        "temp": 296.0, "rain_1h": 1.0,
        "snow_1h": 0.0, "clouds_all": 50,
        "scenario": "Morning — Industrial area traffic",
    },
    "Maddur": {
        "km_from_start": 90,
        "hour": 11, "month": 6, "day_of_week": 1,
        "temp": 296.0, "rain_1h": 0.0,
        "snow_1h": 0.0, "clouds_all": 30,
        "scenario": "Clear Weather — Smooth highway flow",
    },
    "Mandya": {
        "km_from_start": 100,
        "hour": 12, "month": 6, "day_of_week": 1,
        "temp": 299.0, "rain_1h": 0.0,
        "snow_1h": 0.0, "clouds_all": 20,
        "scenario": "Midday — Town traffic near market",
    },
    "Srirangapatna": {
        "km_from_start": 125,
        "hour": 13, "month": 6, "day_of_week": 1,
        "temp": 300.0, "rain_1h": 0.0,
        "snow_1h": 0.0, "clouds_all": 15,
        "scenario": "Tourist Area — Afternoon crowd",
    },
    "Mysore City": {
        "km_from_start": 140,
        "hour": 14, "month": 6, "day_of_week": 1,
        "temp": 298.0, "rain_1h": 0.0,
        "snow_1h": 0.0, "clouds_all": 25,
        "scenario": "Destination — City entry traffic",
    },
    "Hassan": {
        "km_from_start": 180,
        "hour": 15, "month": 6, "day_of_week": 1,
        "temp": 291.0, "rain_1h": 5.0,
        "snow_1h": 0.0, "clouds_all": 80,
        "scenario": "Evening — Moderate highway traffic",
    },
    "Hubli": {
        "km_from_start": 410,
        "hour": 16, "month": 6, "day_of_week": 1,
        "temp": 300.0, "rain_1h": 0.0,
        "snow_1h": 0.0, "clouds_all": 20,
        "scenario": "Afternoon — Major city traffic",
    },
}

LOCATION_NAMES = list(LOCATIONS.keys())


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------


def estimate_co2(traffic_volume):
    return round(traffic_volume * 0.21 * 1.0, 2)


def estimate_fuel_waste(signal_time, traffic_volume):
    return round(traffic_volume * 0.6 * (signal_time / 3600), 4)


def carbon_saved(optimized_time, traffic_volume):
    saved = (
        traffic_volume
        * 0.6
        * ((120 / 3600) - (optimized_time / 3600))
        * 0.735
    )
    return round(saved, 4)


def get_signal_plan(signal_time):
    yellow = 5
    red = max(120 - signal_time - yellow, 20)
    return {
        "Green": signal_time,
        "Yellow": yellow,
        "Red": red,
        "Total Cycle": signal_time + yellow + red,
    }


def get_level(prediction):
    if prediction < 3000:
        return "Low Traffic", 30, "🟢", "green"
    elif prediction < 6000:
        return "Medium Traffic", 60, "🟡", "yellow"
    else:
        return "High Traffic", 90, "🔴", "red"


def get_action_plan(level):
    plans = {
        "Low Traffic": (
            "✅ Normal signal operation. No intervention needed. "
            "Energy-saving mode can be activated at intersections."
        ),
        "Medium Traffic": (
            "⚡ Activate adaptive signal mode. "
            "Reduce red light duration on parallel roads. "
            "Monitor closely for escalation to high traffic."
        ),
        "High Traffic": (
            "🚨 Deploy traffic personnel immediately. "
            "Activate alternate route signage. "
            "Switch all signals to high-flow mode urgently."
        ),
    }
    return plans[level]


def get_route_suggestion(level):
    routes = {
        "Low Traffic": {
            "main": ("🟢", "NH 275 Main Highway", "CLEAR"),
            "alt1": ("🟢", "Old Mysore Road", "CLEAR"),
            "alt2": ("🟢", "NICE Road Bypass", "CLEAR"),
            "rec": "All routes clear. Stay on NH 275.",
            "saved": "0 min",
        },
        "Medium Traffic": {
            "main": ("🟡", "NH 275 Main Highway", "MODERATE"),
            "alt1": ("🟢", "Old Mysore Road", "CLEAR"),
            "alt2": ("🟡", "NICE Road Bypass", "MODERATE"),
            "rec": "Switch to Old Mysore Road to avoid congestion.",
            "saved": "~10 min",
        },
        "High Traffic": {
            "main": ("🔴", "NH 275 Main Highway", "CONGESTED"),
            "alt1": ("🟢", "Old Mysore Road", "CLEAR"),
            "alt2": ("🟡", "NICE Road Bypass", "MODERATE"),
            "rec": "Divert immediately to Old Mysore Road.",
            "saved": "~20 min",
        },
    }
    return routes[level]


def get_hourly_forecast(base_input):
    records = []
    for h in range(24):
        row = base_input.copy()
        row["hour"] = h
        records.append(row)
    forecast_df = pd.DataFrame(records)
    for col in feature_columns:
        if col not in forecast_df.columns:
            forecast_df[col] = 0
    forecast_df = forecast_df[feature_columns]
    return model.predict(forecast_df)


def predict_traffic(input_data):
    input_df = pd.DataFrame([input_data])
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]
    return int(model.predict(input_df)[0])


def get_accident_risk(rain, traffic_volume, hour, clouds):
    risk = (
        (rain * 2.5)
        + (traffic_volume / 200)
        + (5 if hour < 6 or hour > 21 else 0)
        + (clouds / 20)
    )
    risk = min(round(risk, 1), 100)
    if risk < 30:
        return risk, "Low Risk", "green"
    elif risk < 60:
        return risk, "Moderate Risk", "yellow"
    else:
        return risk, "High Risk", "red"


def get_prediction_confidence(input_df):
    all_preds = [
        tree.predict(input_df)[0] for tree in model.estimators_
    ]
    mean_pred = np.mean(all_preds)
    std_pred = np.std(all_preds)
    confidence = max(
        0, min(100, round(100 - (std_pred / mean_pred) * 100, 1))
    )
    return confidence


def get_journey_score(results):
    score = 100
    for r in results:
        if r["color"] == "red":
            score -= 15
        elif r["color"] == "yellow":
            score -= 7
        if r["cp"]["rain_1h"] > 5:
            score -= 5
        elif r["cp"]["rain_1h"] > 0:
            score -= 2
    score = max(0, score)
    if score >= 70:
        return score, "Good Journey", "green"
    elif score >= 45:
        return score, "Moderate Journey", "yellow"
    else:
        return score, "Difficult Journey", "red"


def get_best_travel_time(hourly_preds):
    min_hour = int(np.argmin(hourly_preds))
    max_hour = int(np.argmax(hourly_preds))
    peak_hours = [
        i for i, v in enumerate(hourly_preds) if v > 5000
    ]
    avoid = (
        f"{peak_hours[0]}:00 - {peak_hours[-1] + 1}:00"
        if peak_hours
        else "No peak window today"
    )
    return min_hour, max_hour, avoid


def build_checkpoints(start, destination):
    start_km = LOCATIONS[start]["km_from_start"]
    dest_km = LOCATIONS[destination]["km_from_start"]
    lo = min(start_km, dest_km)
    hi = max(start_km, dest_km)
    checkpoints = []
    for name, loc in LOCATIONS.items():
        if lo <= loc["km_from_start"] <= hi:
            cp = loc.copy()
            cp["name"] = name
            checkpoints.append(cp)
    checkpoints.sort(key=lambda x: x["km_from_start"])
    return checkpoints


def make_route_card(label, emoji, name, status):
    color_map = {
        "🟢": ("route-card-green", "route-status-green"),
        "🟡": ("route-card-yellow", "route-status-yellow"),
        "🔴": ("route-card-red", "route-status-red"),
    }
    cc, sc = color_map[emoji]
    lbl_style = (
        "font-family:'Exo 2',sans-serif;"
        "color:rgba(160,220,255,0.38);font-size:0.68rem;"
        "letter-spacing:2px;text-transform:uppercase;"
        "margin-bottom:3px;"
    )
    return (
        f'<div class="route-card {cc}">'
        f'<div><div style="{lbl_style}">{label}</div>'
        f'<div class="route-name">{emoji} {name}</div></div>'
        f'<div class="{sc}">{status}</div></div>'
    )


def metric_card(icon, label, value_html, sub=""):
    sub_html = (
        f'<div class="metric-label">{sub}</div>' if sub else ""
    )
    return (
        f'<div class="metric-card">'
        f'<div class="metric-icon">{icon}</div>'
        f'<div class="metric-label">{label}</div>'
        f'{value_html}{sub_html}</div>'
    )


def mv(color, text):
    cls = (
        "metric-value"
        if color == "green"
        else f"metric-value-{color}"
    )
    return f'<div class="{cls}">{text}</div>'


def alert(kind, html):
    return f'<div class="alert-{kind}">{html}</div>'


def section(text):
    return f'<div class="section-header">{text}</div>'


# -----------------------------
# CSS
# -----------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron\
:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap');

* { box-sizing: border-box; }
.stApp {
    background: #020b14;
    background-image:
        radial-gradient(ellipse at 20% 50%,
            rgba(0,255,150,0.04) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%,
            rgba(0,180,255,0.04) 0%, transparent 60%);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }

.hero-banner {
    background: linear-gradient(
        135deg, #020b14 0%, #061a2e 50%, #020b14 100%
    );
    border: 1px solid rgba(0,255,150,0.2);
    border-radius: 20px; padding: 38px 50px;
    margin-bottom: 10px; text-align: center;
    position: relative; overflow: hidden;
}
.hero-banner::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(
        90deg, transparent, #00FF96, #00BFFF, transparent
    );
}
.hero-title {
    font-family: 'Orbitron', monospace; font-size: 2.2rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00FF96, #00BFFF, #00FF96);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s infinite linear;
    margin: 0 0 8px 0; letter-spacing: 2px;
}
.hero-sub {
    font-family: 'Exo 2', sans-serif;
    color: rgba(160,220,255,0.7); font-size: 0.88rem;
    letter-spacing: 3px; text-transform: uppercase;
}
.hero-badges {
    display: flex; justify-content: center;
    gap: 10px; margin-top: 16px; flex-wrap: wrap;
}
.badge {
    font-family: 'Exo 2', sans-serif; font-size: 0.72rem;
    padding: 4px 13px; border-radius: 20px; border: 1px solid;
    letter-spacing: 1px; text-transform: uppercase;
}
.badge-green {
    color: #00FF96; border-color: rgba(0,255,150,0.4);
    background: rgba(0,255,150,0.08);
}
.badge-blue {
    color: #00BFFF; border-color: rgba(0,191,255,0.4);
    background: rgba(0,191,255,0.08);
}
.badge-purple {
    color: #c084fc; border-color: rgba(192,132,252,0.4);
    background: rgba(192,132,252,0.08);
}
.badge-orange {
    color: #fb923c; border-color: rgba(251,146,60,0.4);
    background: rgba(251,146,60,0.08);
}
@keyframes shimmer {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(6,26,46,0.8) !important;
    border-radius: 12px !important; padding: 5px !important;
    gap: 3px !important;
    border: 1px solid rgba(0,191,255,0.15) !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 600 !important;
    color: rgba(160,220,255,0.6) !important;
    border-radius: 8px !important;
    padding: 9px 16px !important;
    letter-spacing: 0.5px !important; border: none !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(
        135deg,
        rgba(0,255,150,0.15),
        rgba(0,191,255,0.15)
    ) !important;
    color: #00FF96 !important;
    border: 1px solid rgba(0,255,150,0.3) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #00FF96, #00BFFF) !important;
    color: #020b14 !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important; font-size: 0.88rem !important;
    letter-spacing: 2px !important; border: none !important;
    border-radius: 12px !important; height: 3.3em !important;
    width: 100% !important; text-transform: uppercase !important;
    box-shadow: 0 4px 20px rgba(0,255,150,0.3) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,255,150,0.5) !important;
}

.input-card {
    background: linear-gradient(135deg, #061a2e, #071f38);
    border: 1px solid rgba(0,191,255,0.2);
    border-radius: 16px; padding: 18px 22px 4px 22px;
    margin-bottom: 14px; position: relative; overflow: hidden;
}
.input-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(
        90deg, transparent, #00BFFF, transparent
    );
}
.input-card-title {
    font-family: 'Orbitron', monospace; color: #00BFFF;
    font-size: 0.78rem; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 14px;
}

.route-select-card {
    background: linear-gradient(135deg, #061a2e, #071f38);
    border: 1px solid rgba(0,255,150,0.2);
    border-radius: 16px; padding: 20px 24px;
    margin-bottom: 16px; position: relative; overflow: hidden;
}
.route-select-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(
        90deg, transparent, #00FF96, transparent
    );
}
.route-select-title {
    font-family: 'Orbitron', monospace; color: #00FF96;
    font-size: 0.78rem; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 14px;
}

.metric-card {
    background: linear-gradient(135deg, #061a2e, #071f38);
    border: 1px solid rgba(0,255,150,0.15);
    border-radius: 14px; padding: 18px; text-align: center;
    position: relative; overflow: hidden; margin-bottom: 12px;
}
.metric-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(
        90deg, transparent, #00FF96, transparent
    );
}
.metric-icon { font-size: 1.7rem; margin-bottom: 5px; }
.metric-label {
    font-family: 'Exo 2', sans-serif;
    color: rgba(160,220,255,0.5); font-size: 0.7rem;
    letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 5px;
}
.metric-value {
    font-family: 'Orbitron', monospace;
    color: #00FF96; font-size: 1.65rem; font-weight: 700;
}
.metric-value-blue {
    font-family: 'Orbitron', monospace;
    color: #00BFFF; font-size: 1.65rem; font-weight: 700;
}
.metric-value-yellow {
    font-family: 'Orbitron', monospace;
    color: #FFD700; font-size: 1.65rem; font-weight: 700;
}
.metric-value-red {
    font-family: 'Orbitron', monospace;
    color: #FF4D6D; font-size: 1.65rem; font-weight: 700;
}

.section-header {
    font-family: 'Orbitron', monospace; color: #00BFFF;
    font-size: 0.82rem; letter-spacing: 3px;
    text-transform: uppercase; margin: 20px 0 13px 0;
    padding-bottom: 9px;
    border-bottom: 1px solid rgba(0,191,255,0.2);
}

.alert-success {
    background: rgba(0,255,150,0.08);
    border: 1px solid rgba(0,255,150,0.3);
    border-left: 4px solid #00FF96; border-radius: 10px;
    padding: 13px 17px; font-family: 'Exo 2', sans-serif;
    color: rgba(200,255,230,0.9); margin: 9px 0;
    font-size: 0.88rem;
}
.alert-warning {
    background: rgba(255,215,0,0.08);
    border: 1px solid rgba(255,215,0,0.3);
    border-left: 4px solid #FFD700; border-radius: 10px;
    padding: 13px 17px; font-family: 'Exo 2', sans-serif;
    color: rgba(255,240,180,0.9); margin: 9px 0;
    font-size: 0.88rem;
}
.alert-danger {
    background: rgba(255,77,109,0.08);
    border: 1px solid rgba(255,77,109,0.3);
    border-left: 4px solid #FF4D6D; border-radius: 10px;
    padding: 13px 17px; font-family: 'Exo 2', sans-serif;
    color: rgba(255,200,210,0.9); margin: 9px 0;
    font-size: 0.88rem;
}
.alert-info {
    background: rgba(0,191,255,0.08);
    border: 1px solid rgba(0,191,255,0.3);
    border-left: 4px solid #00BFFF; border-radius: 10px;
    padding: 13px 17px; font-family: 'Exo 2', sans-serif;
    color: rgba(180,230,255,0.9); margin: 9px 0;
    font-size: 0.88rem;
}

.signal-container {
    display: flex; justify-content: center;
    gap: 22px; margin: 18px 0; flex-wrap: wrap;
}
.signal-block {
    background: #061a2e;
    border: 1px solid rgba(0,191,255,0.2);
    border-radius: 14px; padding: 16px 26px;
    text-align: center; min-width: 115px;
}
.signal-light {
    width: 44px; height: 44px;
    border-radius: 50%; margin: 0 auto 7px;
}
.signal-light-green {
    background: #00FF96; box-shadow: 0 0 20px #00FF96;
}
.signal-light-yellow {
    background: #FFD700; box-shadow: 0 0 20px #FFD700;
}
.signal-light-red {
    background: #FF4D6D; box-shadow: 0 0 20px #FF4D6D;
}
.signal-light-dim { background: #1a2a3a; box-shadow: none; }
.signal-time {
    font-family: 'Orbitron', monospace;
    font-size: 1.2rem; font-weight: 700; color: #fff;
}
.signal-sublabel {
    font-family: 'Exo 2', sans-serif; font-size: 0.68rem;
    color: rgba(160,220,255,0.4); text-transform: uppercase;
    letter-spacing: 1px; margin-top: 3px;
}

.route-card {
    background: #061a2e; border-radius: 12px;
    padding: 14px 18px; margin-bottom: 9px; border: 1px solid;
    display: flex; align-items: center;
    justify-content: space-between; flex-wrap: wrap; gap: 8px;
}
.route-card-green { border-color: rgba(0,255,150,0.3); }
.route-card-yellow { border-color: rgba(255,215,0,0.3); }
.route-card-red { border-color: rgba(255,77,109,0.3); }
.route-name {
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem; color: #fff; font-weight: 700;
}
.route-status-green {
    font-family: 'Exo 2', sans-serif; font-size: 0.72rem;
    color: #00FF96; background: rgba(0,255,150,0.1);
    border: 1px solid rgba(0,255,150,0.3);
    border-radius: 20px; padding: 3px 11px; letter-spacing: 1px;
}
.route-status-yellow {
    font-family: 'Exo 2', sans-serif; font-size: 0.72rem;
    color: #FFD700; background: rgba(255,215,0,0.1);
    border: 1px solid rgba(255,215,0,0.3);
    border-radius: 20px; padding: 3px 11px; letter-spacing: 1px;
}
.route-status-red {
    font-family: 'Exo 2', sans-serif; font-size: 0.72rem;
    color: #FF4D6D; background: rgba(255,77,109,0.1);
    border: 1px solid rgba(255,77,109,0.3);
    border-radius: 20px; padding: 3px 11px; letter-spacing: 1px;
}

.custom-progress-bg {
    background: rgba(6,26,46,0.8); border-radius: 10px;
    height: 15px; overflow: hidden;
    border: 1px solid rgba(0,191,255,0.2); margin: 5px 0;
}
.custom-progress-fill-green {
    height: 100%; border-radius: 10px;
    background: linear-gradient(90deg, #00FF96, #00BFFF);
    box-shadow: 0 0 10px rgba(0,255,150,0.5);
}
.custom-progress-fill-yellow {
    height: 100%; border-radius: 10px;
    background: linear-gradient(90deg, #FFD700, #FF8C00);
    box-shadow: 0 0 10px rgba(255,215,0,0.5);
}
.custom-progress-fill-red {
    height: 100%; border-radius: 10px;
    background: linear-gradient(90deg, #FF4D6D, #FF0040);
    box-shadow: 0 0 10px rgba(255,77,109,0.5);
}

.checkpoint-card {
    background: linear-gradient(135deg, #061a2e, #071f38);
    border-radius: 16px; padding: 20px; margin-bottom: 12px;
    border: 1px solid rgba(0,191,255,0.2);
    position: relative; overflow: hidden;
}
.checkpoint-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
}
.checkpoint-card-green::before {
    background: linear-gradient(
        90deg, transparent, #00FF96, transparent
    );
}
.checkpoint-card-yellow::before {
    background: linear-gradient(
        90deg, transparent, #FFD700, transparent
    );
}
.checkpoint-card-red::before {
    background: linear-gradient(
        90deg, transparent, #FF4D6D, transparent
    );
}
.cp-header {
    display: flex; justify-content: space-between;
    align-items: flex-start; flex-wrap: wrap;
    gap: 9px; margin-bottom: 13px;
}
.cp-name {
    font-family: 'Orbitron', monospace;
    color: #fff; font-size: 0.92rem; font-weight: 700;
}
.cp-meta {
    font-family: 'Exo 2', sans-serif;
    color: rgba(160,220,255,0.5); font-size: 0.75rem;
    letter-spacing: 1px; margin-top: 3px;
}
.cp-scenario {
    font-family: 'Exo 2', sans-serif;
    color: rgba(160,220,255,0.62); font-size: 0.78rem;
    font-style: italic; margin-top: 3px;
}
.cp-badge-green {
    font-family: 'Exo 2', sans-serif; font-size: 0.72rem;
    color: #00FF96; background: rgba(0,255,150,0.1);
    border: 1px solid rgba(0,255,150,0.3);
    border-radius: 20px; padding: 3px 12px;
    letter-spacing: 1px; white-space: nowrap;
}
.cp-badge-yellow {
    font-family: 'Exo 2', sans-serif; font-size: 0.72rem;
    color: #FFD700; background: rgba(255,215,0,0.1);
    border: 1px solid rgba(255,215,0,0.3);
    border-radius: 20px; padding: 3px 12px;
    letter-spacing: 1px; white-space: nowrap;
}
.cp-badge-red {
    font-family: 'Exo 2', sans-serif; font-size: 0.72rem;
    color: #FF4D6D; background: rgba(255,77,109,0.1);
    border: 1px solid rgba(255,77,109,0.3);
    border-radius: 20px; padding: 3px 12px;
    letter-spacing: 1px; white-space: nowrap;
}
.cp-metrics {
    display: flex; gap: 9px; flex-wrap: wrap;
    margin-bottom: 11px;
}
.cp-metric {
    background: rgba(2,11,20,0.6);
    border: 1px solid rgba(0,191,255,0.15);
    border-radius: 9px; padding: 8px 12px;
    text-align: center; flex: 1; min-width: 85px;
}
.cp-metric-label {
    font-family: 'Exo 2', sans-serif;
    color: rgba(160,220,255,0.42); font-size: 0.62rem;
    letter-spacing: 1px; text-transform: uppercase;
    margin-bottom: 3px;
}
.cp-val-green {
    font-family: 'Orbitron', monospace;
    color: #00FF96; font-size: 0.9rem; font-weight: 700;
}
.cp-val-blue {
    font-family: 'Orbitron', monospace;
    color: #00BFFF; font-size: 0.9rem; font-weight: 700;
}
.cp-val-yellow {
    font-family: 'Orbitron', monospace;
    color: #FFD700; font-size: 0.9rem; font-weight: 700;
}
.cp-val-red {
    font-family: 'Orbitron', monospace;
    color: #FF4D6D; font-size: 0.9rem; font-weight: 700;
}
.cp-route {
    background: rgba(2,11,20,0.6); border-radius: 9px;
    padding: 10px 13px; font-family: 'Exo 2', sans-serif;
    font-size: 0.78rem; color: rgba(180,230,255,0.8);
    border: 1px solid rgba(0,191,255,0.1); line-height: 1.7;
}

.summary-card {
    background: linear-gradient(135deg, #061a2e, #071f38);
    border: 1px solid rgba(0,255,150,0.25);
    border-radius: 16px; padding: 24px; margin-top: 16px;
    position: relative; overflow: hidden;
}
.summary-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(
        90deg, transparent, #00FF96, #00BFFF, transparent
    );
}
.summary-title {
    font-family: 'Orbitron', monospace; color: #00FF96;
    font-size: 0.82rem; letter-spacing: 3px;
    text-transform: uppercase; margin-bottom: 16px;
}
.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
    gap: 11px;
}
.summary-item {
    background: rgba(2,11,20,0.6);
    border: 1px solid rgba(0,191,255,0.15);
    border-radius: 11px; padding: 13px; text-align: center;
}
.summary-icon { font-size: 1.3rem; margin-bottom: 5px; }
.summary-label {
    font-family: 'Exo 2', sans-serif;
    color: rgba(160,220,255,0.42); font-size: 0.65rem;
    letter-spacing: 1px; text-transform: uppercase;
    margin-bottom: 5px;
}
.summary-val-green {
    font-family: 'Orbitron', monospace;
    color: #00FF96; font-size: 0.95rem; font-weight: 700;
}
.summary-val-blue {
    font-family: 'Orbitron', monospace;
    color: #00BFFF; font-size: 0.95rem; font-weight: 700;
}
.summary-val-red {
    font-family: 'Orbitron', monospace;
    color: #FF4D6D; font-size: 0.95rem; font-weight: 700;
}
.summary-val-yellow {
    font-family: 'Orbitron', monospace;
    color: #FFD700; font-size: 0.95rem; font-weight: 700;
}

.journey-bar {
    background: rgba(6,26,46,0.8);
    border: 1px solid rgba(0,191,255,0.2);
    border-radius: 16px; padding: 16px 26px; margin-bottom: 18px;
}
.journey-track {
    display: flex; align-items: center;
    justify-content: space-between; position: relative;
}
.journey-track::before {
    content: ''; position: absolute;
    top: 50%; left: 0; right: 0; height: 2px;
    background: rgba(0,191,255,0.2);
    transform: translateY(-50%); z-index: 0;
}
.journey-dot {
    width: 12px; height: 12px; border-radius: 50%;
    background: #00FF96; box-shadow: 0 0 8px #00FF96;
    z-index: 1; flex-shrink: 0;
}
.journey-dot-dim {
    width: 8px; height: 8px; border-radius: 50%;
    background: rgba(0,191,255,0.3); z-index: 1; flex-shrink: 0;
}
.stSlider label {
    font-family: 'Exo 2', sans-serif !important;
    color: rgba(160,220,255,0.8) !important;
    font-size: 0.83rem !important;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# -----------------------------
# HERO
# -----------------------------
st.markdown(
    """
<div class="hero-banner">
    <div class="hero-title">🌱 GREEN AI TRAFFIC OPTIMIZER</div>
    <div class="hero-sub">
        Intelligent Signal Control · Carbon Reduction
        · Route Optimization
    </div>
    <div class="hero-badges">
        <span class="badge badge-green">🤖 ML Powered</span>
        <span class="badge badge-blue">⚡ Real-Time Analysis</span>
        <span class="badge badge-purple">🌍 Eco Friendly</span>
        <span class="badge badge-orange">🛣️ Journey Simulation</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# MAIN TABS
# -----------------------------
main_tab1, main_tab2 = st.tabs([
    "🎛️ Live Traffic Predictor",
    "🛣️ Journey Simulation",
])

# ================================================
# TAB 1 — LIVE PREDICTOR
# ================================================
with main_tab1:

    st.markdown(
        section("⚙️ Input Parameters"), unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div class="input-card">'
            '<div class="input-card-title">'
            "🌦 Weather Conditions</div></div>",
            unsafe_allow_html=True,
        )
        temp = st.slider(
            "🌡️ Temperature (Kelvin)", 250.0, 330.0, 280.0
        )
        rain = st.slider(
            "🌧️ Rain — last 1 hour (mm)", 0.0, 50.0, 0.0
        )
        snow = st.slider(
            "❄️ Snow — last 1 hour (mm)", 0.0, 50.0, 0.0
        )
        clouds = st.slider("☁️ Cloud Coverage (%)", 0, 100, 50)

    with col2:
        st.markdown(
            '<div class="input-card">'
            '<div class="input-card-title">'
            "📅 Date &amp; Time</div></div>",
            unsafe_allow_html=True,
        )
        hour = st.slider("🕐 Hour of Day", 0, 23, 8)
        day = st.slider("📆 Day of Month", 1, 31, 15)
        month = st.slider("🗓️ Month", 1, 12, 6)
        day_of_week = st.slider(
            "📅 Day of Week (0=Mon, 6=Sun)", 0, 6, 2
        )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button(
        "🚀  ANALYZE TRAFFIC & GENERATE SOLUTION",
        key="predict_btn",
    )

    input_data = {
        "temp": temp, "rain_1h": rain,
        "snow_1h": snow, "clouds_all": clouds,
        "hour": hour, "day": day,
        "month": month, "day_of_week": day_of_week,
    }
    input_df = pd.DataFrame([input_data])
    for c in feature_columns:
        if c not in input_df.columns:
            input_df[c] = 0
    input_df = input_df[feature_columns]

    if predict_btn:
        prediction = int(model.predict(input_df)[0])
        level, signal_time, color_emoji, level_color = get_level(
            prediction
        )
        plan = get_signal_plan(signal_time)
        routes = get_route_suggestion(level)
        hourly_preds = get_hourly_forecast(input_data.copy())
        avg_traffic = int(np.mean(hourly_preds))
        diff_pct = round(
            ((prediction - avg_traffic) / avg_traffic) * 100, 1
        )
        risk_val, risk_label, risk_color = get_accident_risk(
            rain, prediction, hour, clouds
        )
        confidence = get_prediction_confidence(input_df)
        best_hour, _, avoid_window = get_best_travel_time(
            hourly_preds
        )

        t1, t2, t3, t4, t5, t6 = st.tabs([
            "📊 Overview", "🚦 Signal Plan",
            "🗺️ Route Guide", "📈 24H Forecast",
            "🌍 Eco Impact", "🧠 AI Insights",
        ])

        with t1:
            st.markdown(
                section("📊 Overview"), unsafe_allow_html=True
            )
            traffic_pct = min(prediction / 8000, 1.0)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(
                    metric_card(
                        "🚗", "Traffic Volume",
                        mv("green", f"{prediction:,}"),
                        "vehicles / hr",
                    ),
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    metric_card(
                        color_emoji, "Traffic Level",
                        mv(level_color, level.split()[0]),
                        level.split()[1],
                    ),
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    metric_card(
                        "⚙️", "ML Confidence",
                        mv("blue", f"{confidence}%"),
                        "across 80 trees",
                    ),
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    metric_card(
                        "⚠️", "Accident Risk",
                        mv(risk_color, f"{risk_val}%"),
                        risk_label,
                    ),
                    unsafe_allow_html=True,
                )

            fill_cls = f"custom-progress-fill-{level_color}"
            st.markdown(
                section("🚦 Traffic Intensity")
                + f'<div style="font-family:\'Exo 2\',sans-serif;'
                f"color:rgba(160,220,255,0.42);font-size:0.75rem;"
                f'margin-bottom:5px;letter-spacing:1px;">'
                f"{int(traffic_pct * 100)}% of maximum capacity"
                f"</div>"
                f'<div class="custom-progress-bg">'
                f'<div class="{fill_cls}" '
                f'style="width:{int(traffic_pct * 100)}%">'
                f"</div></div>",
                unsafe_allow_html=True,
            )

            action = get_action_plan(level)
            al = (
                "success" if level_color == "green"
                else ("warning" if level_color == "yellow"
                      else "danger")
            )
            st.markdown(
                section("📋 Action Plan")
                + alert(al, action),
                unsafe_allow_html=True,
            )

            if diff_pct > 15:
                st.markdown(
                    alert(
                        "warning",
                        f"⚠️ Traffic is <b>{diff_pct}%</b> above "
                        f"daily average ({avg_traffic:,} veh/hr). "
                        "Early intervention recommended!",
                    ),
                    unsafe_allow_html=True,
                )
            elif diff_pct < -15:
                st.markdown(
                    alert(
                        "success",
                        f"✅ Traffic is <b>{abs(diff_pct)}%</b> "
                        "below daily average. No intervention needed.",
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    alert(
                        "info",
                        f"ℹ️ Traffic within normal range "
                        f"({diff_pct:+}% vs daily avg).",
                    ),
                    unsafe_allow_html=True,
                )

        with t2:
            st.markdown(
                section("🚦 Signal Cycle"), unsafe_allow_html=True
            )
            rc = "#FF4D6D" if level_color == "red" else "#1a3a5a"
            yc = "#FFD700" if level_color == "yellow" else "#1a3a5a"
            gc = "#00FF96" if level_color == "green" else "#1a3a5a"
            rl = (
                "signal-light-red"
                if level_color == "red" else "signal-light-dim"
            )
            yl = (
                "signal-light-yellow"
                if level_color == "yellow" else "signal-light-dim"
            )
            gl = (
                "signal-light-green"
                if level_color == "green" else "signal-light-dim"
            )
            st.markdown(
                f'<div class="signal-container">'
                f'<div class="signal-block">'
                f'<div class="signal-light {rl}"></div>'
                f'<div class="signal-time" style="color:{rc}">'
                f"{plan['Red']}s</div>"
                f'<div class="signal-sublabel">Red Phase</div>'
                f"</div>"
                f'<div class="signal-block">'
                f'<div class="signal-light {yl}"></div>'
                f'<div class="signal-time" style="color:{yc}">'
                f"{plan['Yellow']}s</div>"
                f'<div class="signal-sublabel">Yellow Phase</div>'
                f"</div>"
                f'<div class="signal-block">'
                f'<div class="signal-light {gl}"></div>'
                f'<div class="signal-time" style="color:{gc}">'
                f"{plan['Green']}s</div>"
                f'<div class="signal-sublabel">Green Phase</div>'
                f"</div></div>",
                unsafe_allow_html=True,
            )
            eff = round(
                (plan["Green"] / plan["Total Cycle"]) * 100, 1
            )
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    metric_card(
                        "", "Total Cycle",
                        mv("blue", f"{plan['Total Cycle']}s"),
                    ),
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    metric_card(
                        "", "Green Efficiency",
                        mv("green", f"{eff}%"),
                    ),
                    unsafe_allow_html=True,
                )
            st.markdown(
                alert(
                    "info",
                    "💡 Signal timing dynamically adjusted based on "
                    "predicted traffic to minimize idle time and "
                    "reduce carbon emissions.",
                ),
                unsafe_allow_html=True,
            )

        with t3:
            st.markdown(
                section("🗺️ Route Status"), unsafe_allow_html=True
            )
            e, n, s = routes["main"]
            st.markdown(
                make_route_card("Current Route", e, n, s),
                unsafe_allow_html=True,
            )
            e, n, s = routes["alt1"]
            st.markdown(
                make_route_card("Alternate Route 1", e, n, s),
                unsafe_allow_html=True,
            )
            e, n, s = routes["alt2"]
            st.markdown(
                make_route_card("Alternate Route 2", e, n, s),
                unsafe_allow_html=True,
            )
            rec_al = (
                "success" if level_color == "green"
                else ("warning" if level_color == "yellow"
                      else "danger")
            )
            st.markdown(
                alert(
                    rec_al,
                    f"🧭 <b>Recommendation:</b> {routes['rec']}<br>"
                    f"⏱️ <b>Time Saved:</b> {routes['saved']}",
                ),
                unsafe_allow_html=True,
            )

        with t4:
            st.markdown(
                section("📈 24H Forecast"), unsafe_allow_html=True
            )
            fdf = pd.DataFrame({
                "Hour": list(range(24)),
                "Predicted Traffic Volume": hourly_preds.astype(int),
            })
            fdf["Status"] = fdf[
                "Predicted Traffic Volume"
            ].apply(
                lambda x: "🟢 Low"
                if x < 3000
                else ("🟡 Medium" if x < 6000 else "🔴 High")
            )
            peak_h = int(
                fdf.loc[
                    fdf["Predicted Traffic Volume"].idxmax(), "Hour"
                ]
            )
            peak_v = int(fdf["Predicted Traffic Volume"].max())
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(
                    metric_card(
                        "", "Best Time",
                        mv("green", f"{best_hour}:00"),
                    ),
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    metric_card(
                        "", "Peak Hour",
                        mv("red", f"{peak_h}:00"),
                    ),
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    metric_card(
                        "", "Peak Volume",
                        mv("red", f"{peak_v:,}"),
                    ),
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    metric_card(
                        "", "Daily Average",
                        mv("blue", f"{avg_traffic:,}"),
                    ),
                    unsafe_allow_html=True,
                )
            st.markdown(
                alert(
                    "success",
                    f"✅ <b>Best time:</b> {best_hour}:00<br>"
                    f"⚠️ <b>Avoid:</b> {avoid_window}",
                ),
                unsafe_allow_html=True,
            )
            st.line_chart(
                fdf.set_index("Hour")["Predicted Traffic Volume"]
            )
            st.dataframe(
                fdf, use_container_width=True, hide_index=True
            )

        with t5:
            st.markdown(
                section("🌍 Eco Impact"), unsafe_allow_html=True
            )
            co2 = estimate_co2(prediction)
            fuel = estimate_fuel_waste(signal_time, prediction)
            saved = carbon_saved(signal_time, prediction)
            trees = round(saved / 21.77, 2)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    metric_card(
                        "💨", "CO2 Emissions",
                        mv("red", f"{co2} kg/hr"),
                    ),
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    metric_card(
                        "⛽", "Fuel Wasted",
                        mv("yellow", f"{fuel} L/hr"),
                    ),
                    unsafe_allow_html=True,
                )
            c3, c4 = st.columns(2)
            with c3:
                st.markdown(
                    metric_card(
                        "✅", "Carbon Saved",
                        mv("green", f"{saved} kg CO2"),
                    ),
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    metric_card(
                        "🌳", "Tree Equivalent",
                        mv("green", f"{trees}/yr"),
                    ),
                    unsafe_allow_html=True,
                )
            st.markdown(
                alert(
                    "success",
                    "🌱 Optimized signal timing reduces vehicle idle "
                    "time, cutting fuel consumption and CO2 emissions.",
                )
                + alert(
                    "info",
                    "📊 CO2: 0.21 kg/km · Idle fuel: 0.6 L/hr "
                    "· Tree absorbs 21.77 kg CO2/year",
                ),
                unsafe_allow_html=True,
            )

        with t6:
            st.markdown(
                section("🧠 AI Insights"), unsafe_allow_html=True
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    metric_card(
                        "🎯", "Confidence",
                        mv("green", f"{confidence}%"),
                        "across 80 trees",
                    ),
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    metric_card(
                        "🌲", "Model Type",
                        mv("blue", "RF"),
                        "Random Forest",
                    ),
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    metric_card(
                        "⚠️", "Accident Risk",
                        mv(risk_color, f"{risk_val}%"),
                        risk_label,
                    ),
                    unsafe_allow_html=True,
                )
            st.markdown(
                alert(
                    "info",
                    "🤖 <b>Confidence:</b> 80 trees vote on the "
                    "prediction. Agreement = high confidence.",
                )
                + alert(
                    "warning",
                    f"⚠️ <b>Risk factors:</b> Rain · Traffic density"
                    f" · Night hours · Cloud cover. "
                    f"Current: <b>{risk_label} ({risk_val}%)</b>",
                )
                + alert(
                    "success",
                    f"✅ <b>Best travel time:</b> {best_hour}:00<br>"
                    f"🚫 <b>Avoid:</b> {avoid_window}",
                ),
                unsafe_allow_html=True,
            )

        st.balloons()

# ================================================
# TAB 2 — JOURNEY SIMULATION
# ================================================
with main_tab2:

    st.markdown(
        """
    <div class="hero-banner" style="margin-top:14px">
        <div class="hero-title" style="font-size:1.55rem">
            🛣️ JOURNEY SIMULATION
        </div>
        <div class="hero-sub">
            Select Your Start &amp; Destination
            · AI Analyzes Every Checkpoint
        </div>
    </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        section("📍 Select Your Route")
        + '<div class="route-select-card">'
        '<div class="route-select-title">'
        "🗺️ Choose Start &amp; Destination</div></div>",
        unsafe_allow_html=True,
    )

    rc1, rc2 = st.columns(2)
    with rc1:
        start_loc = st.selectbox(
            "📍 Start Location", LOCATION_NAMES, index=0
        )
    with rc2:
        dest_options = [
            loc for loc in LOCATION_NAMES if loc != start_loc
        ]
        destination_loc = st.selectbox(
            "🏁 Destination",
            dest_options,
            index=len(dest_options) - 1,
        )

    start_km = LOCATIONS[start_loc]["km_from_start"]
    dest_km = LOCATIONS[destination_loc]["km_from_start"]
    total_km = abs(dest_km - start_km)

    st.markdown(
        f'<div class="journey-bar">'
        f'<div style="font-family:\'Exo 2\',sans-serif;'
        f"color:rgba(160,220,255,0.4);font-size:0.7rem;"
        f"letter-spacing:2px;text-transform:uppercase;"
        f'margin-bottom:9px;">'
        f"Route: {start_loc} &rarr; {destination_loc}"
        f" &nbsp;&middot;&nbsp; {total_km} km</div>"
        f'<div class="journey-track">'
        f'<div class="journey-dot"></div>'
        f'<div class="journey-dot-dim"></div>'
        f'<div class="journey-dot-dim"></div>'
        f'<div class="journey-dot" style="background:#FF4D6D;'
        f'box-shadow:0 0 8px #FF4D6D;"></div>'
        f"</div>"
        f'<div style="display:flex;justify-content:space-between;'
        f"margin-top:7px;font-family:'Exo 2',sans-serif;"
        f'color:rgba(160,220,255,0.32);font-size:0.68rem;">'
        f"<span>📍 {start_loc}</span>"
        f"<span>🏁 {destination_loc}</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    run_btn = st.button(
        "🚀  RUN JOURNEY SIMULATION", key="journey_btn"
    )

    if run_btn:
        if start_loc == destination_loc:
            st.markdown(
                alert(
                    "warning",
                    "⚠️ Start and destination cannot be the same. "
                    "Please select different locations.",
                ),
                unsafe_allow_html=True,
            )
        else:
            checkpoints = build_checkpoints(
                start_loc, destination_loc
            )
            if len(checkpoints) < 2:
                st.markdown(
                    alert(
                        "warning",
                        "⚠️ Not enough checkpoints between selected "
                        "locations. Try a longer route.",
                    ),
                    unsafe_allow_html=True,
                )
            else:
                results = []
                for cp in checkpoints:
                    inp = {
                        "temp": cp["temp"],
                        "rain_1h": cp["rain_1h"],
                        "snow_1h": cp["snow_1h"],
                        "clouds_all": cp["clouds_all"],
                        "hour": cp["hour"],
                        "day": 15,
                        "month": cp["month"],
                        "day_of_week": cp["day_of_week"],
                    }
                    pred = predict_traffic(inp)
                    lv, st_t, em, col = get_level(pred)
                    pl = get_signal_plan(st_t)
                    rt = get_route_suggestion(lv)
                    rv, rl_label, rc_color = get_accident_risk(
                        cp["rain_1h"],
                        pred,
                        cp["hour"],
                        cp["clouds_all"],
                    )
                    results.append({
                        "cp": cp, "pred": pred,
                        "level": lv, "sig_time": st_t,
                        "emoji": em, "color": col,
                        "plan": pl, "routes": rt,
                        "co2": estimate_co2(pred),
                        "fuel": estimate_fuel_waste(st_t, pred),
                        "saved": carbon_saved(st_t, pred),
                        "risk_val": rv,
                        "risk_label": rl_label,
                        "risk_color": rc_color,
                    })

                score, score_label, score_color = get_journey_score(
                    results
                )
                score_col = (
                    "#00FF96"
                    if score_color == "green"
                    else (
                        "#FFD700"
                        if score_color == "yellow"
                        else "#FF4D6D"
                    )
                )
                n_cp = len(checkpoints)

                st.markdown(
                    f'<div class="summary-card" style="margin:14px 0">'
                    f'<div class="summary-title">'
                    f"🏆 Journey Health Score</div>"
                    f'<div style="display:flex;align-items:center;'
                    f'gap:28px;flex-wrap:wrap;">'
                    f'<div style="text-align:center">'
                    f'<div style="font-family:\'Orbitron\',monospace;'
                    f"font-size:3rem;font-weight:900;"
                    f'color:{score_col}">{score}</div>'
                    f'<div style="font-family:\'Exo 2\',sans-serif;'
                    f"color:rgba(160,220,255,0.45);font-size:0.72rem;"
                    f'letter-spacing:2px;text-transform:uppercase;">'
                    f"out of 100</div></div>"
                    f"<div>"
                    f'<div style="font-family:\'Orbitron\',monospace;'
                    f'font-size:1.1rem;color:#fff;margin-bottom:5px;">'
                    f"{score_label}</div>"
                    f'<div style="font-family:\'Exo 2\',sans-serif;'
                    f"color:rgba(160,220,255,0.58);font-size:0.83rem;"
                    f'max-width:380px;">'
                    f"{start_loc} &rarr; {destination_loc} "
                    f"&middot; {total_km} km "
                    f"&middot; {n_cp} checkpoints analyzed."
                    f"</div></div></div></div>",
                    unsafe_allow_html=True,
                )

                jt1, jt2, jt3 = st.tabs([
                    "🛣️ Checkpoint Details",
                    "📈 Journey Charts",
                    "🌍 Eco Summary",
                ])

                with jt1:
                    st.markdown(
                        section("📍 Checkpoints"),
                        unsafe_allow_html=True,
                    )
                    for r in results:
                        cp = r["cp"]
                        col = r["color"]
                        vol_cls = (
                            "cp-val-green"
                            if col == "green"
                            else (
                                "cp-val-yellow"
                                if col == "yellow"
                                else "cp-val-red"
                            )
                        )
                        risk_cls = (
                            "cp-val-green"
                            if r["risk_color"] == "green"
                            else (
                                "cp-val-yellow"
                                if r["risk_color"] == "yellow"
                                else "cp-val-red"
                            )
                        )
                        e1, n1, s1 = r["routes"]["main"]
                        e2, n2, s2 = r["routes"]["alt1"]
                        e3, n3, s3 = r["routes"]["alt2"]
                        st.markdown(
                            f'<div class="checkpoint-card '
                            f'checkpoint-card-{col}">'
                            f'<div class="cp-header"><div>'
                            f'<div class="cp-name">'
                            f"📍 {cp['name']}</div>"
                            f'<div class="cp-meta">'
                            f"KM {cp['km_from_start']} &nbsp;"
                            f"&middot;&nbsp; {cp['hour']}:00 hrs"
                            f" &nbsp;&middot;&nbsp;"
                            f" 🌧️ {cp['rain_1h']}mm</div>"
                            f'<div class="cp-scenario">'
                            f"💬 {cp['scenario']}</div></div>"
                            f'<div class="cp-badge-{col}">'
                            f"{r['emoji']} {r['level']}</div></div>"
                            f'<div class="cp-metrics">'
                            f'<div class="cp-metric">'
                            f'<div class="cp-metric-label">Volume</div>'
                            f'<div class="{vol_cls}">'
                            f"{r['pred']:,}</div></div>"
                            f'<div class="cp-metric">'
                            f'<div class="cp-metric-label">'
                            f"🟢 Signal</div>"
                            f'<div class="cp-val-green">'
                            f"{r['sig_time']}s</div></div>"
                            f'<div class="cp-metric">'
                            f'<div class="cp-metric-label">'
                            f"🔴 Signal</div>"
                            f'<div class="cp-val-red">'
                            f"{r['plan']['Red']}s</div></div>"
                            f'<div class="cp-metric">'
                            f'<div class="cp-metric-label">Risk</div>'
                            f'<div class="{risk_cls}">'
                            f"{r['risk_val']}%</div></div>"
                            f'<div class="cp-metric">'
                            f'<div class="cp-metric-label">CO2</div>'
                            f'<div class="cp-val-yellow">'
                            f"{r['co2']}kg</div></div>"
                            f'<div class="cp-metric">'
                            f'<div class="cp-metric-label">Saved</div>'
                            f'<div class="cp-val-blue">'
                            f"{r['saved']}kg</div></div></div>"
                            f'<div class="cp-route">'
                            f"🛣️ {e1} {n1}&mdash;{s1} &nbsp;|&nbsp;"
                            f" {e2} {n2}&mdash;{s2} &nbsp;|&nbsp;"
                            f" {e3} {n3}&mdash;{s3}<br>"
                            f"🧭 {r['routes']['rec']}"
                            f" &nbsp;⏱️ {r['routes']['saved']}"
                            f"</div></div>",
                            unsafe_allow_html=True,
                        )

                with jt2:
                    st.markdown(
                        section("📈 Traffic Volume"),
                        unsafe_allow_html=True,
                    )
                    cdf = pd.DataFrame({
                        "Checkpoint": [
                            r["cp"]["name"] for r in results
                        ],
                        "Traffic Volume": [
                            r["pred"] for r in results
                        ],
                    }).set_index("Checkpoint")
                    st.line_chart(cdf)

                    st.markdown(
                        section("⚠️ Accident Risk"),
                        unsafe_allow_html=True,
                    )
                    rdf = pd.DataFrame({
                        "Checkpoint": [
                            r["cp"]["name"] for r in results
                        ],
                        "Risk Score (%)": [
                            r["risk_val"] for r in results
                        ],
                    }).set_index("Checkpoint")
                    st.line_chart(rdf)

                    st.markdown(
                        section("🌧️ Weather"),
                        unsafe_allow_html=True,
                    )
                    wdf = pd.DataFrame({
                        "Checkpoint": [
                            r["cp"]["name"] for r in results
                        ],
                        "Rain (mm)": [
                            r["cp"]["rain_1h"] for r in results
                        ],
                        "Clouds (%)": [
                            r["cp"]["clouds_all"] for r in results
                        ],
                    }).set_index("Checkpoint")
                    st.line_chart(wdf)

                    st.markdown(
                        section("📋 Data Table"),
                        unsafe_allow_html=True,
                    )
                    tbl = pd.DataFrame([{
                        "Checkpoint": r["cp"]["name"],
                        "KM": r["cp"]["km_from_start"],
                        "Hour": r["cp"]["hour"],
                        "Volume": r["pred"],
                        "Level": r["level"],
                        "Signal(s)": r["sig_time"],
                        "Risk %": r["risk_val"],
                        "CO2 kg": r["co2"],
                        "Saved kg": r["saved"],
                    } for r in results])
                    st.dataframe(
                        tbl,
                        use_container_width=True,
                        hide_index=True,
                    )

                with jt3:
                    st.markdown(
                        section("🌍 Eco Summary"),
                        unsafe_allow_html=True,
                    )
                    total_co2 = round(
                        sum(r["co2"] for r in results), 2
                    )
                    total_fuel = round(
                        sum(r["fuel"] for r in results), 4
                    )
                    total_saved = round(
                        sum(r["saved"] for r in results), 4
                    )
                    total_trees = round(total_saved / 21.77, 2)
                    avg_vol = int(
                        np.mean([r["pred"] for r in results])
                    )
                    worst = max(
                        results, key=lambda x: x["pred"]
                    )
                    best = min(
                        results, key=lambda x: x["pred"]
                    )
                    high_ct = sum(
                        1 for r in results if r["color"] == "red"
                    )
                    med_ct = sum(
                        1 for r in results
                        if r["color"] == "yellow"
                    )
                    low_ct = sum(
                        1 for r in results if r["color"] == "green"
                    )
                    max_risk = max(
                        results, key=lambda x: x["risk_val"]
                    )
                    sc_key = (
                        "summary-val-green"
                        if score_color == "green"
                        else (
                            "summary-val-yellow"
                            if score_color == "yellow"
                            else "summary-val-red"
                        )
                    )
                    st.markdown(
                        f'<div class="summary-card">'
                        f'<div class="summary-title">'
                        f"🌱 Environmental Impact</div>"
                        f'<div class="summary-grid">'
                        f'<div class="summary-item">'
                        f'<div class="summary-icon">💨</div>'
                        f'<div class="summary-label">Total CO2</div>'
                        f'<div class="summary-val-red">'
                        f"{total_co2}kg</div></div>"
                        f'<div class="summary-item">'
                        f'<div class="summary-icon">⛽</div>'
                        f'<div class="summary-label">Fuel Wasted</div>'
                        f'<div class="summary-val-red">'
                        f"{total_fuel}L</div></div>"
                        f'<div class="summary-item">'
                        f'<div class="summary-icon">✅</div>'
                        f'<div class="summary-label">'
                        f"Carbon Saved</div>"
                        f'<div class="summary-val-green">'
                        f"{total_saved}kg</div></div>"
                        f'<div class="summary-item">'
                        f'<div class="summary-icon">🌳</div>'
                        f'<div class="summary-label">'
                        f"Tree Equivalent</div>"
                        f'<div class="summary-val-green">'
                        f"{total_trees}</div></div>"
                        f'<div class="summary-item">'
                        f'<div class="summary-icon">🚗</div>'
                        f'<div class="summary-label">Avg Traffic</div>'
                        f'<div class="summary-val-blue">'
                        f"{avg_vol:,}</div></div>"
                        f'<div class="summary-item">'
                        f'<div class="summary-icon">🏆</div>'
                        f'<div class="summary-label">'
                        f"Journey Score</div>"
                        f'<div class="{sc_key}">'
                        f"{score}/100</div></div>"
                        f"</div></div>"
                        f'<div class="summary-card" '
                        f'style="margin-top:12px">'
                        f'<div class="summary-title">'
                        f"📍 Highlights</div>"
                        f'<div class="summary-grid">'
                        f'<div class="summary-item">'
                        f'<div class="summary-icon">⚠️</div>'
                        f'<div class="summary-label">Worst Point</div>'
                        f'<div class="summary-val-red" '
                        f'style="font-size:0.75rem">'
                        f"{worst['cp']['name']}</div></div>"
                        f'<div class="summary-item">'
                        f'<div class="summary-icon">✅</div>'
                        f'<div class="summary-label">Best Point</div>'
                        f'<div class="summary-val-green" '
                        f'style="font-size:0.75rem">'
                        f"{best['cp']['name']}</div></div>"
                        f'<div class="summary-item">'
                        f'<div class="summary-icon">🚨</div>'
                        f'<div class="summary-label">'
                        f"Highest Risk</div>"
                        f'<div class="summary-val-red" '
                        f'style="font-size:0.75rem">'
                        f"{max_risk['cp']['name']} "
                        f"({max_risk['risk_val']}%)</div></div>"
                        f'<div class="summary-item">'
                        f'<div class="summary-icon">🔴</div>'
                        f'<div class="summary-label">High Zones</div>'
                        f'<div class="summary-val-red">'
                        f"{high_ct}/{n_cp}</div></div>"
                        f'<div class="summary-item">'
                        f'<div class="summary-icon">🟡</div>'
                        f'<div class="summary-label">'
                        f"Medium Zones</div>"
                        f'<div class="summary-val-yellow">'
                        f"{med_ct}/{n_cp}</div></div>"
                        f'<div class="summary-item">'
                        f'<div class="summary-icon">🟢</div>'
                        f'<div class="summary-label">Low Zones</div>'
                        f'<div class="summary-val-green">'
                        f"{low_ct}/{n_cp}</div></div>"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )

                st.balloons()
