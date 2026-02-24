import streamlit as st
import pandas as pd
import numpy as np
import joblib
import urllib.request
import json

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
# LOAD MODEL (cached)
# -----------------------------


@st.cache_resource
def load_model():
    m = joblib.load("traffic_model.pkl")
    fc = joblib.load("feature_columns.pkl")
    return m, fc


model, feature_columns = load_model()

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
        "lat": 12.9716, "lon": 77.5946,
    },
    "Kengeri": {
        "km_from_start": 18,
        "hour": 8, "month": 6, "day_of_week": 1,
        "temp": 294.0, "rain_1h": 2.0,
        "snow_1h": 0.0, "clouds_all": 70,
        "scenario": "Light Rain — Visibility reduced",
        "lat": 12.9141, "lon": 77.4822,
    },
    "Bidadi": {
        "km_from_start": 40,
        "hour": 9, "month": 6, "day_of_week": 1,
        "temp": 292.0, "rain_1h": 8.0,
        "snow_1h": 0.0, "clouds_all": 90,
        "scenario": "Heavy Rain — Peak congestion",
        "lat": 12.7986, "lon": 77.3886,
    },
    "Ramanagara": {
        "km_from_start": 50,
        "hour": 10, "month": 6, "day_of_week": 1,
        "temp": 294.0, "rain_1h": 0.0,
        "snow_1h": 0.0, "clouds_all": 35,
        "scenario": "Mid-morning — Silk town traffic",
        "lat": 12.7157, "lon": 77.2801,
    },
    "Channapatna": {
        "km_from_start": 60,
        "hour": 10, "month": 6, "day_of_week": 1,
        "temp": 293.0, "rain_1h": 3.0,
        "snow_1h": 0.0, "clouds_all": 60,
        "scenario": "Rain Clearing — Traffic moderating",
        "lat": 12.6512, "lon": 77.2063,
    },
    "Tumkur": {
        "km_from_start": 70,
        "hour": 9, "month": 6, "day_of_week": 1,
        "temp": 296.0, "rain_1h": 1.0,
        "snow_1h": 0.0, "clouds_all": 50,
        "scenario": "Morning — Industrial area traffic",
        "lat": 13.3379, "lon": 77.1173,
    },
    "Maddur": {
        "km_from_start": 90,
        "hour": 11, "month": 6, "day_of_week": 1,
        "temp": 296.0, "rain_1h": 0.0,
        "snow_1h": 0.0, "clouds_all": 30,
        "scenario": "Clear Weather — Smooth highway flow",
        "lat": 12.5837, "lon": 77.0435,
    },
    "Mandya": {
        "km_from_start": 100,
        "hour": 12, "month": 6, "day_of_week": 1,
        "temp": 299.0, "rain_1h": 0.0,
        "snow_1h": 0.0, "clouds_all": 20,
        "scenario": "Midday — Town traffic near market",
        "lat": 12.5218, "lon": 76.8951,
    },
    "Srirangapatna": {
        "km_from_start": 125,
        "hour": 13, "month": 6, "day_of_week": 1,
        "temp": 300.0, "rain_1h": 0.0,
        "snow_1h": 0.0, "clouds_all": 15,
        "scenario": "Tourist Area — Afternoon crowd",
        "lat": 12.4244, "lon": 76.6899,
    },
    "Mysore City": {
        "km_from_start": 140,
        "hour": 14, "month": 6, "day_of_week": 1,
        "temp": 298.0, "rain_1h": 0.0,
        "snow_1h": 0.0, "clouds_all": 25,
        "scenario": "Destination — City entry traffic",
        "lat": 12.2958, "lon": 76.6394,
    },
    "Hassan": {
        "km_from_start": 180,
        "hour": 15, "month": 6, "day_of_week": 1,
        "temp": 291.0, "rain_1h": 5.0,
        "snow_1h": 0.0, "clouds_all": 80,
        "scenario": "Evening — Moderate highway traffic",
        "lat": 13.0074, "lon": 76.1004,
    },
    "Hubli": {
        "km_from_start": 410,
        "hour": 16, "month": 6, "day_of_week": 1,
        "temp": 300.0, "rain_1h": 0.0,
        "snow_1h": 0.0, "clouds_all": 20,
        "scenario": "Afternoon — Major city traffic",
        "lat": 15.3647, "lon": 75.1240,
    },
}

LOCATION_NAMES = list(LOCATIONS.keys())

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------


def estimate_co2(traffic_volume):
    return round(traffic_volume * 0.21, 2)


def estimate_fuel_waste(signal_time, traffic_volume):
    return round(traffic_volume * 0.6 * (signal_time / 3600), 4)


def carbon_saved(optimized_time, traffic_volume):
    saved = (
        traffic_volume * 0.6
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
    return "High Traffic", 90, "🔴", "red"


def get_action_plan(level):
    plans = {
        "Low Traffic": (
            "✅ Normal signal operation. No intervention needed. "
            "Energy-saving mode can be activated."
        ),
        "Medium Traffic": (
            "⚡ Activate adaptive signal mode. "
            "Reduce red light duration on parallel roads. "
            "Monitor closely for escalation."
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
            "rec": "Switch to Old Mysore Road.",
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


@st.cache_data(ttl=300)
def get_hourly_forecast(base_input_tuple):
    base_input = dict(base_input_tuple)
    records = []
    for h in range(24):
        row = base_input.copy()
        row["hour"] = h
        records.append(row)
    fdf = pd.DataFrame(records)
    for col in feature_columns:
        if col not in fdf.columns:
            fdf[col] = 0
    return model.predict(fdf[feature_columns])


@st.cache_data(ttl=300)
def predict_traffic(input_tuple):
    input_data = dict(input_tuple)
    idf = pd.DataFrame([input_data])
    for col in feature_columns:
        if col not in idf.columns:
            idf[col] = 0
    return int(model.predict(idf[feature_columns])[0])


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
    return risk, "High Risk", "red"


def get_prediction_confidence(idf):
    preds = [t.predict(idf)[0] for t in model.estimators_]
    mean_p = np.mean(preds)
    std_p = np.std(preds)
    conf = max(0, min(100, round(100 - (std_p / mean_p) * 100, 1)))
    return conf


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
    return score, "Difficult Journey", "red"


def get_best_travel_time(hourly_preds):
    min_hour = int(np.argmin(hourly_preds))
    peak_hours = [i for i, v in enumerate(hourly_preds) if v > 5000]
    avoid = (
        f"{peak_hours[0]}:00 - {peak_hours[-1] + 1}:00"
        if peak_hours else "No peak window today"
    )
    return min_hour, avoid


def build_checkpoints(start, destination):
    s_km = LOCATIONS[start]["km_from_start"]
    d_km = LOCATIONS[destination]["km_from_start"]
    lo, hi = min(s_km, d_km), max(s_km, d_km)
    cps = []
    for name, loc in LOCATIONS.items():
        km = loc["km_from_start"]
        # Always include start and end, include intermediate points
        if name == start or name == destination or (lo < km < hi):
            cp = loc.copy()
            cp["name"] = name
            cps.append(cp)
    cps.sort(key=lambda x: x["km_from_start"])
    # Ensure correct order if travelling in reverse direction
    if s_km > d_km:
        cps = cps[::-1]
    return cps


def get_optimized_solution(results, start, dest, total_km):
    high_zones = [r for r in results if r["color"] == "red"]
    med_zones = [r for r in results if r["color"] == "yellow"]
    max_risk = max(results, key=lambda x: x["risk_val"])
    best_dep = None
    best_score = -1

    for h in range(5, 20):
        s = 0
        for r in results:
            inp = r["cp"].copy()
            p = predict_traffic(tuple({
                "temp": inp["temp"],
                "rain_1h": inp["rain_1h"],
                "snow_1h": inp["snow_1h"],
                "clouds_all": inp["clouds_all"],
                "hour": h, "day": 15,
                "month": inp["month"],
                "day_of_week": inp["day_of_week"],
            }.items()))
            s += p
        if best_dep is None or s < best_score:
            best_score = s
            best_dep = h

    tips = []
    if high_zones:
        names = ", ".join(z["cp"]["name"] for z in high_zones)
        tips.append(
            f"🚨 High traffic at: <b>{names}</b> — "
            "use alternate route or travel earlier."
        )
    if max_risk["risk_val"] > 50:
        tips.append(
            f"⚠️ Highest accident risk at "
            f"<b>{max_risk['cp']['name']}</b> "
            f"({max_risk['risk_val']}%) — "
            "reduce speed and stay alert."
        )
    if any(r["cp"]["rain_1h"] > 5 for r in results):
        tips.append(
            "🌧️ Heavy rain expected on route — "
            "maintain safe following distance."
        )
    if not tips:
        tips.append(
            "✅ Route conditions are generally good. "
            "Normal driving recommended."
        )

    total_saved = round(sum(r["saved"] for r in results), 4)
    total_co2 = round(sum(r["co2"] for r in results), 2)
    trees = round(total_saved / 21.77, 2)

    return {
        "best_departure": best_dep,
        "tips": tips,
        "total_saved": total_saved,
        "total_co2": total_co2,
        "trees": trees,
        "high_count": len(high_zones),
        "med_count": len(med_zones),
    }


# ── Live weather fetch via Open-Meteo (free, no API key) ──
@st.cache_data(ttl=600)
def fetch_live_weather(lat, lon):
    """Fetch current weather from Open-Meteo API."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,precipitation,"
        f"cloudcover,weathercode"
        f"&timezone=Asia%2FKolkata"
    )
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        cur = data["current"]
        temp_c = cur["temperature_2m"]
        temp_k = round(temp_c + 273.15, 2)
        rain = cur["precipitation"]
        clouds = cur["cloudcover"]
        return {
            "temp_k": temp_k,
            "temp_c": temp_c,
            "rain_1h": rain,
            "clouds_all": clouds,
            "ok": True,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# -----------------------------
# HTML / CSS HELPERS
# -----------------------------

def mv(color, text):
    cls = (
        "metric-value" if color == "green"
        else f"metric-value-{color}"
    )
    return f'<div class="{cls}">{text}</div>'


def alert(kind, html):
    return f'<div class="alert-{kind}">{html}</div>'


def section(text):
    return f'<div class="section-header">{text}</div>'


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


def make_route_card(label, emoji, name, status):
    cmap = {
        "🟢": ("route-card-green", "route-status-green"),
        "🟡": ("route-card-yellow", "route-status-yellow"),
        "🔴": ("route-card-red", "route-status-red"),
    }
    cc, sc = cmap[emoji]
    lbl_s = (
        "font-family:'Exo 2',sans-serif;"
        "color:rgba(160,220,255,0.38);font-size:0.68rem;"
        "letter-spacing:2px;text-transform:uppercase;"
        "margin-bottom:3px;"
    )
    return (
        f'<div class="route-card {cc}">'
        f'<div><div style="{lbl_s}">{label}</div>'
        f'<div class="route-name">{emoji} {name}</div></div>'
        f'<div class="{sc}">{status}</div></div>'
    )


def color_val_class(color):
    if color == "green":
        return "cp-val-green"
    elif color == "yellow":
        return "cp-val-yellow"
    return "cp-val-red"


def summary_val_class(color):
    if color == "green":
        return "summary-val-green"
    elif color == "yellow":
        return "summary-val-yellow"
    return "summary-val-red"


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
        135deg, #020b14 0%, #061a2e 50%, #020b14 100%);
    border: 1px solid rgba(0,255,150,0.2);
    border-radius: 20px; padding: 38px 50px;
    margin-bottom: 10px; text-align: center;
    position: relative; overflow: hidden;
}
.hero-banner::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(
        90deg, transparent, #00FF96, #00BFFF, transparent);
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
.badge-pink {
    color: #f472b6; border-color: rgba(244,114,182,0.4);
    background: rgba(244,114,182,0.08);
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
        rgba(0,255,150,0.15), rgba(0,191,255,0.15)
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
        90deg, transparent, #00BFFF, transparent);
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
        90deg, transparent, #00FF96, transparent);
}
.route-select-title {
    font-family: 'Orbitron', monospace; color: #00FF96;
    font-size: 0.78rem; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 6px;
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
        90deg, transparent, #00FF96, transparent);
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
    font-size: 0.88rem; line-height: 1.7;
}
.alert-warning {
    background: rgba(255,215,0,0.08);
    border: 1px solid rgba(255,215,0,0.3);
    border-left: 4px solid #FFD700; border-radius: 10px;
    padding: 13px 17px; font-family: 'Exo 2', sans-serif;
    color: rgba(255,240,180,0.9); margin: 9px 0;
    font-size: 0.88rem; line-height: 1.7;
}
.alert-danger {
    background: rgba(255,77,109,0.08);
    border: 1px solid rgba(255,77,109,0.3);
    border-left: 4px solid #FF4D6D; border-radius: 10px;
    padding: 13px 17px; font-family: 'Exo 2', sans-serif;
    color: rgba(255,200,210,0.9); margin: 9px 0;
    font-size: 0.88rem; line-height: 1.7;
}
.alert-info {
    background: rgba(0,191,255,0.08);
    border: 1px solid rgba(0,191,255,0.3);
    border-left: 4px solid #00BFFF; border-radius: 10px;
    padding: 13px 17px; font-family: 'Exo 2', sans-serif;
    color: rgba(180,230,255,0.9); margin: 9px 0;
    font-size: 0.88rem; line-height: 1.7;
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
.signal-light-dim { background: #1a2a3a; }
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
        90deg, transparent, #00FF96, transparent);
}
.checkpoint-card-yellow::before {
    background: linear-gradient(
        90deg, transparent, #FFD700, transparent);
}
.checkpoint-card-red::before {
    background: linear-gradient(
        90deg, transparent, #FF4D6D, transparent);
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
        90deg, transparent, #00FF96, #00BFFF, transparent);
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
.journey-dot {
    width: 12px; height: 12px; border-radius: 50%;
    background: #00FF96; box-shadow: 0 0 8px #00FF96;
    z-index: 1; flex-shrink: 0;
}
.journey-dot-dim {
    width: 8px; height: 8px; border-radius: 50%;
    background: rgba(0,191,255,0.3); z-index: 1; flex-shrink: 0;
}

.feature-card {
    background: linear-gradient(135deg, #061a2e, #071f38);
    border: 1px solid rgba(0,191,255,0.2);
    border-radius: 16px; padding: 22px;
    text-align: center; margin-bottom: 12px;
    position: relative; overflow: hidden;
}
.feature-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(
        90deg, transparent, #00BFFF, transparent);
}
.feature-icon { font-size: 2rem; margin-bottom: 10px; }
.feature-title {
    font-family: 'Orbitron', monospace; color: #00FF96;
    font-size: 0.8rem; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 8px;
}
.feature-desc {
    font-family: 'Exo 2', sans-serif;
    color: rgba(160,220,255,0.65); font-size: 0.82rem;
    line-height: 1.6;
}

.stat-row {
    display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0;
}
.stat-box {
    background: rgba(2,11,20,0.7);
    border: 1px solid rgba(0,255,150,0.2);
    border-radius: 12px; padding: 16px 20px;
    flex: 1; min-width: 130px; text-align: center;
}
.stat-val {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem; font-weight: 700; color: #00FF96;
}
.stat-label {
    font-family: 'Exo 2', sans-serif;
    color: rgba(160,220,255,0.5); font-size: 0.7rem;
    letter-spacing: 1px; text-transform: uppercase;
    margin-top: 4px;
}

.opt-solution-card {
    background: linear-gradient(135deg, #061a2e, #071f38);
    border: 2px solid rgba(0,255,150,0.3);
    border-radius: 18px; padding: 28px; margin: 16px 0;
    position: relative; overflow: hidden;
}
.opt-solution-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #00FF96, #00BFFF, #00FF96);
}
.opt-title {
    font-family: 'Orbitron', monospace; color: #00FF96;
    font-size: 1rem; letter-spacing: 3px;
    text-transform: uppercase; margin-bottom: 20px;
}

/* Live Weather Feature Styles */
.weather-card {
    background: linear-gradient(135deg, #061a2e, #071428);
    border: 1px solid rgba(0,255,150,0.3);
    border-radius: 18px; padding: 24px; margin-bottom: 18px;
    position: relative; overflow: hidden;
}
.weather-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(
        90deg, transparent, #00FF96, #00BFFF, transparent);
}
.weather-title {
    font-family: 'Orbitron', monospace; color: #00FF96;
    font-size: 0.9rem; letter-spacing: 3px;
    text-transform: uppercase; margin-bottom: 18px;
}
.weather-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 12px; margin-bottom: 18px;
}
.weather-item {
    background: rgba(2,11,20,0.7);
    border: 1px solid rgba(0,191,255,0.2);
    border-radius: 12px; padding: 14px; text-align: center;
}
.weather-val {
    font-family: 'Orbitron', monospace; color: #00BFFF;
    font-size: 1.3rem; font-weight: 700;
}
.weather-label {
    font-family: 'Exo 2', sans-serif;
    color: rgba(160,220,255,0.5); font-size: 0.68rem;
    letter-spacing: 1px; text-transform: uppercase;
    margin-top: 4px;
}

.stSlider label {
    font-family: 'Exo 2', sans-serif !important;
    color: rgba(160,220,255,0.8) !important;
    font-size: 0.83rem !important;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ================================================
# NAVIGATION
# ================================================
pages = st.tabs([
    "🏠 Home",
    "🎛️ Live Predictor",
    "🌤️ Live Weather Predict",
    "🛣️ Journey Planner",
    "🔬 Model Insights",
    "🧪 What-If Simulator",
])

pg_home, pg_pred, pg_weather, pg_journey, pg_model, pg_whatif = pages

# ================================================
# PAGE 1 — HOME
# ================================================
with pg_home:

    st.markdown(
        """
    <div class="hero-banner">
        <div class="hero-title">🌱 GREEN AI TRAFFIC OPTIMIZER</div>
        <div class="hero-sub">
            Intelligent Signal Control · Carbon Reduction
            · Route Optimization · Live Weather Prediction
        </div>
        <div class="hero-badges">
            <span class="badge badge-green">🤖 ML Powered</span>
            <span class="badge badge-blue">⚡ Real-Time</span>
            <span class="badge badge-purple">🌍 Eco Friendly</span>
            <span class="badge badge-orange">🛣️ Journey AI</span>
            <span class="badge badge-pink">🌤️ Live Weather</span>
        </div>
    </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(section("🌍 System Impact at a Glance"), unsafe_allow_html=True)
    st.markdown(
        '<div class="stat-row">'
        '<div class="stat-box">'
        '<div class="stat-val" style="color:#00FF96">12</div>'
        '<div class="stat-label">Locations Covered</div>'
        "</div>"
        '<div class="stat-box">'
        '<div class="stat-val" style="color:#00BFFF">80</div>'
        '<div class="stat-label">Decision Trees</div>'
        "</div>"
        '<div class="stat-box">'
        '<div class="stat-val" style="color:#00FF96">94%</div>'
        '<div class="stat-label">Model R² Accuracy</div>'
        "</div>"
        '<div class="stat-box">'
        '<div class="stat-val" style="color:#FFD700">6</div>'
        '<div class="stat-label">App Pages</div>'
        "</div>"
        '<div class="stat-box">'
        '<div class="stat-val" style="color:#c084fc">Live</div>'
        '<div class="stat-label">Weather Data</div>'
        "</div>"
        '<div class="stat-box">'
        '<div class="stat-val" style="color:#00FF96">CO2</div>'
        '<div class="stat-label">Eco Tracking</div>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(section("🚀 What This System Does"), unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-icon">🎛️</div>'
            '<div class="feature-title">Live Predictor</div>'
            '<div class="feature-desc">'
            "Enter weather and time conditions to predict "
            "traffic volume and get signal timing, route "
            "suggestions, and eco impact instantly."
            "</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-icon">🌤️</div>'
            '<div class="feature-title">Live Weather Predict</div>'
            '<div class="feature-desc">'
            "Select any location — real weather data is fetched "
            "automatically. AI predicts traffic using live temp, "
            "rain and cloud conditions right now."
            "</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-icon">🛣️</div>'
            '<div class="feature-title">Journey Planner</div>'
            '<div class="feature-desc">'
            "Select start and destination from 12 locations. "
            "AI predicts every checkpoint and delivers an "
            "optimized travel solution with departure time."
            "</div></div>",
            unsafe_allow_html=True,
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-icon">📊</div>'
            '<div class="feature-title">Live Dashboard</div>'
            '<div class="feature-desc">'
            "All key metrics in one view — traffic level, "
            "signal plan, eco impact, risk score and "
            "24H forecast at a glance."
            "</div></div>",
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-icon">🧪</div>'
            '<div class="feature-title">What-If Simulator</div>'
            '<div class="feature-desc">'
            "Change one variable — rain, hour, temperature "
            "— and see how traffic and eco impact changes. "
            "Compare two scenarios side by side."
            "</div></div>",
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-icon">🌍</div>'
            '<div class="feature-title">Eco Impact</div>'
            '<div class="feature-desc">'
            "Every prediction includes CO2 emissions, fuel "
            "waste, carbon saved vs unoptimized signals, "
            "and tree equivalent calculations."
            "</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        section("⚙️ How It Works — Step by Step"),
        unsafe_allow_html=True
    )
    step_style_bg = [
        "rgba(0,255,150,0.06)", "rgba(0,191,255,0.06)",
        "rgba(192,132,252,0.06)", "rgba(251,146,60,0.06)",
        "rgba(0,255,150,0.06)",
    ]
    step_border = [
        "rgba(0,255,150,0.2)", "rgba(0,191,255,0.2)",
        "rgba(192,132,252,0.2)", "rgba(251,146,60,0.2)",
        "rgba(0,255,150,0.2)",
    ]
    step_color = ["#00FF96", "#00BFFF", "#c084fc", "#fb923c", "#00FF96"]
    steps = [
        ("🌦️", "INPUT", "Weather + Time conditions entered"),
        ("🌲", "MODEL", "80 Random Forest trees predict traffic volume"),
        ("🧠", "ANALYZE", "Level, signal timing, risk, confidence calculated"),
        ("🚦", "OPTIMIZE", "Signal plan + route suggestion generated"),
        ("🌍", "ECO RESULT", "CO2 saved, fuel reduced, trees equivalent shown"),
    ]
    html_steps = '<div style="display:flex;gap:0;flex-wrap:wrap;">'
    for i, (icon, title, desc) in enumerate(steps):
        html_steps += (
            f'<div style="flex:1;min-width:140px;'
            f"background:{step_style_bg[i]};"
            f"border:1px solid {step_border[i]};"
            f"border-radius:12px;padding:18px;margin:4px;"
            f'text-align:center;">'
            f'<div style="font-size:1.6rem">{icon}</div>'
            f'<div style="font-family:\'Orbitron\',monospace;'
            f"color:{step_color[i]};font-size:0.72rem;"
            f'letter-spacing:2px;margin:8px 0 5px;">{title}</div>'
            f'<div style="font-family:\'Exo 2\',sans-serif;'
            f"color:rgba(160,220,255,0.6);"
            f'font-size:0.78rem;">{desc}</div></div>'
        )
        if i < len(steps) - 1:
            html_steps += (
                '<div style="display:flex;align-items:center;'
                'padding:0 4px;font-size:1.2rem;color:#00BFFF;">→</div>'
            )
    html_steps += "</div>"
    st.markdown(html_steps, unsafe_allow_html=True)

    st.markdown(section("🤖 About The ML Model"), unsafe_allow_html=True)
    st.markdown(
        alert(
            "info",
            "Trained on the <b>Metro Interstate Traffic Volume "
            "dataset</b> — real highway traffic data with "
            "weather, holiday, and time features. "
            "The model predicts continuous traffic volume "
            "(vehicles/hr) — a regression problem.",
        ),
        unsafe_allow_html=True,
    )
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(
            metric_card("🌲", "Model", mv("green", "RF"), "Random Forest"),
            unsafe_allow_html=True,
        )
    with mc2:
        st.markdown(
            metric_card("🌳", "Trees", mv("blue", "80"), "Estimators"),
            unsafe_allow_html=True,
        )
    with mc3:
        st.markdown(
            metric_card("🎯", "R² Score", mv("green", "~0.94"), "Accuracy"),
            unsafe_allow_html=True,
        )
    with mc4:
        st.markdown(
            metric_card("📉", "MAE", mv("blue", "~280"), "Avg Error"),
            unsafe_allow_html=True,
        )

# ================================================
# PAGE 2 — LIVE PREDICTOR
# ================================================
with pg_pred:

    st.markdown(section("⚙️ Input Parameters"), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div class="input-card">'
            '<div class="input-card-title">🌦 Weather Conditions</div></div>',
            unsafe_allow_html=True,
        )
        temp = st.slider(
            "🌡️ Temperature (Kelvin)", 250.0, 330.0, 280.0,
            key="p_temp"
        )
        rain = st.slider(
            "🌧️ Rain — last 1 hour (mm)", 0.0, 50.0, 0.0,
            key="p_rain"
        )
        snow = st.slider(
            "❄️ Snow — last 1 hour (mm)", 0.0, 50.0, 0.0,
            key="p_snow"
        )
        clouds = st.slider("☁️ Cloud Coverage (%)", 0, 100, 50, key="p_clouds")
    with col2:
        st.markdown(
            '<div class="input-card">'
            '<div class="input-card-title">📅 Date &amp; Time</div></div>',
            unsafe_allow_html=True,
        )
        hour = st.slider("🕐 Hour of Day", 0, 23, 8, key="p_hour")
        day = st.slider("📆 Day of Month", 1, 31, 15, key="p_day")
        month = st.slider("🗓️ Month", 1, 12, 6, key="p_month")
        dow = st.slider("📅 Day of Week (0=Mon, 6=Sun)", 0, 6, 2, key="p_dow")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button(
        "🚀  ANALYZE TRAFFIC & GENERATE SOLUTION", key="predict_btn"
    )

    inp_data = {
        "temp": temp, "rain_1h": rain,
        "snow_1h": snow, "clouds_all": clouds,
        "hour": hour, "day": day,
        "month": month, "day_of_week": dow,
    }
    idf = pd.DataFrame([inp_data])
    for c in feature_columns:
        if c not in idf.columns:
            idf[c] = 0
    idf = idf[feature_columns]

    if predict_btn:
        prediction = int(model.predict(idf)[0])
        level, sig_t, c_emoji, lc = get_level(prediction)
        plan = get_signal_plan(sig_t)
        routes = get_route_suggestion(level)
        h_preds = get_hourly_forecast(tuple(inp_data.items()))
        avg_t = int(np.mean(h_preds))
        diff_p = round(((prediction - avg_t) / avg_t) * 100, 1)
        rv, rl, rc = get_accident_risk(rain, prediction, hour, clouds)
        conf = get_prediction_confidence(idf)
        best_h, avoid_w = get_best_travel_time(h_preds)

        t1, t2, t3, t4, t5 = st.tabs([
            "📊 Overview", "🚦 Signal Plan",
            "🗺️ Route Guide", "📈 24H Forecast", "🌍 Eco Impact",
        ])

        with t1:
            st.markdown(section("📊 Overview"), unsafe_allow_html=True)
            tp = min(prediction / 8000, 1.0)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(
                    metric_card(
                        "🚗", "Traffic Volume",
                        mv("green", f"{prediction:,}"), "vehicles / hr",
                    ),
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    metric_card(
                        c_emoji, "Traffic Level",
                        mv(lc, level.split()[0]), level.split()[1],
                    ),
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    metric_card(
                        "⚙️", "ML Confidence",
                        mv("blue", f"{conf}%"), "across 80 trees",
                    ),
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    metric_card("⚠️", "Accident Risk", mv(rc, f"{rv}%"), rl),
                    unsafe_allow_html=True,
                )

            fc_fill = f"custom-progress-fill-{lc}"
            st.markdown(
                section("🚦 Traffic Intensity")
                + f'<div style="font-family:\'Exo 2\',sans-serif;'
                f"color:rgba(160,220,255,0.42);font-size:0.75rem;"
                f'margin-bottom:5px;">'
                f'{int(tp * 100)}% of maximum capacity</div>'
                f'<div class="custom-progress-bg">'
                f'<div class="{fc_fill}" style="width:{int(tp*100)}%"></div>'
                f"</div>",
                unsafe_allow_html=True,
            )

            al = "success" if lc == "green" else (
                "warning" if lc == "yellow" else "danger"
            )
            st.markdown(
                section("📋 Action Plan") + alert(al, get_action_plan(level)),
                unsafe_allow_html=True,
            )

            if diff_p > 15:
                st.markdown(
                    alert(
                        "warning",
                        f"⚠️ Traffic is <b>{diff_p}%</b> above "
                        f"daily avg ({avg_t:,} veh/hr). "
                        "Early intervention recommended!",
                    ),
                    unsafe_allow_html=True,
                )
            elif diff_p < -15:
                st.markdown(
                    alert(
                        "success",
                        f"✅ Traffic is <b>{abs(diff_p)}%</b> "
                        "below daily avg. No intervention needed.",
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    alert(
                        "info",
                        "ℹ️ Traffic within normal range "
                        f"({diff_p:+}% vs daily avg).",
                    ),
                    unsafe_allow_html=True,
                )

        with t2:
            st.markdown(section("🚦 Signal Cycle"), unsafe_allow_html=True)
            rl_c = "signal-light-red" if lc == "red" else "signal-light-dim"
            yl_c = (
                "signal-light-yellow" if lc == "yellow"
                else "signal-light-dim"
            )
            gl_c = (
                "signal-light-green" if lc == "green"
                else "signal-light-dim"
            )
            rc_col = "#FF4D6D" if lc == "red" else "#1a3a5a"
            yc_col = "#FFD700" if lc == "yellow" else "#1a3a5a"
            gc_col = "#00FF96" if lc == "green" else "#1a3a5a"
            st.markdown(
                f'<div class="signal-container">'
                f'<div class="signal-block">'
                f'<div class="signal-light {rl_c}"></div>'
                f'<div class="signal-time" style="color:{rc_col}">'
                f"{plan['Red']}s</div>"
                f'<div class="signal-sublabel">Red</div></div>'
                f'<div class="signal-block">'
                f'<div class="signal-light {yl_c}"></div>'
                f'<div class="signal-time" style="color:{yc_col}">'
                f"{plan['Yellow']}s</div>"
                f'<div class="signal-sublabel">Yellow</div></div>'
                f'<div class="signal-block">'
                f'<div class="signal-light {gl_c}"></div>'
                f'<div class="signal-time" style="color:{gc_col}">'
                f"{plan['Green']}s</div>"
                f'<div class="signal-sublabel">Green</div></div>'
                f"</div>",
                unsafe_allow_html=True,
            )
            eff = round((plan["Green"] / plan["Total Cycle"]) * 100, 1)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    metric_card(
                        "", "Total Cycle",
                        mv("blue", f"{plan['Total Cycle']}s")
                    ),
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    metric_card(
                        "", "Green Efficiency",
                        mv("green", f"{eff}%")
                    ),
                    unsafe_allow_html=True,
                )
            st.markdown(
                alert(
                    "info",
                    "💡 Signal timing dynamically adjusted to "
                    "minimize idle time and reduce CO2 emissions.",
                ),
                unsafe_allow_html=True,
            )

        with t3:
            st.markdown(section("🗺️ Route Status"), unsafe_allow_html=True)
            e, n, s = routes["main"]
            st.markdown(
                make_route_card("Current Route", e, n, s),
                unsafe_allow_html=True
            )
            e, n, s = routes["alt1"]
            st.markdown(
                make_route_card("Alternate Route 1", e, n, s),
                unsafe_allow_html=True
            )
            e, n, s = routes["alt2"]
            st.markdown(
                make_route_card("Alternate Route 2", e, n, s),
                unsafe_allow_html=True
            )
            ra = "success" if lc == "green" else (
                "warning" if lc == "yellow" else "danger"
            )
            st.markdown(
                alert(
                    ra,
                    f"🧭 <b>Recommendation:</b> {routes['rec']}"
                    f"<br>⏱️ <b>Time Saved:</b> {routes['saved']}",
                ),
                unsafe_allow_html=True,
            )

        with t4:
            st.markdown(section("📈 24H Forecast"), unsafe_allow_html=True)
            fdf = pd.DataFrame({
                "Hour": list(range(24)),
                "Traffic Volume": h_preds.astype(int),
            })
            pk_h = int(fdf.loc[fdf["Traffic Volume"].idxmax(), "Hour"])
            pk_v = int(fdf["Traffic Volume"].max())
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(
                    metric_card("", "Best Time", mv("green", f"{best_h}:00")),
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    metric_card("", "Peak Hour", mv("red", f"{pk_h}:00")),
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    metric_card("", "Peak Volume", mv("red", f"{pk_v:,}")),
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    metric_card("", "Daily Average", mv("blue", f"{avg_t:,}")),
                    unsafe_allow_html=True,
                )
            st.markdown(
                alert(
                    "success",
                    f"✅ <b>Best time to travel:</b> {best_h}:00"
                    f"<br>⚠️ <b>Avoid:</b> {avoid_w}",
                ),
                unsafe_allow_html=True,
            )
            st.line_chart(fdf.set_index("Hour")["Traffic Volume"])

        with t5:
            st.markdown(section("🌍 Eco Impact"), unsafe_allow_html=True)
            co2 = estimate_co2(prediction)
            fuel = estimate_fuel_waste(sig_t, prediction)
            saved = carbon_saved(sig_t, prediction)
            trees = round(saved / 21.77, 2)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    metric_card(
                        "💨", "CO2 Emissions",
                        mv("red", f"{co2} kg/hr")
                    ),
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    metric_card(
                        "⛽", "Fuel Wasted",
                        mv("yellow", f"{fuel} L/hr")
                    ),
                    unsafe_allow_html=True,
                )
            c3, c4 = st.columns(2)
            with c3:
                st.markdown(
                    metric_card(
                        "✅", "Carbon Saved",
                        mv("green", f"{saved} kg")
                    ),
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    metric_card(
                        "🌳", "Tree Equivalent",
                        mv("green", f"{trees}/yr")
                    ),
                    unsafe_allow_html=True,
                )
            st.markdown(
                alert(
                    "success",
                    "🌱 Optimized signal timing reduces vehicle "
                    "idle time, cutting fuel and CO2 emissions.",
                )
                + alert(
                    "info",
                    "📊 CO2: 0.21 kg/km · Idle fuel: 0.6 L/hr"
                    " · Tree: 21.77 kg CO2/yr",
                ),
                unsafe_allow_html=True,
            )

        st.balloons()

# ================================================
# PAGE 3 — LIVE WEATHER PREDICTOR
# ================================================
with pg_weather:

    st.markdown(
        """
    <div class="hero-banner" style="margin-top:8px">
        <div class="hero-title" style="font-size:1.6rem">
            🌤️ LIVE WEATHER TRAFFIC PREDICTOR
        </div>
        <div class="hero-sub">
            Real-Time Weather · Auto Prediction · Signal Plan
            · Best Travel Window · Eco Impact
        </div>
        <div class="hero-badges">
            <span class="badge badge-green">🌍 Open-Meteo API</span>
            <span class="badge badge-blue">⚡ Live Data</span>
            <span class="badge badge-orange">🤖 Auto Predict</span>
            <span class="badge badge-purple">📈 24H Analysis</span>
            <span class="badge badge-pink">🚦 Signal Plan</span>
        </div>
    </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        alert(
            "info",
            "🌐 Fetches <b>real-time weather</b> "
            "(temperature, rainfall, cloud cover) "
            "for the selected location using the free "
            "Open-Meteo API — no API key needed. "
            "The ML model then predicts traffic and "
            "generates a complete solution: signal plan, "
            "route advice, 24H forecast, best travel "
            "windows, and eco impact.",
        ),
        unsafe_allow_html=True,
    )

    wl_c1, wl_c2, wl_c3 = st.columns([3, 2, 1])
    with wl_c1:
        weather_loc = st.selectbox(
            "📍 Select Location for Live Weather",
            LOCATION_NAMES, key="weather_loc",
        )
    with wl_c2:
        wl_dow = st.selectbox(
            "📅 Day of Week",
            ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"],
            index=0, key="wl_dow",
        )
        wl_dow_num = ["Monday", "Tuesday", "Wednesday",
                      "Thursday", "Friday", "Saturday", "Sunday"].index(wl_dow)
    with wl_c3:
        wl_hour = st.slider("🕐 Current Hour", 0, 23, 8, key="wl_hour")

    fetch_btn = st.button(
        "🌤️  FETCH LIVE WEATHER & GENERATE FULL ANALYSIS",
        key="fetch_weather_btn"
    )

    if fetch_btn:
        loc_data = LOCATIONS[weather_loc]
        with st.spinner("Fetching live weather data from Open-Meteo..."):
            wx = fetch_live_weather(loc_data["lat"], loc_data["lon"])

        if not wx["ok"]:
            st.markdown(
                alert(
                    "danger",
                    "❌ Could not fetch live weather: "
                    f"{wx.get('error', 'Unknown error')}. "
                    "Please check your internet connection and try again.",
                ),
                unsafe_allow_html=True,
            )
        else:
            temp_k = wx["temp_k"]
            temp_c = wx["temp_c"]
            rain = wx["rain_1h"]
            clouds = wx["clouds_all"]

            rain_emoji = "🌧️" if rain > 5 else ("🌦️" if rain > 0 else "☀️")
            cloud_emoji = (
                "⛅" if clouds > 50
                else ("🌤️" if clouds > 20 else "☀️")
            )
            temp_emoji = (
                "🔥" if temp_c > 35
                else ("🌡️" if temp_c > 25 else "❄️")
            )

            # ── LIVE WEATHER CARD ──────────────────────────
            st.markdown(
                f'<div class="weather-card">'
                f'<div class="weather-title">'
                f'🌍 Live Weather — {weather_loc}</div>'
                f'<div class="weather-grid">'
                f'<div class="weather-item">'
                f'<div style="font-size:2rem">{temp_emoji}</div>'
                f'<div class="weather-val">{temp_c}°C</div>'
                f'<div class="weather-label">'
                f'Temperature · {temp_k}K</div></div>'
                f'<div class="weather-item">'
                f'<div style="font-size:2rem">{rain_emoji}</div>'
                f'<div class="weather-val">{rain}mm</div>'
                f'<div class="weather-label">Rainfall (1h)</div></div>'
                f'<div class="weather-item">'
                f'<div style="font-size:2rem">{cloud_emoji}</div>'
                f'<div class="weather-val">{clouds}%</div>'
                f'<div class="weather-label">Cloud Cover</div></div>'
                f'<div class="weather-item">'
                f'<div style="font-size:2rem">🕐</div>'
                f'<div class="weather-val">{wl_hour}:00</div>'
                f'<div class="weather-label">{wl_dow}</div></div>'
                f'<div class="weather-item">'
                f'<div style="font-size:2rem">📍</div>'
                f'<div class="weather-val" '
                f'style="font-size:0.9rem">{weather_loc}</div>'
                f'<div class="weather-label">'
                f'KM {loc_data["km_from_start"]}</div></div>'
                f"</div>"
                '<div style="font-family:\'Exo 2\',sans-serif;'
                'color:rgba(160,220,255,0.4);'
                f'font-size:0.72rem;text-align:right;margin-top:8px;">'
                "Source: Open-Meteo API · Cached 10 min"
                " · Auto-refreshes on next fetch"
                f"</div></div>",
                unsafe_allow_html=True,
            )

            # ── COMPUTE ALL METRICS ────────────────────────
            live_inp = {
                "temp": temp_k,
                "rain_1h": rain,
                "snow_1h": loc_data.get("snow_1h", 0.0),
                "clouds_all": clouds,
                "hour": wl_hour, "day": 15,
                "month": loc_data["month"],
                "day_of_week": wl_dow_num,
            }
            live_pred = predict_traffic(tuple(live_inp.items()))
            live_level, live_sig, live_em, live_lc = get_level(live_pred)
            live_plan = get_signal_plan(live_sig)
            live_routes = get_route_suggestion(live_level)
            live_rv, live_rl, live_rc = get_accident_risk(
                rain, live_pred, wl_hour, clouds
            )
            live_co2 = estimate_co2(live_pred)
            live_fuel = estimate_fuel_waste(live_sig, live_pred)
            live_saved = carbon_saved(live_sig, live_pred)
            live_trees = round(live_saved / 21.77, 2)
            live_eff = round(
                (live_plan["Green"] / live_plan["Total Cycle"]) * 100, 1
            )

            # 24H forecast
            h_preds_live = get_hourly_forecast(tuple(live_inp.items()))
            best_h_live, avoid_live = get_best_travel_time(h_preds_live)
            avg_live = int(np.mean(h_preds_live))
            peak_h_live = int(np.argmax(h_preds_live))
            peak_v_live = int(np.max(h_preds_live))
            min_v_live = int(np.min(h_preds_live))

            # Confidence
            live_idf = pd.DataFrame([live_inp])
            for c in feature_columns:
                if c not in live_idf.columns:
                    live_idf[c] = 0
            live_idf = live_idf[feature_columns]
            live_conf = get_prediction_confidence(live_idf)

            # Diff vs daily avg
            diff_pct = round(((live_pred - avg_live) / avg_live) * 100, 1)

            # Build hourly classification table
            h_levels = []
            for hv in h_preds_live:
                if hv < 3000:
                    h_levels.append("🟢 Low")
                elif hv < 6000:
                    h_levels.append("🟡 Medium")
                else:
                    h_levels.append("🔴 High")

            # Good travel windows (consecutive low/medium hours)
            good_windows = []
            window_start = None
            for hi, hl in enumerate(h_levels):
                if "🔴" not in hl:
                    if window_start is None:
                        window_start = hi
                else:
                    if window_start is not None and hi - window_start >= 2:
                        good_windows.append((window_start, hi - 1))
                    window_start = None
            if window_start is not None and 23 - window_start >= 2:
                good_windows.append((window_start, 23))

            al_live = "success" if live_lc == "green" else (
                "warning" if live_lc == "yellow" else "danger"
            )

            # ── TABS ──────────────────────────────────────
            wt1, wt2, wt3, wt4, wt5 = st.tabs([
                "🚦 Traffic & Signal",
                "📈 24H Forecast",
                "🕐 Best Travel Times",
                "🗺️ Route Guide",
                "🌍 Eco Impact",
            ])

            with wt1:
                st.markdown(
                    section("🚗 Live Traffic Prediction"),
                    unsafe_allow_html=True
                )

                lv1, lv2, lv3, lv4, lv5 = st.columns(5)
                with lv1:
                    st.markdown(
                        metric_card(
                            "🚗", "Predicted Volume",
                            mv("green", f"{live_pred:,}"), "vehicles/hr",
                        ),
                        unsafe_allow_html=True,
                    )
                with lv2:
                    st.markdown(
                        metric_card(
                            live_em, "Traffic Level",
                            mv(live_lc, live_level.split()[0]),
                            live_level.split()[1],
                        ),
                        unsafe_allow_html=True,
                    )
                with lv3:
                    st.markdown(
                        metric_card(
                            "🎯", "ML Confidence",
                            mv("blue", f"{live_conf}%"), "80 trees",
                        ),
                        unsafe_allow_html=True,
                    )
                with lv4:
                    st.markdown(
                        metric_card(
                            "⚠️", "Accident Risk",
                            mv(live_rc, f"{live_rv}%"), live_rl,
                        ),
                        unsafe_allow_html=True,
                    )
                with lv5:
                    diff_c = "red" if diff_pct > 15 else (
                        "yellow" if diff_pct > 0 else "green"
                    )
                    diff_s = (
                        f"+{diff_pct}%" if diff_pct >= 0
                        else f"{diff_pct}%"
                    )
                    st.markdown(
                        metric_card(
                            "📊", "vs Daily Avg",
                            mv(diff_c, diff_s), f"avg {avg_live:,}/hr",
                        ),
                        unsafe_allow_html=True,
                    )

                # Traffic intensity bar
                tp = min(live_pred / 8000, 1.0)
                fc_fill = f"custom-progress-fill-{live_lc}"
                st.markdown(
                    section("🚦 Traffic Intensity")
                    + f'<div style="font-family:\'Exo 2\',sans-serif;'
                    f"color:rgba(160,220,255,0.42);font-size:0.75rem;"
                    f'margin-bottom:5px;">'
                    f'{int(tp*100)}% of maximum capacity</div>'
                    f'<div class="custom-progress-bg">'
                    f'<div class="{fc_fill}"'
                    f' style="width:{int(tp*100)}%"></div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    alert(al_live, get_action_plan(live_level)),
                    unsafe_allow_html=True,
                )

                st.markdown(
                    section("🚦 Optimized Signal Plan"),
                    unsafe_allow_html=True
                )
                rl_c = (
                    "signal-light-red" if live_lc == "red"
                    else "signal-light-dim"
                )
                yl_c = (
                    "signal-light-yellow" if live_lc == "yellow"
                    else "signal-light-dim"
                )
                gl_c = (
                    "signal-light-green" if live_lc == "green"
                    else "signal-light-dim"
                )
                rc_c = "#FF4D6D" if live_lc == "red" else "#1a3a5a"
                yc_c = "#FFD700" if live_lc == "yellow" else "#1a3a5a"
                gc_c = "#00FF96" if live_lc == "green" else "#1a3a5a"
                st.markdown(
                    f'<div class="signal-container">'
                    f'<div class="signal-block">'
                    f'<div class="signal-light {rl_c}"></div>'
                    f'<div class="signal-time" style="color:{rc_c}">'
                    f'{live_plan["Red"]}s</div>'
                    f'<div class="signal-sublabel">Red</div></div>'
                    f'<div class="signal-block">'
                    f'<div class="signal-light {yl_c}"></div>'
                    f'<div class="signal-time" style="color:{yc_c}">'
                    f'{live_plan["Yellow"]}s</div>'
                    f'<div class="signal-sublabel">Yellow</div></div>'
                    f'<div class="signal-block">'
                    f'<div class="signal-light {gl_c}"></div>'
                    f'<div class="signal-time" style="color:{gc_c}">'
                    f'{live_plan["Green"]}s</div>'
                    f'<div class="signal-sublabel">Green</div></div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                sg1, sg2, sg3 = st.columns(3)
                with sg1:
                    st.markdown(
                        metric_card(
                            "🔄", "Total Cycle",
                            mv("blue", f"{live_plan['Total Cycle']}s"),
                            "full cycle",
                        ),
                        unsafe_allow_html=True,
                    )
                with sg2:
                    st.markdown(
                        metric_card(
                            "📗", "Green Efficiency",
                            mv("green", f"{live_eff}%"), "green ratio",
                        ),
                        unsafe_allow_html=True,
                    )
                with sg3:
                    idle_reduction = round(
                        ((120 - live_plan["Total Cycle"]) / 120) * 100, 1
                    )
                    st.markdown(
                        metric_card(
                            "⬇️", "Idle Reduction",
                            mv("green", f"{max(0, idle_reduction)}%"),
                            "vs unoptimized",
                        ),
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    alert(
                        "info",
                        f"💡 Signal optimized for live conditions: "
                        f"<b>{live_plan['Green']}s green"
                        f" · {live_plan['Red']}s red</b>. "
                        f"Green efficiency = "
                        f"<b>{live_eff}%</b> of cycle time.",
                    ),
                    unsafe_allow_html=True,
                )

            with wt2:
                st.markdown(
                    section("📈 24-Hour Traffic Forecast"),
                    unsafe_allow_html=True
                )

                fc1, fc2, fc3, fc4 = st.columns(4)
                with fc1:
                    st.markdown(
                        metric_card(
                            "⬇️", "Minimum Volume",
                            mv("green", f"{min_v_live:,}"),
                            "vehicles/hr"
                        ),
                        unsafe_allow_html=True,
                    )
                with fc2:
                    st.markdown(
                        metric_card(
                            "📊", "Daily Average",
                            mv("blue", f"{avg_live:,}"),
                            "vehicles/hr"
                        ),
                        unsafe_allow_html=True,
                    )
                with fc3:
                    st.markdown(
                        metric_card(
                            "⬆️", "Peak Volume",
                            mv("red", f"{peak_v_live:,}"),
                            f"at {peak_h_live}:00"
                        ),
                        unsafe_allow_html=True,
                    )
                with fc4:
                    st.markdown(
                        metric_card(
                            "⏱️", "Best Hour",
                            mv("green", f"{best_h_live}:00"),
                            "lowest traffic"
                        ),
                        unsafe_allow_html=True,
                    )

                live_fdf = pd.DataFrame({
                    "Hour": list(range(24)),
                    "Traffic Volume": h_preds_live.astype(int),
                    "Traffic Level": h_levels,
                })
                st.line_chart(live_fdf.set_index("Hour")["Traffic Volume"])

                st.markdown(
                    section("📋 Hourly Traffic Table"),
                    unsafe_allow_html=True
                )
                st.dataframe(
                    live_fdf.rename(
                        columns={"Traffic Volume": "Volume (veh/hr)"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                # Peak vs off-peak breakdown
                peak_hours_list = [
                    h for h, v in enumerate(h_preds_live) if v > 5000
                ]
                off_peak_hours = [
                    h for h, v in enumerate(h_preds_live) if v < 3000
                ]
                peak_str = (
                    ", ".join(f"{h}:00" for h in peak_hours_list)
                    if peak_hours_list else "None today — great!"
                )
                off_str = (
                    ", ".join(f"{h}:00" for h in off_peak_hours[:8])
                    + ("..." if len(off_peak_hours) > 8 else "")
                )
                st.markdown(
                    alert(
                        "warning" if peak_hours_list else "success",
                        f"🔴 <b>Peak congestion hours:</b> {peak_str}"
                        f"<br>🟢 <b>Off-peak (low traffic):</b> {off_str}",
                    ),
                    unsafe_allow_html=True,
                )

            with wt3:
                st.markdown(
                    section("🕐 Best Travel Windows"),
                    unsafe_allow_html=True
                )
                st.markdown(
                    alert(
                        "info",
                        "Travel windows are calculated from "
                        "the 24H forecast using live weather. "
                        "Green windows = safe to travel. "
                        "Yellow = moderate. Red = avoid.",
                    ),
                    unsafe_allow_html=True,
                )

                # Best single hour
                st.markdown(
                    f'<div class="opt-solution-card">'
                    f'<div class="opt-title">🏆 Optimal Travel Plan</div>'
                    '<div style="display:flex;gap:14px;'
                    'flex-wrap:wrap;margin-bottom:20px;">'
                    f'<div class="stat-box" style="flex:1">'
                    f'<div class="stat-val">{best_h_live}:00</div>'
                    f'<div class="stat-label">🟢 Best Single Hour</div></div>'
                    f'<div class="stat-box" style="flex:1">'
                    f'<div class="stat-val" '
                    f'style="color:#FF4D6D">{peak_h_live}:00</div>'
                    f'<div class="stat-label">🔴 Peak Congestion</div></div>'
                    f'<div class="stat-box" style="flex:1">'
                    f'<div class="stat-val" '
                    f'style="color:#FFD700">{avg_live:,}</div>'
                    f'<div class="stat-label">📊 Daily Avg Volume</div></div>'
                    f'<div class="stat-box" style="flex:1">'
                    f'<div class="stat-val">{len(good_windows)}</div>'
                    f'<div class="stat-label">🟢 Good Windows</div></div>'
                    f"</div>"
                    '<div style="background:rgba(0,255,150,0.08);'
                    'border:1px solid rgba(0,255,150,0.25);'
                    'border-radius:12px;padding:16px 20px;'
                    f'margin-bottom:12px;">'
                    '<div style="font-family:\'Orbitron\',monospace;'
                    'color:#00FF96;font-size:0.78rem;'
                    'letter-spacing:2px;margin-bottom:8px;">'
                    f"✅ RECOMMENDED DEPARTURE</div>"
                    f'<div style="font-family:\'Exo 2\',sans-serif;'
                    f"color:rgba(200,255,230,0.9);font-size:0.92rem;"
                    f'">Leave at <b>{best_h_live}:00</b>'
                    ' — lowest predicted traffic '
                    f"({min_v_live:,} veh/hr) based on current"
                    " live weather conditions at "
                    f"{weather_loc}. Avoid departing between "
                    f"{avoid_live}.</div></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Good travel windows list
                if good_windows:
                    st.markdown(
                        section("🟢 Extended Travel Windows"),
                        unsafe_allow_html=True
                    )
                    for ws, we in good_windows:
                        wlen = we - ws + 1
                        wmax = int(max(h_preds_live[ws:we+1]))
                        wmin = int(min(h_preds_live[ws:we+1]))
                        wc = "green" if wmax < 3000 else "yellow"
                        st.markdown(
                            f'<div class="route-card route-card-{wc}" '
                            f'style="margin-bottom:8px;">'
                            f'<div>'
                            f'<div class="route-name">'
                            f"{'🟢' if wc == 'green' else '🟡'} "
                            f"{ws}:00 — {we}:00</div>"
                            f'<div style="font-family:\'Exo 2\',sans-serif;'
                            f"color:rgba(160,220,255,0.5);font-size:0.72rem;"
                            f'margin-top:3px;">{wlen} hour window · '
                            f"Traffic {wmin:,}–{wmax:,} veh/hr</div>"
                            f"</div>"
                            f'<div class="route-status-{wc}">'
                            f"{'CLEAR' if wc == 'green' else 'MODERATE'}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        alert(
                            "danger",
                            "🚨 No extended clear windows found today. "
                            f"Best single hour is <b>{best_h_live}:00</b>. "
                            "Plan a short trip or consider"
                            " travelling tomorrow.",
                        ),
                        unsafe_allow_html=True,
                    )

                # Hour-by-hour recommendation strip
                st.markdown(
                    section("🕐 Hour-by-Hour Status"),
                    unsafe_allow_html=True
                )
                hour_strip_html = (
                    '<div style="display:flex;flex-wrap:wrap;'
                    'gap:6px;margin:10px 0;">'
                )
                for hi, (hv, hl) in enumerate(zip(h_preds_live, h_levels)):
                    hc = (
                        "#00FF96" if "🟢" in hl
                        else ("#FFD700" if "🟡" in hl else "#FF4D6D")
                    )
                    hbg = "rgba(0,255,150,0.1)" if "🟢" in hl else (
                        "rgba(255,215,0,0.1)" if "🟡" in hl
                        else "rgba(255,77,109,0.1)"
                    )
                    border = (
                        hc.replace(")", ",0.3)").replace("rgb", "rgba")
                        if "rgb" in hc else hc
                    )
                    is_best = hi == best_h_live
                    is_peak = hi == peak_h_live
                    extra_style = (
                        "border-width:2px;transform:scale(1.1);"
                        if is_best or is_peak else ""
                    )
                    label = "BEST" if is_best else ("PEAK" if is_peak else "")
                    hour_strip_html += (
                        f'<div style="background:{hbg};border:1px solid {hc};'
                        f"border-radius:8px;padding:6px 8px;text-align:center;"
                        f"min-width:52px;{extra_style}"
                        f'">'
                        f'<div style="'
                        f'font-family:\'Orbitron\',monospace;color:{hc};'
                        f'font-size:0.65rem;font-weight:700;">{hi}:00</div>'
                        f'<div style="font-family:\'Exo 2\',sans-serif;'
                        f"color:rgba(160,220,255,0.6);font-size:0.6rem;"
                        f'">{int(hv/100)/10}k</div>'
                        + (
                            f'<div style="font-size:0.55rem;color:{hc};'
                            f'font-weight:700;">{label}</div>' if label else ""
                        )
                        + f"</div>"
                    )
                hour_strip_html += "</div>"
                st.markdown(hour_strip_html, unsafe_allow_html=True)

            with wt4:
                st.markdown(
                    section("🗺️ Route Recommendations"),
                    unsafe_allow_html=True
                )
                e, n, s = live_routes["main"]
                st.markdown(
                    make_route_card("Current Route", e, n, s),
                    unsafe_allow_html=True
                )
                e, n, s = live_routes["alt1"]
                st.markdown(
                    make_route_card("Alternate Route 1", e, n, s),
                    unsafe_allow_html=True
                )
                e, n, s = live_routes["alt2"]
                st.markdown(
                    make_route_card("Alternate Route 2", e, n, s),
                    unsafe_allow_html=True
                )

                ra = "success" if live_lc == "green" else (
                    "warning" if live_lc == "yellow" else "danger"
                )
                st.markdown(
                    alert(
                        ra,
                        f"🧭 <b>Recommendation:</b> {live_routes['rec']}"
                        f"<br>⏱️ <b>Time Saved:</b> {live_routes['saved']}"
                        f"<br>🌧️ <b>Rain:</b> {rain}mm — "
                        + (
                            "Wet roads, reduce speed and"
                            " increase following distance."
                            if rain > 0
                            else "Dry conditions, normal driving."
                        ),
                    ),
                    unsafe_allow_html=True,
                )

                # Weather impact on route
                st.markdown(
                    section("🌦️ Weather Impact on Driving"),
                    unsafe_allow_html=True
                )
                wi1, wi2, wi3 = st.columns(3)
                visibility = (
                    "Poor" if rain > 10
                    else ("Reduced" if rain > 2 else "Clear")
                )
                vis_c = (
                    "red" if rain > 10
                    else ("yellow" if rain > 2 else "green")
                )
                road_cond = (
                    "Wet & Slippery" if rain > 5
                    else ("Damp" if rain > 0 else "Dry")
                )
                road_c = (
                    "red" if rain > 5
                    else ("yellow" if rain > 0 else "green")
                )
                with wi1:
                    st.markdown(
                        metric_card(
                            "👁️", "Visibility",
                            mv(vis_c, visibility),
                            "road conditions"
                        ),
                        unsafe_allow_html=True,
                    )
                with wi2:
                    st.markdown(
                        metric_card(
                            "🛣️", "Road Surface",
                            mv(road_c, road_cond),
                            "traction"
                        ),
                        unsafe_allow_html=True,
                    )
                with wi3:
                    speed_rec = (
                        "60 km/h" if rain > 10
                        else ("80 km/h" if rain > 2 else "100 km/h")
                    )
                    sp_c = (
                        "red" if rain > 10
                        else ("yellow" if rain > 2 else "green")
                    )
                    st.markdown(
                        metric_card(
                            "🏎️", "Advised Speed",
                            mv(sp_c, speed_rec),
                            "max recommended"
                        ),
                        unsafe_allow_html=True,
                    )

            with wt5:
                st.markdown(
                    section("🌍 Eco Impact — Live Conditions"),
                    unsafe_allow_html=True
                )
                ec1, ec2, ec3, ec4 = st.columns(4)
                with ec1:
                    st.markdown(
                        metric_card(
                            "💨", "CO2 Emissions",
                            mv("red", f"{live_co2}"), "kg/hr"
                        ),
                        unsafe_allow_html=True,
                    )
                with ec2:
                    st.markdown(
                        metric_card(
                            "⛽", "Fuel Wasted",
                            mv("yellow", f"{live_fuel}"), "L/hr"
                        ),
                        unsafe_allow_html=True,
                    )
                with ec3:
                    st.markdown(
                        metric_card(
                            "✅", "Carbon Saved",
                            mv("green", f"{live_saved}"), "kg CO2"
                        ),
                        unsafe_allow_html=True,
                    )
                with ec4:
                    st.markdown(
                        metric_card(
                            "🌳", "Tree Equivalent",
                            mv("green", f"{live_trees}"),
                            "trees/yr"
                        ),
                        unsafe_allow_html=True,
                    )

                # Daily eco projection
                daily_co2 = round(
                    sum(estimate_co2(int(v)) for v in h_preds_live), 1
                )
                daily_trees = round(daily_co2 / 21.77, 1)
                st.markdown(
                    section("📊 Full-Day Eco Projection"),
                    unsafe_allow_html=True
                )
                de1, de2 = st.columns(2)
                with de1:
                    st.markdown(
                        metric_card(
                            "💨", "Daily CO2 (all hours)",
                            mv("red", f"{daily_co2}"), "kg total today",
                        ),
                        unsafe_allow_html=True,
                    )
                with de2:
                    st.markdown(
                        metric_card(
                            "🌲", "Trees to Offset Daily",
                            mv("green", f"{daily_trees}"), "trees/yr needed",
                        ),
                        unsafe_allow_html=True,
                    )

                # CO2 by hour chart
                co2_by_hour = [
                    round(estimate_co2(int(v)), 1)
                    for v in h_preds_live
                ]
                co2_df = pd.DataFrame({
                    "Hour": list(range(24)),
                    "CO2 (kg/hr)": co2_by_hour,
                }).set_index("Hour")
                st.markdown(
                    section("📈 CO2 Emissions by Hour"),
                    unsafe_allow_html=True
                )
                st.area_chart(co2_df)

                st.markdown(
                    alert(
                        "success",
                        "🌱 <b>Eco tip:</b> Travelling at <b>"
                        f"{best_h_live}:00</b> (lowest traffic hour) reduces "
                        f"your CO2 footprint by up to "
                        f"<b>"
                        f"{round(live_co2 - estimate_co2(int(min_v_live)), 1)}"
                        f"kg/hr</b> "
                        "compared to peak hours.",
                    )
                    + alert(
                        "info",
                        "📊 Formulae: CO2 = 0.21 kg/km · "
                        "Idle fuel = 0.6 L/hr · Tree offset = 21.77 kg CO2/yr",
                    ),
                    unsafe_allow_html=True,
                )

            st.balloons()

# ================================================
# PAGE 4 — JOURNEY PLANNER
# ================================================
with pg_journey:

    st.markdown(
        """
    <div class="hero-banner" style="margin-top:8px">
        <div class="hero-title" style="font-size:1.6rem">
            🛣️ AI JOURNEY PLANNER
        </div>
        <div class="hero-sub">
            Select Route · AI Predicts Every Checkpoint
            · Optimized Solution with Departure Time
        </div>
        <div class="hero-badges">
            <span class="badge badge-green">📍 12 Locations</span>
            <span class="badge badge-blue">🤖 AI Checkpoint Analysis</span>
            <span class="badge badge-orange">🧠 Optimized Solution</span>
            <span class="badge badge-purple">🌍 Eco Report</span>
        </div>
    </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(section("📍 Plan Your Journey"), unsafe_allow_html=True)

    rc1, rc2 = st.columns(2)
    with rc1:
        start_loc = st.selectbox("📍 Start Location", LOCATION_NAMES, index=0)
    with rc2:
        dest_opts = [loc for loc in LOCATION_NAMES if loc != start_loc]
        dest_loc = st.selectbox(
            "🏁 Destination", dest_opts, index=len(dest_opts) - 1,
        )

    s_km = LOCATIONS[start_loc]["km_from_start"]
    d_km = LOCATIONS[dest_loc]["km_from_start"]
    total_km = abs(d_km - s_km)
    preview_cps = build_checkpoints(start_loc, dest_loc)
    n_preview = len(preview_cps)

    # Route preview bar
    dot_mid = "".join(
        ['<div class="journey-dot-dim"></div>']
        * max(0, n_preview - 2)
    )
    mid_labels = " &nbsp;·&nbsp; ".join([
        f"<span>{cp['name']}</span>"
        for cp in preview_cps[1:-1]
    ]) if n_preview > 2 else ""

    st.markdown(
        f'<div class="journey-bar">'
        f'<div style="display:flex;justify-content:space-between;'
        f"align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:12px;"
        f'">'
        f'<div style="font-family:\'Orbitron\',monospace;color:#00FF96;'
        f'font-size:0.88rem;letter-spacing:1px;">'
        f"📍 {start_loc}"
        f'<span style="color:#00BFFF;margin:0 10px;">→</span>'
        f"🏁 {dest_loc}</div>"
        f'<div style="display:flex;gap:12px;flex-wrap:wrap;">'
        f'<span style="font-family:\'Exo 2\',sans-serif;font-size:0.75rem;'
        f"color:rgba(160,220,255,0.6);background:rgba(0,191,255,0.1);"
        f"border:1px solid rgba(0,191,255,0.3);border-radius:20px;"
        f'padding:3px 12px;">🛣️ {total_km} km</span>'
        f'<span style="font-family:\'Exo 2\',sans-serif;font-size:0.75rem;'
        f"color:rgba(160,220,255,0.6);background:rgba(0,255,150,0.1);"
        f"border:1px solid rgba(0,255,150,0.3);border-radius:20px;"
        f'padding:3px 12px;">📍 {n_preview} checkpoints</span>'
        f"</div></div>"
        f'<div style="display:flex;align-items:center;'
        f'justify-content:space-between;position:relative;">'
        f'<div class="journey-dot"></div>'
        f"{dot_mid}"
        f'<div class="journey-dot" style="background:#FF4D6D;'
        f'box-shadow:0 0 8px #FF4D6D;"></div>'
        f"</div>"
        f'<div style="display:flex;justify-content:space-between;'
        f"margin-top:7px;font-family:'Exo 2',sans-serif;"
        f'color:rgba(160,220,255,0.32);font-size:0.68rem;">'
        f"<span>📍 {start_loc}</span>"
        f"{mid_labels}"
        f"<span>🏁 {dest_loc}</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    run_btn = st.button(
        "🚀  ANALYZE JOURNEY & GET OPTIMIZED SOLUTION", key="journey_btn"
    )

    if run_btn:
        cps = build_checkpoints(start_loc, dest_loc)
        if len(cps) < 2:
            st.markdown(
                alert(
                    "warning",
                    "⚠️ Not enough checkpoints found. "
                    "Please select two different locations"
                    " with enough distance.",
                ),
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Analyzing all checkpoints..."):
                results = []
                for cp in cps:
                    inp = {
                        "temp": cp["temp"],
                        "rain_1h": cp["rain_1h"],
                        "snow_1h": cp["snow_1h"],
                        "clouds_all": cp["clouds_all"],
                        "hour": cp["hour"], "day": 15,
                        "month": cp["month"],
                        "day_of_week": cp["day_of_week"],
                    }
                    pred = predict_traffic(tuple(inp.items()))
                    lv, st_t, em, col = get_level(pred)
                    pl = get_signal_plan(st_t)
                    rt = get_route_suggestion(lv)
                    rv2, rl2, rc2 = get_accident_risk(
                        cp["rain_1h"], pred, cp["hour"], cp["clouds_all"],
                    )
                    results.append({
                        "cp": cp, "pred": pred,
                        "level": lv, "sig_time": st_t,
                        "emoji": em, "color": col,
                        "plan": pl, "routes": rt,
                        "co2": estimate_co2(pred),
                        "fuel": estimate_fuel_waste(st_t, pred),
                        "saved": carbon_saved(st_t, pred),
                        "risk_val": rv2,
                        "risk_label": rl2,
                        "risk_color": rc2,
                    })

                score, slabel, scolor = get_journey_score(results)
                sol = get_optimized_solution(
                    results, start_loc, dest_loc, total_km
                )

            score_col = (
                "#00FF96" if scolor == "green"
                else ("#FFD700" if scolor == "yellow" else "#FF4D6D")
            )
            n_cp = len(cps)

            # Journey score banner
            st.markdown(
                f'<div class="summary-card" style="margin:14px 0">'
                f'<div class="summary-title">🏆 Journey Health Score</div>'
                '<div style="display:flex;align-items:center;'
                'gap:28px;flex-wrap:wrap;">'
                f'<div style="text-align:center">'
                '<div style="font-family:\'Orbitron\',monospace;'
                'font-size:3rem;'
                f'font-weight:900;color:{score_col}">{score}</div>'
                f'<div style="font-family:\'Exo 2\',sans-serif;'
                f"color:rgba(160,220,255,0.45);font-size:0.72rem;"
                'letter-spacing:2px;text-transform:uppercase;'
                '">out of 100</div>'
                f"</div><div>"
                '<div style="font-family:\'Orbitron\',monospace;'
                'font-size:1.1rem;'
                f'color:#fff;margin-bottom:5px;">{slabel}</div>'
                f'<div style="font-family:\'Exo 2\',sans-serif;'
                "color:rgba(160,220,255,0.58);font-size:0.83rem;"
                "max-width:380px;"
                f'">{start_loc} &rarr; {dest_loc} '
                f"&middot; {total_km} km &middot; {n_cp} checkpoints"
                f"</div></div></div></div>",
                unsafe_allow_html=True,
            )

            # Optimized solution card
            st.markdown(
                section("🧠 AI Optimized Journey Solution"),
                unsafe_allow_html=True
            )

            high_zones_html = (
                f'<div style="background:rgba(255,77,109,0.08);'
                f"border:1px solid rgba(255,77,109,0.25);"
                'border-radius:10px;padding:12px;'
                'text-align:center;flex:1;min-width:100px;">'
                f'<div style="font-family:\'Orbitron\',monospace;'
                'color:#FF4D6D;'
                f'font-size:1.4rem;font-weight:700;">{sol["high_count"]}</div>'
                f'<div style="font-family:\'Exo 2\',sans-serif;'
                f"color:rgba(255,200,210,0.7);font-size:0.68rem;"
                'letter-spacing:1px;text-transform:uppercase;'
                '">🔴 High Zones</div></div>'
            )
            med_zones_html = (
                f'<div style="background:rgba(255,215,0,0.08);'
                f"border:1px solid rgba(255,215,0,0.25);"
                'border-radius:10px;padding:12px;'
                'text-align:center;flex:1;min-width:100px;">'
                f'<div style="font-family:\'Orbitron\',monospace;'
                'color:#FFD700;'
                f'font-size:1.4rem;font-weight:700;">{sol["med_count"]}</div>'
                f'<div style="font-family:\'Exo 2\',sans-serif;'
                f"color:rgba(255,240,180,0.7);font-size:0.68rem;"
                'letter-spacing:1px;text-transform:uppercase;'
                '">🟡 Medium Zones</div></div>'
            )
            clear_count = n_cp - sol["high_count"] - sol["med_count"]
            clear_zones_html = (
                f'<div style="background:rgba(0,255,150,0.08);'
                f"border:1px solid rgba(0,255,150,0.25);"
                'border-radius:10px;padding:12px;'
                'text-align:center;flex:1;min-width:100px;">'
                f'<div style="font-family:\'Orbitron\',monospace;'
                'color:#00FF96;'
                f'font-size:1.4rem;font-weight:700;">{clear_count}</div>'
                f'<div style="font-family:\'Exo 2\',sans-serif;'
                f"color:rgba(200,255,230,0.7);font-size:0.68rem;"
                'letter-spacing:1px;text-transform:uppercase;'
                '">🟢 Clear Zones</div></div>'
            )
            pro_tip_html = (
                f'<div style="background:rgba(0,191,255,0.06);'
                f"border:1px solid rgba(0,191,255,0.2);"
                f'border-radius:10px;padding:12px;flex:2;min-width:180px;">'
                f'<div style="font-family:\'Exo 2\',sans-serif;'
                "color:rgba(180,230,255,0.8);"
                "font-size:0.82rem;line-height:1.6;"
                '">💡 <b>Pro tip:</b> If you must travel'
                ' through high-traffic zones, '
                "use Old Mysore Road as alternate and avoid"
                " peak hours 7–9AM and 5–7PM."
                f"</div></div>"
            )

            st.markdown(
                f'<div class="opt-solution-card">'
                f'<div class="opt-title">'
                f'✅ Optimized Plan: {start_loc} &rarr; {dest_loc}</div>'
                '<div style="display:flex;gap:14px;'
                'flex-wrap:wrap;margin-bottom:22px;">'
                f'<div class="stat-box" style="flex:1">'
                f'<div class="stat-val">{sol["best_departure"]}:00</div>'
                f'<div class="stat-label">🕐 Best Departure</div></div>'
                f'<div class="stat-box" style="flex:1">'
                f'<div class="stat-val" '
                f'style="color:{score_col}">{score}/100</div>'
                f'<div class="stat-label">🏆 Journey Score</div></div>'
                f'<div class="stat-box" style="flex:1">'
                f'<div class="stat-val" '
                f'style="color:#FF4D6D">{sol["high_count"]}</div>'
                f'<div class="stat-label">🔴 High Traffic Zones</div></div>'
                f'<div class="stat-box" style="flex:1">'
                f'<div class="stat-val" '
                f'style="color:#FFD700">{sol["total_co2"]}kg</div>'
                f'<div class="stat-label">💨 Total CO2</div></div>'
                f'<div class="stat-box" style="flex:1">'
                f'<div class="stat-val">{sol["trees"]}</div>'
                f'<div class="stat-label">🌳 Trees Saved/yr</div></div>'
                f'<div class="stat-box" style="flex:1">'
                f'<div class="stat-val">{total_km}km</div>'
                f'<div class="stat-label">🛣️ Total Distance</div></div>'
                f"</div>"
                f'<div style="background:rgba(0,255,150,0.08);'
                f"border:1px solid rgba(0,255,150,0.25);"
                f"border-radius:12px;padding:16px 20px;margin-bottom:14px;"
                f'">'
                '<div style="font-family:\'Orbitron\',monospace;'
                'color:#00FF96;'
                f'font-size:0.78rem;letter-spacing:2px;margin-bottom:8px;">'
                f"🕐 RECOMMENDED DEPARTURE TIME</div>"
                f'<div style="font-family:\'Exo 2\',sans-serif;'
                f"color:rgba(200,255,230,0.9);font-size:0.92rem;"
                f'">Leave at <b>{sol["best_departure"]}:00</b>'
                ' to minimize traffic '
                f"exposure across all {n_cp} checkpoints on your route. "
                f"Avoid {sol['high_count']} high-traffic zones"
                " by departing at this time."
                f"</div></div>",
                unsafe_allow_html=True,
            )

            for tip in sol["tips"]:
                tip_type = (
                    "danger" if "🚨" in tip
                    else (
                        "warning" if ("⚠️" in tip or "🌧️" in tip)
                        else "success"
                    )
                )
                st.markdown(alert(tip_type, tip), unsafe_allow_html=True)

            st.markdown(
                f'<div style="display:flex;gap:10px;flex-wrap:wrap;'
                f"margin-top:16px;border-top:1px solid rgba(0,191,255,0.15);"
                f'padding-top:16px;">'
                f"{high_zones_html}{med_zones_html}"
                f"{clear_zones_html}{pro_tip_html}"
                f"</div></div>",
                unsafe_allow_html=True,
            )

            jt1, jt2, jt3 = st.tabs([
                "🛣️ Checkpoint Details",
                "📈 Journey Charts",
                "🌍 Eco Summary",
            ])

            with jt1:
                st.markdown(section("📍 Checkpoints"), unsafe_allow_html=True)
                for r in results:
                    cp = r["cp"]
                    col = r["color"]
                    vc = color_val_class(col)
                    rk = color_val_class(r["risk_color"])
                    e1, n1, s1 = r["routes"]["main"]
                    e2, n2, s2 = r["routes"]["alt1"]
                    e3, n3, s3 = r["routes"]["alt2"]
                    st.markdown(
                        f'<div class="checkpoint-card checkpoint-card-{col}">'
                        f'<div class="cp-header"><div>'
                        f'<div class="cp-name">📍 {cp["name"]}</div>'
                        f'<div class="cp-meta">'
                        f'KM {cp["km_from_start"]} &middot; '
                        f'{cp["hour"]}:00 &middot; 🌧️ {cp["rain_1h"]}mm</div>'
                        f'<div class="cp-scenario">💬 {cp["scenario"]}</div>'
                        f"</div>"
                        f'<div class="cp-badge-{col}">'
                        f'{r["emoji"]} {r["level"]}</div>'
                        f"</div>"
                        f'<div class="cp-metrics">'
                        f'<div class="cp-metric">'
                        f'<div class="cp-metric-label">Volume</div>'
                        f'<div class="{vc}">{r["pred"]:,}</div></div>'
                        f'<div class="cp-metric">'
                        f'<div class="cp-metric-label">🟢 Signal</div>'
                        f'<div class="cp-val-green">'
                        f'{r["sig_time"]}s</div></div>'
                        f'<div class="cp-metric">'
                        f'<div class="cp-metric-label">🔴 Signal</div>'
                        f'<div class="cp-val-red">'
                        f'{r["plan"]["Red"]}s</div></div>'
                        f'<div class="cp-metric">'
                        f'<div class="cp-metric-label">Risk</div>'
                        f'<div class="{rk}">{r["risk_val"]}%</div></div>'
                        f'<div class="cp-metric">'
                        f'<div class="cp-metric-label">CO2</div>'
                        f'<div class="cp-val-yellow">{r["co2"]}kg</div></div>'
                        f'<div class="cp-metric">'
                        f'<div class="cp-metric-label">Saved</div>'
                        f'<div class="cp-val-blue">{r["saved"]}kg</div></div>'
                        f"</div>"
                        f'<div class="cp-route">'
                        f"🛣️ {e1} {n1}&mdash;{s1} | "
                        f"{e2} {n2}&mdash;{s2} | "
                        f"{e3} {n3}&mdash;{s3}<br>"
                        f"🧭 {r['routes']['rec']} ⏱️ {r['routes']['saved']}"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )

            with jt2:
                st.markdown(
                    section("📈 Traffic Volume"),
                    unsafe_allow_html=True
                )
                cdf = pd.DataFrame({
                    "Checkpoint": [r["cp"]["name"] for r in results],
                    "Traffic Volume": [r["pred"] for r in results],
                }).set_index("Checkpoint")
                st.line_chart(cdf)

                st.markdown(
                    section("⚠️ Accident Risk"),
                    unsafe_allow_html=True
                )
                rdf = pd.DataFrame({
                    "Checkpoint": [r["cp"]["name"] for r in results],
                    "Risk Score (%)": [r["risk_val"] for r in results],
                }).set_index("Checkpoint")
                st.line_chart(rdf)

                st.markdown(
                    section("🌧️ Weather & CO2"),
                    unsafe_allow_html=True
                )
                wdf = pd.DataFrame({
                    "Checkpoint": [r["cp"]["name"] for r in results],
                    "Rain (mm)": [r["cp"]["rain_1h"] for r in results],
                    "CO2 (kg)": [r["co2"] for r in results],
                }).set_index("Checkpoint")
                st.line_chart(wdf)

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
                st.dataframe(tbl, use_container_width=True, hide_index=True)

            with jt3:
                t_co2 = round(sum(r["co2"] for r in results), 2)
                t_fuel = round(sum(r["fuel"] for r in results), 4)
                t_saved = round(sum(r["saved"] for r in results), 4)
                t_trees = round(t_saved / 21.77, 2)
                av = int(np.mean([r["pred"] for r in results]))
                worst = max(results, key=lambda x: x["pred"])
                best = min(results, key=lambda x: x["pred"])
                h_ct = sum(1 for r in results if r["color"] == "red")
                m_ct = sum(1 for r in results if r["color"] == "yellow")
                l_ct = sum(1 for r in results if r["color"] == "green")
                mr = max(results, key=lambda x: x["risk_val"])
                sc_k = summary_val_class(scolor)

                st.markdown(
                    f'<div class="summary-card">'
                    f'<div class="summary-title">🌱 Environmental Impact</div>'
                    f'<div class="summary-grid">'
                    f'<div class="summary-item">'
                    f'<div class="summary-icon">💨</div>'
                    f'<div class="summary-label">CO2</div>'
                    f'<div class="summary-val-red">{t_co2}kg</div></div>'
                    f'<div class="summary-item">'
                    f'<div class="summary-icon">⛽</div>'
                    f'<div class="summary-label">Fuel</div>'
                    f'<div class="summary-val-red">{t_fuel}L</div></div>'
                    f'<div class="summary-item">'
                    f'<div class="summary-icon">✅</div>'
                    f'<div class="summary-label">Saved</div>'
                    f'<div class="summary-val-green">{t_saved}kg</div></div>'
                    f'<div class="summary-item">'
                    f'<div class="summary-icon">🌳</div>'
                    f'<div class="summary-label">Trees</div>'
                    f'<div class="summary-val-green">{t_trees}</div></div>'
                    f'<div class="summary-item">'
                    f'<div class="summary-icon">🚗</div>'
                    f'<div class="summary-label">Avg Vol</div>'
                    f'<div class="summary-val-blue">{av:,}</div></div>'
                    f'<div class="summary-item">'
                    f'<div class="summary-icon">🏆</div>'
                    f'<div class="summary-label">Score</div>'
                    f'<div class="{sc_k}">{score}/100</div></div>'
                    f"</div></div>"
                    f'<div class="summary-card" style="margin-top:12px">'
                    f'<div class="summary-title">📍 Highlights</div>'
                    f'<div class="summary-grid">'
                    f'<div class="summary-item">'
                    f'<div class="summary-icon">⚠️</div>'
                    f'<div class="summary-label">Worst</div>'
                    f'<div class="summary-val-red" style="font-size:0.72rem">'
                    f'{worst["cp"]["name"]}</div></div>'
                    f'<div class="summary-item">'
                    f'<div class="summary-icon">✅</div>'
                    f'<div class="summary-label">Best</div>'
                    '<div class="summary-val-green"'
                    ' style="font-size:0.72rem">'
                    f'{best["cp"]["name"]}</div></div>'
                    f'<div class="summary-item">'
                    f'<div class="summary-icon">🚨</div>'
                    f'<div class="summary-label">Risk Zone</div>'
                    f'<div class="summary-val-red" style="font-size:0.72rem">'
                    f'{mr["cp"]["name"]} ({mr["risk_val"]}%)</div></div>'
                    f'<div class="summary-item">'
                    f'<div class="summary-icon">🔴</div>'
                    f'<div class="summary-label">High</div>'
                    f'<div class="summary-val-red">{h_ct}/{n_cp}</div></div>'
                    f'<div class="summary-item">'
                    f'<div class="summary-icon">🟡</div>'
                    f'<div class="summary-label">Medium</div>'
                    f'<div class="summary-val-yellow">'
                    f'{m_ct}/{n_cp}</div></div>'
                    f'<div class="summary-item">'
                    f'<div class="summary-icon">🟢</div>'
                    f'<div class="summary-label">Low</div>'
                    f'<div class="summary-val-green">{l_ct}/{n_cp}</div></div>'
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

            st.balloons()

# ================================================
# PAGE 5 — MODEL INSIGHTS
# ================================================
with pg_model:

    st.markdown(
        """
    <div class="hero-banner" style="margin-top:8px">
        <div class="hero-title" style="font-size:1.6rem">
            🔬 MODEL INSIGHTS
        </div>
        <div class="hero-sub">
            Performance Metrics · Feature Importance
            · Tree Predictions · Confidence Analysis
        </div>
        <div class="hero-badges">
            <span class="badge badge-green">🌲 80 Trees</span>
            <span class="badge badge-blue">🎯 R² 0.94</span>
            <span class="badge badge-orange">📉 MAE ~280</span>
            <span class="badge badge-purple">🔬 Random Forest</span>
        </div>
    </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(section("🔬 Model Performance Metrics"), unsafe_allow_html=True)
    st.markdown(
        alert(
            "info",
            "These are the actual performance metrics of the "
            "Random Forest Regressor trained on the Metro "
            "Interstate Traffic Volume dataset.",
        ),
        unsafe_allow_html=True,
    )

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(
            metric_card("📉", "MAE", mv("green", "~280"), "Mean Abs Error"),
            unsafe_allow_html=True,
        )
    with mc2:
        st.markdown(
            metric_card("📊", "RMSE", mv("blue", "~420"), "Root MSE"),
            unsafe_allow_html=True,
        )
    with mc3:
        st.markdown(
            metric_card("🎯", "R² Score", mv("green", "~0.94"), "Accuracy"),
            unsafe_allow_html=True,
        )
    with mc4:
        st.markdown(
            metric_card("🌲", "Trees", mv("blue", "80"), "Estimators"),
            unsafe_allow_html=True,
        )

    st.markdown(
        alert(
            "success",
            "✅ R² of ~0.94 means the model explains 94% of "
            "variance in traffic volume. MAE of ~280 means "
            "predictions are off by about 280 vehicles/hr on "
            "average — excellent for a real-world dataset.",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(section("📊 Feature Importance"), unsafe_allow_html=True)
    st.markdown(
        alert(
            "info",
            "Shows which input features the model relies on "
            "most to make predictions. Higher = more important.",
        ),
        unsafe_allow_html=True,
    )

    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).head(12)
    st.bar_chart(feat_df.set_index("Feature")["Importance"])
    st.dataframe(
        feat_df.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(section("🌲 How Random Forest Works"), unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-icon">🌳</div>'
            '<div class="feature-title">Step 1 — Build Trees</div>'
            '<div class="feature-desc">'
            "80 decision trees are built, each trained on a "
            "random subset of the data and features."
            "</div></div>",
            unsafe_allow_html=True,
        )
    with fc2:
        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-icon">🗳️</div>'
            '<div class="feature-title">Step 2 — Vote</div>'
            '<div class="feature-desc">'
            "For each input, all 80 trees predict traffic volume "
            "independently. Their predictions are collected."
            "</div></div>",
            unsafe_allow_html=True,
        )
    with fc3:
        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-icon">📊</div>'
            '<div class="feature-title">Step 3 — Average</div>'
            '<div class="feature-desc">'
            "The final prediction is the average of all 80 tree "
            "outputs. High agreement = high confidence."
            "</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown(section("🎯 Live Confidence Check"), unsafe_allow_html=True)
    cf_c1, cf_c2 = st.columns(2)
    with cf_c1:
        cf_hour = st.slider("Hour", 0, 23, 8, key="cf_hour")
        cf_rain = st.slider("Rain (mm)", 0.0, 50.0, 0.0, key="cf_rain")
    with cf_c2:
        cf_temp = st.slider("Temp (K)", 250.0, 330.0, 290.0, key="cf_temp")
        cf_clouds = st.slider("Clouds (%)", 0, 100, 40, key="cf_clouds")

    cf_btn = st.button("🎯  CHECK MODEL CONFIDENCE", key="cf_btn")
    if cf_btn:
        cf_inp = {
            "temp": cf_temp, "rain_1h": cf_rain,
            "snow_1h": 0.0, "clouds_all": cf_clouds,
            "hour": cf_hour, "day": 15,
            "month": 6, "day_of_week": 1,
        }
        cf_idf = pd.DataFrame([cf_inp])
        for c in feature_columns:
            if c not in cf_idf.columns:
                cf_idf[c] = 0
        cf_idf = cf_idf[feature_columns]

        all_tree_preds = [t.predict(cf_idf)[0] for t in model.estimators_]
        mean_p = np.mean(all_tree_preds)
        std_p = np.std(all_tree_preds)
        conf_v = max(0, min(100, round(100 - (std_p / mean_p) * 100, 1)))
        cf_pred = int(mean_p)

        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.markdown(
                metric_card(
                    "🚗", "Prediction",
                    mv("green", f"{cf_pred:,}"), "veh/hr"
                ),
                unsafe_allow_html=True,
            )
        with cc2:
            st.markdown(
                metric_card(
                    "🎯", "Confidence",
                    mv("blue", f"{conf_v}%"), "model certainty"
                ),
                unsafe_allow_html=True,
            )
        with cc3:
            st.markdown(
                metric_card(
                    "📊", "Std Deviation",
                    mv("yellow", f"{round(std_p, 1)}"),
                    "tree variance"
                ),
                unsafe_allow_html=True,
            )

        tree_df = pd.DataFrame({
            "Tree": list(range(1, 81)),
            "Prediction": [int(p) for p in all_tree_preds],
        }).set_index("Tree")
        st.markdown(
            section("🌳 All 80 Tree Predictions"),
            unsafe_allow_html=True
        )
        st.line_chart(tree_df)

# ================================================
# PAGE 6 — WHAT-IF SIMULATOR
# ================================================
with pg_whatif:

    st.markdown(
        section("🧪 What-If Scenario Simulator"),
        unsafe_allow_html=True
    )
    st.markdown(
        alert(
            "info",
            "Change one variable at a time and compare how "
            "traffic volume and eco impact changes. "
            "All other values stay fixed.",
        ),
        unsafe_allow_html=True,
    )

    wc1, wc2 = st.columns(2)
    with wc1:
        st.markdown(
            '<div class="input-card">'
            '<div class="input-card-title">⚙️ Base Conditions</div></div>',
            unsafe_allow_html=True,
        )
        w_temp = st.slider(
            "🌡️ Temperature (K)", 250.0, 330.0, 290.0,
            key="w_temp"
        )
        w_clouds = st.slider("☁️ Clouds (%)", 0, 100, 40, key="w_clouds")
        w_dow = st.slider("📅 Day of Week", 0, 6, 1, key="w_dow")
        w_month = st.slider("🗓️ Month", 1, 12, 6, key="w_month")

    with wc2:
        st.markdown(
            '<div class="input-card">'
            '<div class="input-card-title">🔄 Variable to Change</div></div>',
            unsafe_allow_html=True,
        )
        vary_var = st.selectbox(
            "Which variable to vary?",
            ["Hour of Day", "Rain (mm)", "Temperature (K)"],
            key="vary_var",
        )

        if vary_var == "Hour of Day":
            val_a = st.slider("Scenario A — Hour", 0, 23, 8, key="va")
            val_b = st.slider("Scenario B — Hour", 0, 23, 17, key="vb")
        elif vary_var == "Rain (mm)":
            val_a = st.slider("Scenario A — Rain", 0.0, 50.0, 0.0, key="va")
            val_b = st.slider("Scenario B — Rain", 0.0, 50.0, 15.0, key="vb")
        else:
            val_a = st.slider(
                "Scenario A — Temp", 250.0, 330.0, 270.0,
                key="va"
            )
            val_b = st.slider(
                "Scenario B — Temp", 250.0, 330.0, 305.0,
                key="vb"
            )

    wi_btn = st.button("🧪  RUN WHAT-IF COMPARISON", key="whatif_btn")

    if wi_btn:
        def make_wi_input(vary, val, base):
            inp = {
                "temp": base["temp"],
                "rain_1h": base["rain"],
                "snow_1h": 0.0,
                "clouds_all": base["clouds"],
                "hour": base["hour"],
                "day": 15,
                "month": base["month"],
                "day_of_week": base["dow"],
            }
            if vary == "Hour of Day":
                inp["hour"] = int(val)
            elif vary == "Rain (mm)":
                inp["rain_1h"] = float(val)
            else:
                inp["temp"] = float(val)
            return inp

        base_d = {
            "temp": w_temp, "rain": 0.0,
            "clouds": w_clouds, "hour": 8,
            "month": w_month, "dow": w_dow,
        }
        inp_a = make_wi_input(vary_var, val_a, base_d)
        inp_b = make_wi_input(vary_var, val_b, base_d)
        pred_a = predict_traffic(tuple(inp_a.items()))
        pred_b = predict_traffic(tuple(inp_b.items()))
        lv_a, st_a, em_a, lc_a = get_level(pred_a)
        lv_b, st_b, em_b, lc_b = get_level(pred_b)
        co2_a = estimate_co2(pred_a)
        co2_b = estimate_co2(pred_b)
        sv_a = carbon_saved(st_a, pred_a)
        sv_b = carbon_saved(st_b, pred_b)
        rv_a, rl_a, rc_a = get_accident_risk(
            inp_a["rain_1h"], pred_a, inp_a["hour"], inp_a["clouds_all"],
        )
        rv_b, rl_b, rc_b = get_accident_risk(
            inp_b["rain_1h"], pred_b, inp_b["hour"], inp_b["clouds_all"],
        )
        diff = pred_b - pred_a
        diff_s = f"+{diff:,}" if diff >= 0 else f"{diff:,}"

        lc_b_val_class = (
            "summary-val-red" if lc_b == "red"
            else (
                "summary-val-yellow" if lc_b == "yellow"
                else "summary-val-green"
            )
        )

        st.markdown(section("📊 Scenario Comparison"), unsafe_allow_html=True)
        col_a, col_mid, col_b = st.columns([5, 1, 5])

        with col_a:
            st.markdown(
                f'<div class="summary-card">'
                f'<div class="summary-title">'
                f'Scenario A — {vary_var}: {val_a}</div>'
                f'<div class="summary-grid">'
                f'<div class="summary-item">'
                f'<div class="summary-icon">{em_a}</div>'
                f'<div class="summary-label">Volume</div>'
                f'<div class="summary-val-green">{pred_a:,}</div></div>'
                f'<div class="summary-item">'
                f'<div class="summary-icon">🚦</div>'
                f'<div class="summary-label">Level</div>'
                f'<div class="summary-val-green">{lv_a}</div></div>'
                f'<div class="summary-item">'
                f'<div class="summary-icon">💨</div>'
                f'<div class="summary-label">CO2</div>'
                f'<div class="summary-val-red">{co2_a}kg</div></div>'
                f'<div class="summary-item">'
                f'<div class="summary-icon">✅</div>'
                f'<div class="summary-label">Saved</div>'
                f'<div class="summary-val-green">{sv_a}kg</div></div>'
                f'<div class="summary-item">'
                f'<div class="summary-icon">⚠️</div>'
                f'<div class="summary-label">Risk</div>'
                f'<div class="summary-val-blue">{rv_a}%</div></div>'
                f"</div></div>",
                unsafe_allow_html=True,
            )

        with col_mid:
            st.markdown(
                '<div style="display:flex;align-items:center;'
                "justify-content:center;height:100%;"
                "font-family:'Orbitron',monospace;"
                "color:#00BFFF;font-size:1.5rem;"
                'padding-top:80px">VS</div>',
                unsafe_allow_html=True,
            )

        with col_b:
            st.markdown(
                f'<div class="summary-card">'
                f'<div class="summary-title">'
                f'Scenario B — {vary_var}: {val_b}</div>'
                f'<div class="summary-grid">'
                f'<div class="summary-item">'
                f'<div class="summary-icon">{em_b}</div>'
                f'<div class="summary-label">Volume</div>'
                f'<div class="{lc_b_val_class}">{pred_b:,}</div></div>'
                f'<div class="summary-item">'
                f'<div class="summary-icon">🚦</div>'
                f'<div class="summary-label">Level</div>'
                f'<div class="{lc_b_val_class}">{lv_b}</div></div>'
                f'<div class="summary-item">'
                f'<div class="summary-icon">💨</div>'
                f'<div class="summary-label">CO2</div>'
                f'<div class="summary-val-red">{co2_b}kg</div></div>'
                f'<div class="summary-item">'
                f'<div class="summary-icon">✅</div>'
                f'<div class="summary-label">Saved</div>'
                f'<div class="summary-val-green">{sv_b}kg</div></div>'
                f'<div class="summary-item">'
                f'<div class="summary-icon">⚠️</div>'
                f'<div class="summary-label">Risk</div>'
                f'<div class="summary-val-blue">{rv_b}%</div></div>'
                f"</div></div>",
                unsafe_allow_html=True,
            )

        d_al = "danger" if diff > 0 else "success"
        d_msg = (
            f"Scenario B has <b>{diff_s}</b> vehicles/hr"
            " more than Scenario A. "
            if diff > 0
            else (
                f"Scenario B has <b>{abs(diff):,}</b>"
                " fewer vehicles/hr than Scenario A. "
            )
        )
        st.markdown(
            alert(
                d_al,
                f"📊 {d_msg}Changing <b>{vary_var}</b> from "
                f"{val_a} to {val_b} causes this difference.",
            ),
            unsafe_allow_html=True,
        )

        st.markdown(section("📊 Visual Comparison"), unsafe_allow_html=True)
        comp_df = pd.DataFrame({
            "Metric": ["Traffic Volume", "CO2 (kg)", "Risk Score (%)"],
            f"Scenario A ({val_a})": [pred_a, co2_a, rv_a],
            f"Scenario B ({val_b})": [pred_b, co2_b, rv_b],
        }).set_index("Metric")
        st.bar_chart(comp_df)
