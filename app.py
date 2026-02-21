import streamlit as st
import joblib
import pandas as pd


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Traffic Optimization",
    page_icon="🚦",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load(open("traffic_model.pkl", "rb"))
feature_columns = joblib.load("feature_columns.pkl", "rb")

# -----------------------------
# CUSTOM CSS (FOR ATTRACTIVE UI)
# -----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    h1 {
        text-align: center;
        color: #00FFAA;
    }
    .stButton>button {
        background-color: #00FFAA;
        color: black;
        font-weight: bold;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.title("🚦 AI-Based Traffic Optimization System")
st.markdown("### Predict Traffic Volume using Machine Learning")

st.divider()

# -----------------------------
# INPUT SECTION
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌦 Weather Conditions")
    temp = st.slider("Temperature (Kelvin)", 250.0, 330.0, 280.0)
    rain = st.slider("Rain (last 1 hour mm)", 0.0, 50.0, 0.0)
    snow = st.slider("Snow (last 1 hour mm)", 0.0, 50.0, 0.0)
    clouds = st.slider("Cloud Coverage (%)", 0, 100, 50)

with col2:
    st.subheader("📅 Date & Time Information")
    hour = st.slider("Hour of Day", 0, 23, 8)
    day = st.slider("Day of Month", 1, 31, 15)
    month = st.slider("Month", 1, 12, 6)
    day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)

st.divider()

# -----------------------------
# CREATE INPUT DATA
# -----------------------------
input_data = {
    "temp": temp,
    "rain_1h": rain,
    "snow_1h": snow,
    "clouds_all": clouds,
    "hour": hour,
    "day": day,
    "month": month,
    "day_of_week": day_of_week
}

input_df = pd.DataFrame([input_data])

# Align with training columns
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_columns]

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🚀 Predict Traffic & Optimize Signal"):

    prediction = model.predict(input_df)[0]
    prediction = int(prediction)

    st.divider()
    st.subheader("📊 Prediction Results")

    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("Predicted Traffic Volume", prediction)

    if prediction < 3000:
        level = "Low Traffic"
        signal_time = 30
        color = "🟢"
    elif prediction < 6000:
        level = "Medium Traffic"
        signal_time = 60
        color = "🟡"
    else:
        level = "High Traffic"
        signal_time = 90
        color = "🔴"

    with colB:
        st.metric("Traffic Level", level)

    with colC:
        st.metric("Recommended Green Signal (seconds)", signal_time)

    st.divider()

    # Progress bar visualization
    st.subheader("🚦 Traffic Intensity Indicator")

    traffic_percent = min(prediction / 8000, 1.0)
    st.progress(traffic_percent)

    st.success(f"{color} Signal Timing Optimized Successfully!")

    st.balloons()


