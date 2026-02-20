🚦 AI-Based Traffic Optimization System
Intelligent Traffic Volume Prediction & Signal Timing Optimization using Machine Learning
🌍 Problem Statement

Urban traffic congestion leads to:

Increased travel time

Fuel wastage

Environmental pollution

Road accidents

Traditional traffic signals operate on fixed timers, regardless of real-time traffic conditions.

👉 This project builds an AI-powered traffic prediction system that dynamically recommends optimal signal timing based on weather and temporal conditions.

🧠 Solution Overview

This system:

✔ Predicts traffic volume using Machine Learning
✔ Classifies traffic as Low / Medium / High
✔ Dynamically recommends optimized green signal timing
✔ Provides an interactive web-based interface
✔ Is fully deployed on Streamlit Cloud

🏗️ System Architecture
Raw Dataset
     ↓
Data Cleaning & Preprocessing
     ↓
Feature Engineering
     ↓
Random Forest Model Training
     ↓
Model Serialization (.pkl)
     ↓
Streamlit Web Application
     ↓
Traffic Prediction + Signal Optimization
📊 Dataset

Metro Interstate Traffic Volume Dataset

Features include:

Temperature

Rain (1 hour)

Snow (1 hour)

Cloud coverage

Holiday

Weather condition

Date & time information

Target Variable:

traffic_volume
🧹 Data Preprocessing

Removed null values

Handled missing holiday entries

Dropped duplicates

Converted datetime into:

Hour

Day

Month

Day of week

Applied One-Hot Encoding for categorical features

Verified dataset integrity

🤖 Machine Learning Model

Algorithm Used:

Random Forest Regressor

Why Random Forest?

Handles non-linearity well

Robust to noise

High prediction accuracy

Works well on structured tabular data

📈 Model Performance
Metric	Value
MAE	---238.05
RMSE ---407.34
R² Score ---0.9578


🚦 Traffic Optimization Logic

Based on predicted traffic volume:

Traffic Volume	Traffic Level	Recommended Green Signal
< 3000	Low	30 seconds
3000 – 6000	Medium	60 seconds
> 6000	High	90 seconds

This enables adaptive traffic control.

💻 Web Application (Streamlit)

Features:

Professional dark UI

Real-time prediction

Traffic intensity indicator

Interactive sliders

Dynamic signal optimization display

Fully cloud deployed

📂 Project Structure
traffic-optimization-capstone/
│
├── app.py                     # Streamlit Application
├── train_model.py             # Model Training Pipeline
├── requirements.txt
├── README.md
│
├── model/
│   ├── traffic_model.pkl      # Optimized trained model (8.5MB)
│   └── feature_columns.pkl
│
└── data/
    └── Metro_Interstate_Traffic_Volume.csv
⚙️ How To Run Locally
1️⃣ Install Dependencies
pip install -r requirements.txt
2️⃣ Train Model
python train_model.py
3️⃣ Run Web App
streamlit run app.py
🌐 Deployment

The application is deployed using:

Streamlit Cloud

Users can interact with the model through a web interface without installing any software.

🔥 Key Highlights

✔ End-to-end ML pipeline
✔ Real-world dataset
✔ Model optimization for deployment
✔ Reduced model size from 123MB → 8.5MB
✔ Cloud deployment
✔ Clean professional UI
✔ Production-ready structure

📌 Future Enhancements

Integration with real-time traffic APIs

IoT-based smart signal automation

Live dashboard analytics

Deep Learning model experimentation

Smart city integration

🎓 Academic Relevance

This project demonstrates:

Data preprocessing & cleaning

Feature engineering

Model training & evaluation

Model serialization

Web application deployment

Real-world problem solving using AI

👨‍💻 Author

R Soumya
Final Year Capstone Project
Artificial Intelligence & Machine Learning

🚀 One-Line Resume Description

AI-Based Traffic Optimization System using Random Forest Regression to predict traffic volume and dynamically optimize signal timing, deployed using Streamlit Cloud.

💡 What Makes This Project Impressive?

Practical real-world application

Intelligent signal timing logic

Cloud deployment

Clean ML pipeline

Model optimization engineering


End-to-end production workflow
