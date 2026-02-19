🚦 AI-Based Traffic Optimization System

An intelligent machine learning system that predicts traffic volume and optimizes traffic signal timing dynamically to reduce congestion.

This project uses Random Forest Regression trained on real-world traffic data and is deployed as an interactive web application using Streamlit Cloud.

📌 Project Overview

Traffic congestion is a major problem in urban areas.
This system:

Predicts traffic volume based on weather & time conditions

Classifies traffic as Low, Medium, or High

Dynamically recommends optimal green signal timing

Provides an interactive web interface

🧠 Machine Learning Model

Algorithm: Random Forest Regressor

Target Variable: traffic_volume

Features Used:

Temperature

Rain (1h)

Snow (1h)

Cloud Coverage

Hour

Day

Month

Day of Week

Encoded Holiday & Weather features

📊 Model Performance

MAE: ~ (your value)

RMSE: ~ (your value)

R² Score: ~ (your value)

(Replace above with your actual metrics from training output)

📂 Project Structure
traffic-optimization/
│
├── app.py                      # Streamlit web application
├── train_model.py              # Model training pipeline
├── requirements.txt            # Required dependencies
├── README.md
│
├── model/
│   ├── traffic_model.pkl       # Trained ML model
│   └── feature_columns.pkl     # Feature alignment file
│
└── data/
    └── Metro_Interstate_Traffic_Volume.csv

⚙️ How It Works

Data Cleaning & Preprocessing

Removed duplicates

Handled null values

Feature engineering (hour, month, day_of_week)

One-hot encoding

Model Training

Train-test split (80-20)

Random Forest training

Model evaluation

Deployment

Model saved using Pickle

Streamlit UI built for prediction

Deployed on Streamlit Cloud

🚀 How to Run Locally
Step 1: Install dependencies
pip install -r requirements.txt

Step 2: Train the model
python train_model.py

Step 3: Run Streamlit app
streamlit run app.py

🌐 Live Deployment

Deployed using Streamlit Cloud

The application predicts traffic intensity and recommends optimized signal timing in real time.

🎯 Traffic Signal Optimization Logic
Traffic Volume	Level	Green Signal Time
< 3000	Low Traffic	30 seconds
3000 – 6000	Medium Traffic	60 seconds
> 6000	High Traffic	90 seconds
💡 Technologies Used

Python

Pandas

NumPy

Scikit-learn

Streamlit

Pickle

📊 Dataset

Metro Interstate Traffic Volume Dataset
Contains weather and time-based traffic data.

👨‍💻 Author

Your Name
Final Year Project / Capstone Project
AI & Machine Learning