# ==============================
# Traffic Optimization - Model Training
# ==============================

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==============================
# 1. Load RAW Dataset
# ==============================

DATA_PATH = "data/Metro_Interstate_Traffic_Volume.csv"
df = pd.read_csv(DATA_PATH)

print("Raw dataset loaded")
print("Initial shape:", df.shape)


# ==============================
# 2. DATA CLEANING (MANDATORY)
# ==============================

# Drop unnecessary column
if "weather_description" in df.columns:
    df.drop(columns=["weather_description"], inplace=True)

# Convert datetime
df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")

# Handle HOLIDAY nulls (IMPORTANT)
df["holiday"].fillna("None", inplace=True)

# Drop rows with remaining nulls
df.dropna(inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

print("\nAfter cleaning:")
print("Null values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Cleaned shape:", df.shape)


# ==============================
# 3. FEATURE ENGINEERING
# ==============================

df["hour"] = df["date_time"].dt.hour
df["day"] = df["date_time"].dt.day
df["month"] = df["date_time"].dt.month
df["day_of_week"] = df["date_time"].dt.dayofweek

df.drop(columns=["date_time"], inplace=True)


# ==============================
# 4. ENCODING
# ==============================

df = pd.get_dummies(
    df,
    columns=["holiday", "weather_main"],
    drop_first=True
)


# ==============================
# 5. FINAL VALIDATION
# ==============================

assert df.isnull().sum().sum() == 0, "❌ Null values still exist"
assert df.duplicated().sum() == 0, "❌ Duplicates still exist"

print("\nFinal dataset is CLEAN ✅")


# ==============================
# 6. SPLIT FEATURES & TARGET
# ==============================

X = df.drop("traffic_volume", axis=1)
y = df["traffic_volume"]


# ==============================
# 7. TRAIN-TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# 8. TRAIN BEST MODEL
# ==============================

model = RandomForestRegressor(
    n_estimators=120,      # reduced from 300
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


print("\nTraining model...")
model.fit(X_train, y_train)
print("Model training completed")


# ==============================
# 9. EVALUATION
# ==============================

y_pred = model.predict(X_test)

print("\nModel Performance:")
print("MAE :", round(mean_absolute_error(y_test, y_pred), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
print("R2  :", round(r2_score(y_test, y_pred), 4))


# ==============================
# 10. SAVE MODEL
# ==============================

os.makedirs("model", exist_ok=True)

joblib.dump(model, open("traffic_model.pkl", "wb"))
joblib.dump(X.columns.tolist(), open("feature_columns.pkl", "wb"))

print("\nModel & features saved successfully ✅")
print("TRAINING PIPELINE COMPLETED 🚦")

