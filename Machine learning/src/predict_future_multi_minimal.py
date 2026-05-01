#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import joblib
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
from pathlib import Path

# =================== CONFIG ===================
CSV_PATH = r"C:\Users\tan kiet\Desktop\IOT_Project\data\data_sensor_sorted.csv"
MODEL_DIR = r"C:\Users\tan kiet\Desktop\IOT_Project\model_rf_hourly"

SERVICE_ACCOUNT_PATH = r"C:\Users\tan kiet\Desktop\IOT_Project\fire-alarm-system-4592c-firebase-adminsdk-fbsvc-80e06d4e42.json"
DATABASE_URL = "https://fire-alarm-system-4592c-default-rtdb.firebaseio.com/"
SENSOR_NODE = "ESP32/Data"

# Chỉ lấy 6 sensor cần thiết
SENSORS = [
    "temperature", "humidity",
    "mq2_Smoke", "mq5_CH4",
    "mq5_LPG", "mq7_CO"
]

# Các mốc dự báo (giờ tới)
FORECAST_LIST = [1, 5, 12, 24]

# Các lag & rolling window giống lúc training
LAGS = [1, 3, 6, 12, 24, 48]
ROLLS = [3, 6, 12, 24]
# ===============================================


# =================== FIREBASE ===================
def init_firebase():
    """Khởi tạo kết nối Firebase nếu chưa khởi tạo."""
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        firebase_admin.initialize_app(cred, {"databaseURL": DATABASE_URL})


def read_realtime_sensors():

    snap = db.reference(SENSOR_NODE).get() or {}
    realtime = {}

    # Đọc 6 sensor
    for s in SENSORS:
        try:
            realtime[s] = float(snap.get(s, np.nan))
        except Exception:
            realtime[s] = np.nan

    # Lấy date & time từ Firebase
    fb_date = snap.get("date", None)   # vd: "15/11/2025"
    fb_time = snap.get("time", None)   # vd: "12:53:23"

    # Chuẩn hóa time về dạng HH:MM:SS
    fb_time_fixed = fix_time_string(fb_time) if fb_time is not None else None

    if fb_date and fb_time_fixed:
        try:
            ts = pd.to_datetime(
                fb_date + " " + fb_time_fixed,
                dayfirst=True,
                errors="coerce"
            )
            if pd.isna(ts):
                ts = datetime.now()
        except Exception:
            ts = datetime.now()
    else:
        # Nếu thiếu date/time thì fallback giờ hiện tại của máy
        ts = datetime.now()

    realtime["timestamp"] = ts

    return realtime
# =================================================


# ================== HELPERS ======================
def fix_time_string(t):
    """
    Chuẩn hóa chuỗi time về dạng HH:MM:SS
    - Nếu sai format -> trả về "00:00:00"
    """
    if t is None or pd.isna(t):
        return "00:00:00"

    parts = str(t).split(":")
    if len(parts) != 3:
        return "00:00:00"
    try:
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}:{int(parts[2]):02d}"
    except Exception:
        return "00:00:00"


def load_history_hourly():
  
    df = pd.read_csv(CSV_PATH)

    # Bỏ các cột Unnamed nếu có
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Chuẩn hóa cột time
    df["time"] = df["time"].astype(str).apply(fix_time_string)

    # Ghép timestamp
    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        dayfirst=True,
        errors="coerce"
    )

    # Bỏ các dòng lỗi timestamp & sort theo thời gian
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Lấy các cột numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Chỉ giữ timestamp + numeric
    df_numeric = df[["timestamp"] + numeric_cols].copy()

    # ==== CLIP OUTLIERS giống training ====
    for col in numeric_cols:
        q1 = df_numeric[col].quantile(0.01)
        q99 = df_numeric[col].quantile(0.99)
        df_numeric[col] = df_numeric[col].clip(q1, q99)

    # ==== HYBRID RESAMPLE (1h) ====
    agg_dict = {}
    for col in numeric_cols:
        if col in ["temperature", "humidity"]:
            agg_dict[col] = "mean"
        elif col in ["mq2_Smoke", "mq5_CH4", "mq5_LPG", "mq7_CO"]:
            agg_dict[col] = "median"
        else:
            agg_dict[col] = "mean"

    df_hour = df_numeric.resample("1h", on="timestamp").agg(agg_dict).reset_index()

    return df_hour


def build_realtime_features(df_hour, sensor, current_value, realtime_timestamp):
    
    df = df_hour.copy()

    # Thêm dòng hiện tại (copy dòng cuối rồi thay sensor + timestamp)
    new = df.iloc[-1:].copy()
    new[sensor] = current_value
    new["timestamp"] = realtime_timestamp
    df = pd.concat([df, new], ignore_index=True)

    # ==== LAG ====
    for l in LAGS:
        df[f"{sensor}_lag{l}"] = df[sensor].shift(l)

    # ==== ROLLING ====
    for r in ROLLS:
        df[f"{sensor}_roll_mean_{r}"] = df[sensor].shift(1).rolling(r).mean()
        df[f"{sensor}_roll_std_{r}"] = df[sensor].shift(1).rolling(r).std()

    # ==== DIFF ====
    df[f"{sensor}_diff1"] = df[sensor].diff(1)
    df[f"{sensor}_diff3"] = df[sensor].diff(3)
    df[f"{sensor}_diff6"] = df[sensor].diff(6)

    # ==== TIME FEATURES (sử dụng timestamp của dòng mới) ====
    last = df.iloc[-1:].copy()
    last["hour"] = last["timestamp"].dt.hour
    last["dow"] = last["timestamp"].dt.dayofweek
    last["hour_sin"] = np.sin(2 * np.pi * last["hour"] / 24.0)
    last["hour_cos"] = np.cos(2 * np.pi * last["hour"] / 24.0)

    return last
# =======================================================

# ================== FIREBASE WRITE ==================
def write_prediction_to_firebase(results, timestamp):
    """
    Ghi kết quả dự đoán ML lên Firebase
    results: dict ALL_RESULTS
    timestamp: datetime
    """
    ref = db.reference("ESP32/Prediction")

    payload = {
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
    }

    for sensor, preds in results.items():
        payload[sensor] = {}

        for key, value in preds.items():
            # Nếu là trend → giữ nguyên string
            if key == "trend":
                payload[sensor]["trend"] = value
            else:
                # Các mốc 1h, 5h, 12h, 24h là số
                payload[sensor][key] = round(float(value), 3)

    ref.set(payload)

    print(" ĐÃ GỬI KẾT QUẢ DỰ ĐOÁN LÊN FIREBASE")
# ====================================================



# ================== TREND DETECTION ==================
def detect_trend(current_value, future_preds, threshold=0.02):
    """
    Xác định xu hướng tăng / giảm / ổn định
    threshold: % thay đổi tối thiểu (2% mặc định)
    """
    try:
        future_24h = future_preds.get("24h", None)
        if future_24h is None:
            return "unknown"

        delta = (future_24h - current_value) / max(abs(current_value), 1e-6)

        if delta > threshold:
            return "increasing"
        elif delta < -threshold:
            return "decreasing"
        else:
            return "stable"
    except Exception:
        return "unknown"
# =====================================================


# =================== MAIN PREDICT =======================
def predict_future_multi():
    """
    - Load lịch sử từ CSV -> resample theo giờ
    - Đọc dữ liệu realtime + timestamp từ Firebase
    - Với mỗi sensor:
        + Tạo features realtime
        + Load model tương ứng các horizon (1h, 5h, 12h, 24h)
        + Dự đoán và in kết quả
    - Trả về dict ALL_RESULTS chứa dự báo cho tất cả sensor
    """

    print("\n=======================================")
    print("=== REALTIME MULTI-HORIZON FORECAST ===")
    print("=======================================\n")

    # Kết nối Firebase
    init_firebase()

    # Load dữ liệu lịch sử theo giờ
    history_hourly = load_history_hourly()

    # Đọc realtime từ Firebase
    realtime = read_realtime_sensors()
    print(" Firebase realtime (kèm timestamp):")
    print(realtime)

    fb_ts = realtime.get("timestamp", datetime.now())
    print(f"\n Timestamp từ Firebase (hoặc fallback): {fb_ts}\n")

    ALL_RESULTS = {}

    # Duyệt từng sensor
    for sensor in SENSORS:
        current_val = realtime.get(sensor, np.nan)

        if np.isnan(current_val):
            print(f"⚠ Sensor {sensor} không có dữ liệu! Bỏ qua.")
            continue

        print(f"\n---  Sensor: {sensor} ---")
        print(f"   Giá trị hiện tại: {current_val:.3f}")

        # Tạo feature realtime sử dụng timestamp của Firebase
        feats = build_realtime_features(history_hourly, sensor, current_val, fb_ts)
        sensor_preds = {}

        for h in FORECAST_LIST:
            model_path = f"{MODEL_DIR}/{sensor}_rf_hourly_{h}h.pkl"

            try:
                bundle = joblib.load(model_path)
            except Exception:
                print(f" Không tìm thấy model: {model_path}")
                continue

            model = bundle["model"]
            feature_cols = bundle["features"]

            # Lấy đúng các cột feature từng được dùng khi train
            X_input = feats[feature_cols]

            pred = float(model.predict(X_input)[0])
            sensor_preds[f"{h}h"] = pred

            print(f" {sensor} dự đoán {h}h tới: {pred:.3f}")

        #  XÁC ĐỊNH XU HƯỚNG
        trend = detect_trend(current_val, sensor_preds)
        sensor_preds["trend"] = trend
        ALL_RESULTS[sensor] = sensor_preds

    print("\n==============================")
    print("=== KẾT QUẢ DỰ ĐOÁN (1h–24h) ===")
    print("==============================")
    print(ALL_RESULTS)

    # 🔥 GỬI KẾT QUẢ LÊN FIREBASE
    write_prediction_to_firebase(ALL_RESULTS, fb_ts)

    return ALL_RESULTS
# =======================================================

if __name__ == "__main__":
    predict_future_multi()
