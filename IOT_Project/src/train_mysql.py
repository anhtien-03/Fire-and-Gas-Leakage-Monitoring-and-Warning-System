#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Huấn luyện mô hình LSTM & Time Profile từ MySQL (sensor_data).
- one-step    : dự báo bước kế tiếp
- multi-step  : dự báo nhiều bước tới
- time-profile: profile thống kê theo giờ
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras import regularizers

# ========= CẤU HÌNH =========
HOST = "localhost"
USER = "root"
PASSWORD = "WJ28@krhps"
PORT = 3306
DB = "iot_data"
TABLE = "sensor_data"

MODEL_DIR = "model"
DATA_DIR = "data"

LOOK_BACK   = 120
N_STEPS     = 100
EPOCHS      = 50
BATCH_SIZE  = 64
SENSORS     = ["Temp", "Humidity", "Mq2", "Mq5", "Mq9"]
MIN_COUNT   = 3
# ============================


# ====== KẾT NỐI MYSQL ======
def load_data_from_mysql() -> pd.DataFrame:
    """Đọc dữ liệu từ MySQL và ghép datetime."""
    encoded_pw = quote_plus(PASSWORD)
    engine = create_engine(f"mysql+pymysql://{USER}:{encoded_pw}@{HOST}:{PORT}/{DB}")
    query = f"""
        SELECT 
            DATE_FORMAT(date, '%%Y-%%m-%%d') AS Date,
            TIME_FORMAT(time, '%%H:%%i:%%s') AS Time,
            {','.join(SENSORS)}
        FROM {TABLE}
        ORDER BY date ASC, time ASC;
    """
    df = pd.read_sql(query, engine)
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    print(f"✅ Đọc {len(df)} dòng dữ liệu từ MySQL")
    return df


# ====== ONE-STEP LSTM ======
def create_sequences(data: np.ndarray, look_back: int):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

def build_lstm_one_step(input_shape, output_dim):
    model = Sequential([
        LSTM(128, return_sequences=True, activation="tanh",
             kernel_regularizer=regularizers.l2(1e-4),
             input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False, activation="tanh",
             kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(0.2),
        Dense(output_dim)
    ])
    model.compile(optimizer="adam", loss=Huber())
    return model

def train_one_step(df):
    print("== Huấn luyện LSTM one-step ==")
    data = df[SENSORS].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = create_sequences(data_scaled, LOOK_BACK)
    n_samples, seq_len, n_features = X.shape
    print(f"✅ Sequence: {X.shape}, y: {y.shape}")

    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_lstm_one_step((seq_len, n_features), n_features)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(f"{MODEL_DIR}/lstm_one_step.h5", monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    ]

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=callbacks)

    joblib.dump(scaler, f"{MODEL_DIR}/scaler_one_step.pkl")
    pd.DataFrame(history.history).to_csv(f"{DATA_DIR}/lstm_one_step_history.csv", index=False)
    print("✅ Đã lưu model one-step.")


# ====== MULTI-STEP LSTM ======
def create_multistep_sequences(data: np.ndarray, look_back: int, n_steps: int):
    X, y = [], []
    for i in range(len(data) - look_back - n_steps):
        X.append(data[i:(i + look_back)])
        y.append(data[(i + look_back):(i + look_back + n_steps)])
    return np.array(X), np.array(y)

def build_lstm_multi_step(input_shape, output_dim):
    model = Sequential([
        LSTM(128, return_sequences=True, activation="tanh",
             kernel_regularizer=regularizers.l2(1e-4),
             input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False, activation="tanh",
             kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(0.2),
        Dense(output_dim)
    ])
    model.compile(optimizer="adam", loss=Huber())
    return model

def train_multi_step(df):
    print("== Huấn luyện LSTM multi-step ==")
    data = df[SENSORS].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = create_multistep_sequences(data_scaled, LOOK_BACK, N_STEPS)
    n_samples, seq_len, n_features = X.shape
    print(f"✅ X: {X.shape}, y: {y.shape}")

    y = y.reshape((y.shape[0], y.shape[1] * y.shape[2]))
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_lstm_multi_step((seq_len, n_features), y.shape[1])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(f"{MODEL_DIR}/lstm_multistep.h5", monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    ]

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=callbacks)

    joblib.dump(scaler, f"{MODEL_DIR}/scaler_multistep.pkl")
    pd.DataFrame(history.history).to_csv(f"{DATA_DIR}/lstm_multistep_history.csv", index=False)
    print("✅ Đã lưu model multi-step.")


# ====== TIME PROFILE ======
def robust_stats(series: pd.Series):
    med = float(series.median())
    mad = float((series - med).abs().median())
    q10 = float(series.quantile(0.10))
    q90 = float(series.quantile(0.90))
    return {
        "median": med,
        "mad": mad,
        "q10": q10,
        "q90": q90,
        "min": float(series.min()),
        "max": float(series.max()),
        "count": int(series.shape[0])
    }

def train_time_profile(df):
    print("== Huấn luyện Time Profile ==")
    df["hour"] = df.index.hour
    profiles, summary_rows = {}, []

    for col in SENSORS:
        g = df.groupby("hour")[col]
        rows = []
        for h, s in g:
            if s.shape[0] < MIN_COUNT:
                continue
            stats = robust_stats(s)
            stats["hour"] = int(h)
            rows.append(stats)
            summary_rows.append({"sensor": col, "hour": h, **stats})

        if rows:
            profiles[col] = pd.DataFrame(rows).set_index("hour").sort_index().to_dict(orient="index")
        else:
            profiles[col] = {}

    bundle = {
        "profiles": profiles,
        "target_sensors": SENSORS,
        "meta": {
            "source": "mysql",
            "table": TABLE,
            "min_count": MIN_COUNT,
            "version": f"time_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
    }
    joblib.dump(bundle, f"{MODEL_DIR}/time_profile.pkl")
    pd.DataFrame(summary_rows).to_csv(f"{DATA_DIR}/time_profile_summary.csv", index=False)
    print("✅ Đã lưu time profile.")


# ====== MAIN ======
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Vui lòng chọn chế độ train: one-step | multi-step | time-profile")
        sys.exit(1)

    mode = sys.argv[1]
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    df = load_data_from_mysql()

    if mode == "one-step":
        train_one_step(df)
    elif mode == "multi-step":
        train_multi_step(df)
    elif mode == "time-profile":
        train_time_profile(df)
    else:
        print("❌ Chế độ không hợp lệ. Chọn: one-step | multi-step | time-profile")
