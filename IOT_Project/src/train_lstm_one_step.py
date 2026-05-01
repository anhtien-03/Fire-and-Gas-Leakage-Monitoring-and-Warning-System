#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Huấn luyện mô hình LSTM one-step (dự báo bước kế tiếp).
- Input : sample_data_trend.csv (30 ngày, nhiều dòng mỗi giờ)
- Output: artefacts/lstm_model.h5 + scaler_one_step.pkl + history.csv
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras import regularizers

# ========= CẤU HÌNH =========
DATA_PATH    = "data/sample_data_trend.csv"
MODEL_PATH   = "model/lstm_one_step.h5"
SCALER_PATH  = "model/scaler_one_step.pkl"
HISTORY_PATH = "data/lstm_one_step_history.csv"

LOOK_BACK   = 120       # số bước nhìn lại (~6 phút dữ liệu)
EPOCHS      = 50
BATCH_SIZE  = 64
SENSORS     = ["Temp", "Humidity", "Mq2", "Mq5", "Mq9"]
# ============================

def load_data(path: str) -> pd.DataFrame:
    """Đọc CSV và ghép Date+Time thành datetime."""
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"],
                                    dayfirst=True, errors="coerce")
    df = df.set_index("datetime").sort_index()
    return df

def create_sequences(data: np.ndarray, look_back: int):
    """Tạo sliding window sequence cho LSTM."""
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

def build_lstm(input_shape, output_dim):
    """Xây dựng model LSTM (stacked + dropout + regularizer)."""
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

def main():
    print("== Huấn luyện LSTM one-step forecasting ==")

    # 1) Load dữ liệu
    df = load_data(DATA_PATH)
    if not set(SENSORS).issubset(df.columns):
        raise ValueError("Thiếu cột cảm biến trong CSV")
    data = df[SENSORS].values

    # 2) Chuẩn hóa dữ liệu
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # 3) Tạo sequence
    X, y = create_sequences(data_scaled, LOOK_BACK)
    n_samples, seq_len, n_features = X.shape
    print(f"✅ Sequence shape: {X.shape}, y: {y.shape}")

    # 4) Chia train/val
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # 5) Xây dựng & train LSTM
    model = build_lstm((seq_len, n_features), n_features)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    ]

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=callbacks)

    # 6) Lưu scaler và history
    Path(os.path.dirname(SCALER_PATH)).mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(HISTORY_PATH, index=False)

    print(f"✅ Đã lưu best model vào {MODEL_PATH}")
    print(f"✅ Đã lưu scaler vào {SCALER_PATH}")
    print(f"✅ Đã lưu lịch sử train vào {HISTORY_PATH}")

if __name__ == "__main__":
    main()
