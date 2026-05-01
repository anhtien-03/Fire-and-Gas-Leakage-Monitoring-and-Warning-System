#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Huấn luyện mô hình LSTM multi-step (dự báo 5 phút = 100 bước tới).
- Input : sample_data_trend.csv
- Output: artefacts/lstm_multistep.h5 + scaler_multistep.pkl + history.csv
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
MODEL_PATH   = "model/lstm_multistep.h5"
SCALER_PATH  = "model/scaler_multistep.pkl"
HISTORY_PATH = "data/lstm_multistep_history.csv"

LOOK_BACK   = 120       # số bước nhìn lại (~6 phút dữ liệu)
N_STEPS     = 100       # số bước cần dự báo (~5 phút)
EPOCHS      = 50
BATCH_SIZE  = 64
SENSORS     = ["Temp", "Humidity", "Mq2", "Mq5", "Mq9"]
# ============================

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"],
                                    dayfirst=True, errors="coerce")
    df = df.set_index("datetime").sort_index()
    return df

def create_multistep_sequences(data: np.ndarray, look_back: int, n_steps: int):
    """
    Tạo X (look_back) và y (n_steps ahead).
    Ví dụ: X = 120 bước gần nhất, y = 100 bước kế tiếp.
    """
    X, y = [], []
    for i in range(len(data) - look_back - n_steps):
        X.append(data[i:(i + look_back)])
        y.append(data[(i + look_back):(i + look_back + n_steps)])
    return np.array(X), np.array(y)   # X:(samples, look_back, features), y:(samples, n_steps, features)

def build_lstm(input_shape, output_dim):
    """Xây dựng model LSTM multi-step."""
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
    print("== Huấn luyện LSTM multi-step forecasting ==")

    # 1) Load dữ liệu
    df = load_data(DATA_PATH)
    if not set(SENSORS).issubset(df.columns):
        raise ValueError("Thiếu cột cảm biến trong CSV")
    data = df[SENSORS].values

    # 2) Chuẩn hóa
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # 3) Tạo sequence multi-step
    X, y = create_multistep_sequences(data_scaled, LOOK_BACK, N_STEPS)
    n_samples, seq_len, n_features = X.shape
    print(f"✅ X: {X.shape}, y: {y.shape}")

    # Reshape y thành (samples, n_steps * n_features) để train
    y = y.reshape((y.shape[0], y.shape[1] * y.shape[2]))

    # 4) Chia train/val
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # 5) Build & train
    model = build_lstm((seq_len, n_features), y.shape[1])

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
