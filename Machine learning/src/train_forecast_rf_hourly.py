#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Huấn luyện mô hình Random Forest Regressor dự đoán giá trị cảm biến
tại các mốc thời gian tương lai (1h, 5h, 12h, 24h).

Dữ liệu được RESAMPLE thành mỗi giờ 1 dòng → dự báo theo GIỜ thật.

BẢN ĐÃ TỐI ƯU:
- Hybrid resample (mean cho temp/hum, median cho MQ)
- Loại outlier (clip 1%–99%)
- Thêm rolling std, diff, encoding hour_sin/hour_cos
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path

# ================== CONFIG ==================
CSV_PATH = r"C:\Users\tan kiet\Desktop\IOT_Project\data\data_sensor_sorted.csv"
MODEL_DIR = r"C:\Users\tan kiet\Desktop\IOT_Project\model_rf_hourly"

SENSORS = [
    "temperature", "humidity",
    "mq2_Smoke", "mq5_CH4",
    "mq5_LPG", "mq7_CO"
]

FORECAST_LIST = [1, 5, 12, 24]   # dự báo 1h, 5h, 12h, 24h

LAGS = [1, 3, 6, 12, 24, 48]     # lag 1h → 48h
ROLLS = [3, 6, 12, 24]           # rolling window 3h → 24h

DROP_COLS = [
    "M_fire", "M_gas",
    "Warning_fire", "Warning_gas",
    "Detailed_Analysis_Fire", "Detailed_Analysis_Gas",
    "Mode", "Relay1", "Relay2", "Relay3", "Relay4"
]

# ========================================================


def fix_time_string(t: str) -> str:
    """Chuẩn hoá time bị sai format."""
    parts = str(t).split(":")
    if len(parts) != 3:
        return "00:00:00"
    try:
        h = int(parts[0])
        m = int(parts[1])
        s = int(parts[2])
        return f"{h:02d}:{m:02d}:{s:02d}"
    except ValueError:
        return "00:00:00"


def add_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Thêm các feature time-series cho 1 cột:
    - Lag
    - Rolling mean
    - Rolling std
    - Diff
    """
    df = df.copy()

    # Lag
    for l in LAGS:
        df[f"{col}_lag{l}"] = df[col].shift(l)

    # Rolling mean + std (dùng giá trị trước đó để tránh leakage)
    for r in ROLLS:
        df[f"{col}_roll_mean_{r}"] = df[col].shift(1).rolling(r).mean()
        df[f"{col}_roll_std_{r}"] = df[col].shift(1).rolling(r).std()

    # Difference (trend ngắn hạn)
    df[f"{col}_diff1"] = df[col].diff(1)
    df[f"{col}_diff3"] = df[col].diff(3)
    df[f"{col}_diff6"] = df[col].diff(6)

    return df


def build_dataset(df: pd.DataFrame, col: str, forecast_h: int):
    """
    Build dataset cho 1 sensor + 1 horizon.
    - Tạo feature từ col
    - Tạo nhãn tương lai col_future_{forecast_h}h
    - Thêm thông tin thời gian (hour, dow, sin/cos)
    """
    df = df.copy()

    # Tạo lag + rolling + diff
    df = add_features(df, col)

    # Tạo nhãn tương lai theo giờ
    future_col = f"{col}_future_{forecast_h}h"
    df[future_col] = df[col].shift(-forecast_h)

    # Thời gian trong ngày
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek

    # Encode chu kỳ 24h bằng sin/cos
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    # Cột bắt buộc phải đủ dữ liệu
    required = [col, future_col] + [c for c in df.columns if c.startswith(col + "_")]

    print(f"\n=== {col} | Forecast: {forecast_h}h ===")
    print("Rows before cleaning:", df.shape[0])

    # Xoá NaN trên các cột quan trọng
    df = df.dropna(subset=required)
    print("Rows after cleaning:", df.shape[0])

    # Không dùng các cột sau làm feature
    remove_cols = ["date", "time", "timestamp", col, future_col]

    # Lấy tất cả các cột dạng số làm feature
    feature_cols = [
        c for c in df.columns
        if (c not in remove_cols) and (df[c].dtype != "object")
    ]

    print("Feature columns:", feature_cols)

    X = df[feature_cols]
    y = df[future_col]

    return X, y, feature_cols


def main():
    print("===============================================")
    print("=== TRAIN HOURLY MULTI-HORIZON RF MODEL =======")
    print("===============================================")

    Path(MODEL_DIR).mkdir(exist_ok=True, parents=True)

    # ======== 1. Đọc dữ liệu ===========
    df = pd.read_csv(CSV_PATH)
    print("📌 Dữ liệu gốc:", df.shape[0], "dòng")

    # Bỏ cột không cần thiết
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Chuẩn hoá time
    df["time"] = df["time"].astype(str).apply(fix_time_string)

    # Tạo timestamp
    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        dayfirst=True,
        errors="coerce"
    )

    # Xoá dòng timestamp lỗi
    df = df.dropna(subset=["timestamp"])

    # ======== 2. Lấy các cột numeric trước khi xử lý ===========
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Chỉ giữ timestamp + cột số
    df_numeric = df[["timestamp"] + numeric_cols].copy()

    # ======== 3. Loại outlier nhẹ (clip 1%–99%) ===========
    print("🔧 Đang clip outlier (1%–99%) cho các cột numeric...")
    for col in numeric_cols:
        q1 = df_numeric[col].quantile(0.01)
        q99 = df_numeric[col].quantile(0.99)
        df_numeric[col] = df_numeric[col].clip(lower=q1, upper=q99)

    # ======== 4. RESAMPLE mỗi giờ 1 dòng (Hybrid mean/median) ===========
    print("🔧 Đang resample theo giờ (hybrid mean/median)...")

    agg_dict = {}
    for col in numeric_cols:
        if col in ["temperature", "humidity"]:
            agg_dict[col] = "mean"
        elif col in ["mq2_Smoke", "mq5_CH4", "mq5_LPG", "mq7_CO"]:
            agg_dict[col] = "median"
        else:
            # Mặc định: mean cho các cột số khác (IAQ, OAQ, v.v.)
            agg_dict[col] = "mean"

    df_hour = df_numeric.resample("1h", on="timestamp").agg(agg_dict).reset_index()

    print("📌 Dữ liệu sau resample (hourly):", df_hour.shape[0], "dòng")
    print("🔎 Các cột sau resample:", df_hour.columns.tolist())

    # ======== 5. TRAIN TỪNG CẢM BIẾN ===========
    for sensor in SENSORS:

        if sensor not in df_hour.columns:
            print(f"⚠ Bỏ qua {sensor}: không tồn tại trong dữ liệu sau resample.")
            continue

        print("\n======================================")
        print(f"🔧 Training sensor: {sensor}")
        print("======================================")

        # Train từng mốc thời gian dự báo
        for forecast_h in FORECAST_LIST:

            print(f"\n⏳ Training dự báo {forecast_h} giờ...")

            X, y, feature_cols = build_dataset(df_hour, sensor, forecast_h)

            if len(X) < 30:
                print("⚠ Cảnh báo: dữ liệu hơi ít cho sensor này + horizon này!")

            # 20–25% test cho time series (không shuffle)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, shuffle=False
            )

            # Random Forest Regressor
            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, pred)

            print(f"➡ MAE ({sensor}, {forecast_h}h) = {mae:.3f}")

            # Lưu model
            model_path = f"{MODEL_DIR}/{sensor}_rf_hourly_{forecast_h}h.pkl"

            joblib.dump(
                {
                    "model": model,
                    "features": feature_cols,
                    "forecast_h": forecast_h,
                    "target_sensor": sensor
                },
                model_path
            )

            print(f"📁 Saved: {model_path}")

    print("\n🎉 HOÀN THÀNH TRAINING!")
    print(f"📁 Models saved in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
