#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Realtime anomaly watch HYBRID + Multi-step Forecast (Firebase input, offline output).
- Đọc dữ liệu sensor từ Firebase RTDB.
- Phát hiện bất thường (Profile + LSTM one-step).
- Dự báo xu hướng 5 phút tới bằng LSTM multi-step.
- Ghi log ra file (anomaly + forecast).
- Gửi cảnh báo qua Firebase Cloud Messaging (FCM API v1).
- Có 2 luồng cảnh báo: Basic (từ ESP warning/dangerType) & Detailed (phân tích mô hình).
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any
from collections import deque
import requests

import joblib
import numpy as np
from tensorflow.keras.models import load_model

import firebase_admin
from firebase_admin import credentials, db

# Google Auth để gửi FCM API v1
from google.oauth2 import service_account
import google.auth.transport.requests

# ========= CẤU HÌNH =========
SERVICE_ACCOUNT_PATH = "thiet-bi-canh-bao-chay-khi-gas-firebase-adminsdk-fbsvc-181eb6828d.json"
DATABASE_URL         = "https://thiet-bi-canh-bao-chay-khi-gas-default-rtdb.firebaseio.com/"
SENSOR_NODE          = "users/0776707441/123456/ESP32"

PROFILE_MODEL_PATH   = "model/time_profile.pkl"
LSTM_ONE_PATH        = "model/lstm_one_step.h5"
LSTM_MULTI_PATH      = "model/lstm_multistep.h5"
SCALER_ONE_PATH      = "model/scaler_one_step.pkl"
SCALER_MULTI_PATH    = "model/scaler_multistep.pkl"

MODEL_SENSORS = ["Temp", "Humidity", "Mq2", "Mq5", "Mq9"]
FB_FIELD_MAP  = {
    "Temp": "temperature",
    "Humidity": "humidity",
    "Mq2": "mq2",
    "Mq5": "mq5",
    "Mq9": "mq9"
}

LOOK_BACK = 120
N_STEPS   = 100
K_MAD = 3.5
USE_QUANTILE = True

SEVERITY_THRESHOLDS = {"critical": 0.90, "high": 0.75, "medium": 0.50}
FUSION_WEIGHTS = {"profile": 0.6, "lstm": 0.4}
FUSION_ANOMALY_THRESHOLD = 0.5

POLL_INTERVAL = 1

# ========= FCM CONFIG (API v1) =========
SERVICE_ACCOUNT_FILE = SERVICE_ACCOUNT_PATH
PROJECT_ID = "thiet-bi-canh-bao-chay-khi-gas"
FCM_DEVICE_TOKEN = "cQqJUAaDQG-VAtNsuRXOsD:APA91bF-KQpUvJDZjIV_mpR9hsoIb3p4BNXJlQnEnE-UpPT37YRU5WmmaXzCoRhZZKwD7jclYsfAH2Ij-M6CaDib1IJVgO405NH9ArhJZ_wDOE9jhQpUFdA"
SCOPES = ["https://www.googleapis.com/auth/firebase.messaging"]
# =======================================

# logging
logging.basicConfig(filename="logs/anomaly.log",
                    level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
forecast_logger = logging.getLogger("forecast")
forecast_handler = logging.FileHandler("logs/forecast.log")
forecast_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
forecast_logger.addHandler(forecast_handler)
forecast_logger.setLevel(logging.INFO)

# history
history = {s: deque(maxlen=LOOK_BACK) for s in MODEL_SENSORS}

# ===== Helper functions =====
def safe_float(v) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")

def z_to_percent(z_abs: float) -> float:
    from math import erf, sqrt
    pct = (1.0 + erf(z_abs / sqrt(2.0))) / 2.0
    return round(pct * 100.0, 2)

def score_profile(x: float, prof: Dict[str, float]) -> Dict[str, Any]:
    if not prof:
        return {"status": "no_profile"}
    med = prof["median"]
    mad = prof["mad"] if prof["mad"] != 0 else 1e-6
    robust_sigma = 1.4826 * mad
    z = abs(x - med) / robust_sigma
    severity = 1 - np.exp(-z)
    out_of_band = (x < prof["q10"]) or (x > prof["q90"]) if USE_QUANTILE else False
    is_anom = bool((z > K_MAD) or out_of_band)
    return {
        "status": "anomaly" if is_anom else "normal",
        "value": round(float(x), 3),
        "severity": round(float(severity), 3),
        "z_mad": round(float(z), 3),
        "percentile_normal": z_to_percent(z),
        "median": round(float(med), 3),
        "q10": round(float(prof["q10"]), 3),
        "q90": round(float(prof["q90"]), 3)
    }

def score_lstm(sensor: str, model, scaler, x: float) -> Dict[str, Any]:
    try:
        hist = history[sensor]
        hist.append(x)
        if len(hist) < LOOK_BACK:
            return {"status": "warming_up"}

        latest_seq = []
        for s in MODEL_SENSORS:
            if len(history[s]) == LOOK_BACK:
                latest_seq.append(history[s])
            else:
                return {"status": "warming_up"}
        latest_seq = np.array(latest_seq).T

        scaled_seq = scaler.transform(latest_seq)
        x_input = np.expand_dims(scaled_seq, axis=0)
        y_pred_scaled = model.predict(x_input, verbose=0)
        y_pred = scaler.inverse_transform(y_pred_scaled)[0]

        resid = x - y_pred[MODEL_SENSORS.index(sensor)]
        sigma = np.std(latest_seq[:, MODEL_SENSORS.index(sensor)]) + 1e-6
        sev = 1 - np.exp(-abs(resid) / sigma)
        status = "anomaly" if sev >= 0.85 else "normal"
        return {
            "status": status,
            "value": round(float(x), 3),
            "forecast": round(float(y_pred[MODEL_SENSORS.index(sensor)]), 3),
            "resid": round(float(resid), 3),
            "severity": round(float(sev), 3)
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}

def fuse_scores(res_prof: Dict[str, Any], res_lstm: Dict[str, Any]) -> Dict[str, Any]:
    sev_prof = res_prof.get("severity", 0)
    sev_lstm = res_lstm.get("severity", 0)
    if res_lstm.get("status") == "warming_up":
        return {"severity": sev_prof, "status": res_prof.get("status")}
    fused_sev = FUSION_WEIGHTS["profile"] * sev_prof + FUSION_WEIGHTS["lstm"] * sev_lstm
    anomaly_condition = ((res_prof.get("status") == "anomaly") or 
                         (res_lstm.get("status") == "anomaly")) and (fused_sev >= FUSION_ANOMALY_THRESHOLD)
    status = "anomaly" if anomaly_condition else "normal"
    return {"severity": round(fused_sev, 3), "status": status}

def decide_overall_status(per_sensor: Dict[str, Dict[str, Any]]) -> str:
    max_sev = max([r.get("severity", 0.0) for r in per_sensor.values() if r.get("status") == "anomaly"] or [0.0])
    if max_sev >= SEVERITY_THRESHOLDS["critical"]:
        return "Khan cap"
    if max_sev >= SEVERITY_THRESHOLDS["high"]:
        return "Rat nguy hiem"
    if max_sev >= SEVERITY_THRESHOLDS["medium"]:
        return "Nguy hiem"
    return "An toan"

def multi_step_forecast(model, scaler):
    try:
        if not all(len(history[s]) == LOOK_BACK for s in MODEL_SENSORS):
            return {"status": "warming_up"}

        latest_seq = np.array([history[s] for s in MODEL_SENSORS]).T
        scaled_seq = scaler.transform(latest_seq)
        x_input = np.expand_dims(scaled_seq, axis=0)

        y_pred_scaled = model.predict(x_input, verbose=0)
        y_pred = scaler.inverse_transform(
            y_pred_scaled.reshape(N_STEPS, len(MODEL_SENSORS))
        )

        summary = {}
        log_summary = {}
        for i, sensor in enumerate(MODEL_SENSORS):
            series = y_pred[:, i]
            trend_symbol = "↑" if series[-1] > series[0] else "↓"
            trend_text = "up" if series[-1] > series[0] else "down"

            summary[sensor] = {
                "min": float(series.min()),
                "max": float(series.max()),
                "last": float(series[-1]),
                "trend_symbol": trend_symbol,
            }
            log_summary[sensor] = {
                "min": float(series.min()),
                "max": float(series.max()),
                "last": float(series[-1]),
                "trend": trend_text
            }

        forecast_logger.info(f"Forecast summary: {log_summary}")
        return summary

    except Exception as e:
        forecast_logger.error(f"Lỗi forecast: {e}")
        return {"status": "error", "detail": str(e)}

# ===== Firebase =====
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        firebase_admin.initialize_app(cred, {"databaseURL": DATABASE_URL})

def read_current_snapshot() -> Dict[str, Any]:
    return db.reference(SENSOR_NODE).get() or {}

# ===== Alert format =====
def format_combined_alert(per_sensor_results: Dict[str, Any],
                          forecast_summary: Dict[str, Any],
                          overall: str) -> str:
    """Cảnh báo bất thường từ mô hình AI"""
    units = {
        "Temp": "°C",
        "Humidity": "%",
        "Mq2": "ppm",
        "Mq5": "ppm",
        "Mq9": "ppm"
    }

    lines = [f"🚨 [AI] Cảnh báo bất thường - Hệ thống: {overall}"]

    for sensor, res in per_sensor_results.items():
        if res.get("status") == "anomaly":
            val = res["profile"].get("value", "-")
            severity = res.get("severity", 0.0)
            if severity >= 0.85:
                level_text = "Khẩn cấp"
            elif severity >= 0.70:
                level_text = "Rất nguy hiểm"
            elif severity >= 0.45:
                level_text = "Nguy hiểm"
            else:
                level_text = "Cảnh báo nhẹ"
            # Xu hướng dự báo
            trend_txt = "Không có dữ liệu"
            if forecast_summary and sensor in forecast_summary:
                trend_symbol = forecast_summary[sensor].get("trend_symbol", "-")
                trend_txt = f"Xu hướng: {trend_symbol} {'tăng' if trend_symbol == '↑' else 'giảm'}"
            # Gộp lại một dòng đẹp hơn
            lines.append(
                f"- {sensor}: {val}{units.get(sensor, '')} | "
                f"Mức độ: {severity:.2f} ({level_text}) | {trend_txt}"
            )
    lines.append(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)

def format_basic_alert(snapshot: Dict[str, Any]) -> str:
    """Cảnh báo cơ bản từ ESP (chuẩn hóa warning & dangerType)"""
    warning_raw = snapshot.get("warning", "An toan")
    danger_raw = snapshot.get("dangerType", "unknown")

    # Map dangerType (không dấu) sang tiếng Việt có dấu
    danger_map = {
        "Chay": "Cháy",
        "Ro ri Gas": "Rò rỉ Gas",
        "unknown": "Không xác định"
    }
    danger = danger_map.get(danger_raw, danger_raw)

    # Map warning (không dấu) sang tiếng Việt có dấu
    warning_map = {
        "An toan": "An toàn",
        "Nguy hiem": "Nguy hiểm",
        "Rat nguy hiem": "Rất nguy hiểm",
        "Khan cap": "Khẩn cấp"
    }
    warning = warning_map.get(warning_raw, warning_raw)

    return (
        f"⚠️ [ESP] Cảnh báo cơ bản\n"
        f"Cảnh báo: {danger}\n"
        f"Trạng thái: {warning}\n"
        f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

# ===== FCM push notification (API v1) =====
def send_basic_alert(title: str, body: str):
    """Cảnh báo cơ bản từ ESP"""
    try:
        credentials_sa = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
        auth_req = google.auth.transport.requests.Request()
        credentials_sa.refresh(auth_req)
        access_token = credentials_sa.token

        url = f"https://fcm.googleapis.com/v1/projects/{PROJECT_ID}/messages:send"
        message = {
            "message": {
                "token": FCM_DEVICE_TOKEN,
                "notification": {"title": f"⚠️ [ESP] {title}", "body": body}
            }
        }
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.post(url, headers=headers, json=message)

        if response.status_code == 200:
            print("✅ Đã gửi Basic Alert thành công")
        else:
            print("❌ Lỗi gửi Basic Alert:", response.text)

    except Exception as e:
        print("❌ Exception khi gửi Basic Alert:", e)

def send_detailed_alert(title: str, body: str):
    """Cảnh báo bất thường từ mô hình phân tích"""
    try:
        credentials_sa = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
        auth_req = google.auth.transport.requests.Request()
        credentials_sa.refresh(auth_req)
        access_token = credentials_sa.token

        url = f"https://fcm.googleapis.com/v1/projects/{PROJECT_ID}/messages:send"
        message = {
            "message": {
                "token": FCM_DEVICE_TOKEN,
                "notification": {"title": f"🚨 [AI] {title}", "body": body}
            }
        }
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.post(url, headers=headers, json=message)

        if response.status_code == 200:
            print("✅ Đã gửi Detailed Alert thành công")
        else:
            print("❌ Lỗi gửi Detailed Alert:", response.text)

    except Exception as e:
        print("❌ Exception khi gửi Detailed Alert:", e)

# ===== Main loop =====
last_overall_status = "An toan"
last_basic_status = "An toan"

def main():
    global last_overall_status, last_basic_status
    init_firebase()
    profile_bundle = joblib.load(PROFILE_MODEL_PATH)
    profiles: Dict[str, Dict[int, Dict[str, float]]] = profile_bundle["profiles"]

    lstm_one   = load_model(LSTM_ONE_PATH, compile=False)
    lstm_multi = load_model(LSTM_MULTI_PATH, compile=False)
    scaler_one = joblib.load(SCALER_ONE_PATH)
    scaler_multi = joblib.load(SCALER_MULTI_PATH)

    print("== HYBRID Anomaly Detection + Multi-step Forecast (Firebase input, offline output) ==")

    while True:
        try:
            snap = read_current_snapshot()
            if not snap:
                print("⚠️ Không đọc được dữ liệu từ Firebase!")
                time.sleep(POLL_INTERVAL)
                continue

            # ====== Luồng 1: Cảnh báo cơ bản từ ESP ======
            warning = snap.get("warning", "An toan")
            if warning in ["Nguy hiem", "Rat nguy hiem", "Khan cap"] and warning != last_basic_status:
                msg = format_basic_alert(snap)
                print("\n>>> BASIC ALERT <<<\n", msg)
                send_basic_alert("Cảnh báo cơ bản", msg)
            last_basic_status = warning

            # ====== Luồng 2: Cảnh báo bất thường từ mô hình ======
            now = datetime.now()
            hour = now.hour
            per_sensor_results, notes = {}, []

            for sensor in MODEL_SENSORS:
                fb_key = FB_FIELD_MAP[sensor]
                x = safe_float(snap.get(fb_key, np.nan))
                if np.isnan(x):
                    per_sensor_results[sensor] = {"status": "missing_input"}
                    continue
                prof_for_hour = profiles.get(sensor, {}).get(hour)
                res_prof = score_profile(x, prof_for_hour)
                res_lstm = score_lstm(sensor, lstm_one, scaler_one, x)
                fused = fuse_scores(res_prof, res_lstm)
                per_sensor_results[sensor] = {
                    "status": fused["status"],
                    "severity": fused["severity"],
                    "profile": res_prof,
                    "lstm": res_lstm
                }
                if fused["status"] == "anomaly":
                    notes.append(f"{sensor} bất thường")

            overall = decide_overall_status(per_sensor_results)
            forecast_summary = multi_step_forecast(lstm_multi, scaler_multi)

            print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}]")
            print(f"Trạng thái hệ thống: {overall}")
            print_sensor_table(per_sensor_results)
            print_forecast_table(forecast_summary)
            logging.info(f"Status={overall}, Notes={notes}")

            if overall in ["Nguy hiem", "Rat nguy hiem", "Khan cap"]:
                if overall != last_overall_status:
                    msg = format_combined_alert(per_sensor_results, forecast_summary, overall)
                    print("\n>>> DETAILED ALERT <<<\n", msg)
                    send_detailed_alert("Cảnh báo phân tích", msg)
            last_overall_status = overall

        except Exception as e:
            print("❌ Lỗi vòng lặp:", e)
            logging.error(f"Lỗi runtime: {str(e)}")

        time.sleep(POLL_INTERVAL)

# ===== Print helpers =====
def print_sensor_table(per_sensor: Dict[str, Any]):
    header = (
        f"{'Sensor':<10} | {'Value':<10} | {'Forecast(1-step)':<16} | "
        f"{'Median':<7} | {'q10':<9} | {'q90':<9} | "
        f"{'Severity':<8} | {'Status':<8}"
    )
    print("Phân tích dữ liệu cảm biến:")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for sensor, res in per_sensor.items():
        prof = res.get("profile", {})
        lstm = res.get("lstm", {})
        print(
            f"{sensor:<10} | {prof.get('value','-')!s:<10} | {lstm.get('forecast','-')!s:<16} | "
            f"{prof.get('median','-')!s:<7} | {prof.get('q10','-')!s:<9} | {prof.get('q90','-')!s:<9} | "
            f"{res.get('severity','-')!s:<8} | {res.get('status','-'):<8}"
        )
    print("-" * len(header))

def print_forecast_table(forecast_summary: Dict[str, Any]):
    header = f"{'Sensor':<10} | {'Min':<10} | {'Max':<10} | {'Last':<10} | {'Trend':<6}"
    print("\nDự báo 5 phút tới:")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    if not forecast_summary or forecast_summary.get("status") == "warming_up":
        print(" Forecast 5 phút tới chưa sẵn sàng (warming_up)")
    elif forecast_summary.get("status") == "error":
        print(" Lỗi khi dự báo:", forecast_summary.get("detail", ""))
    else:
        for sensor, stats in forecast_summary.items():
            print(
                f"{sensor:<10} | {stats['min']:<10.3f} | {stats['max']:<10.3f} | "
                f"{stats['last']:<10.3f} | {stats['trend_symbol']:<6}"
            )
    print("-" * len(header))

if __name__ == "__main__":
    main()
