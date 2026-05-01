#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Huấn luyện mô hình phát hiện bất thường theo khung giờ từ dữ liệu CSV 30 ngày.

- Mô hình thống kê: median, MAD, q10, q90, min, max, count
- Lưu profile ra artefacts/time_profile.pkl + profile_summary.csv
"""

from pathlib import Path
from typing import Dict
import joblib
import pandas as pd
from datetime import datetime

# ======= Cấu hình =======
CSV_PATH   = "data/sample_data_trend.csv"
DATE_COL   = "Date"
TIME_COL   = "Time"
SENSORS    = ["Temp", "Humidity", "Mq2", "Mq5", "Mq9"]  
OUT_PATH   = "model/time_profile.pkl"
SUMMARY_CSV= "data/time_profile_summary.csv"
MIN_COUNT  = 3
DAYFIRST   = True
# =========================

def robust_stats(series: pd.Series) -> Dict[str, float]:
    """Tính median, MAD, q10, q90, min, max, count."""
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

def build_timestamp(df: pd.DataFrame, date_col: str, time_col: str, dayfirst: bool) -> pd.Series:
    """Ghép cột ngày + giờ thành timestamp."""
    try:
        ts = pd.to_datetime(
            df[date_col] + " " + df[time_col],
            dayfirst=dayfirst,
            errors="coerce"
        )
    except Exception:
        ts = pd.to_datetime(df[date_col], dayfirst=dayfirst, errors="coerce")
    return ts

def main():
    # 1) Đọc CSV
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        raise ValueError(f"File CSV rỗng: {CSV_PATH}")

    # 2) Tạo timestamp và cột giờ
    ts = build_timestamp(df, DATE_COL, TIME_COL, DAYFIRST)
    df = df.assign(timestamp=ts).dropna(subset=["timestamp"]).copy()
    df["hour"] = df["timestamp"].dt.hour

    # 3) Chuyển các cột cảm biến sang số
    for col in SENSORS:
        if col not in df.columns:
            raise KeyError(f"Không tìm thấy cột '{col}' trong CSV.")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=SENSORS)

    # 4) Tính profile cho từng giờ
    profiles = {}
    summary_rows = []
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
            prof_df = pd.DataFrame(rows).set_index("hour").sort_index()
            profiles[col] = prof_df.to_dict(orient="index")
        else:
            profiles[col] = {}

    # 5) Lưu mô hình
    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "profiles": profiles,
        "target_sensors": SENSORS,
        "meta": {
            "csv": str(Path(CSV_PATH).resolve()),
            "date_col": DATE_COL,
            "time_col": TIME_COL,
            "min_count": MIN_COUNT,
            "dayfirst": DAYFIRST,
            "version": f"time_profile_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        },
    }
    joblib.dump(bundle, out_path)

    # 6) Xuất summary CSV
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)

    # 7) In tóm tắt
    print(f"✅ Đã lưu mô hình: {out_path}")
    print(f"✅ Đã lưu summary CSV: {SUMMARY_CSV}")
    for col in SENSORS:
        prof = profiles.get(col, {})
        print(f"\n=== Profile theo giờ cho {col} (count>={MIN_COUNT}) ===")
        if not prof:
            print("  (không có giờ nào đủ mẫu)")
            continue
        for h in sorted(prof.keys()):
            s = prof[h]
            print(f"Giờ {h:02d}: n={s['count']}  "
                  f"median={s['median']:.2f}  MAD={s['mad']:.2f}  "
                  f"q10={s['q10']:.2f}  q90={s['q90']:.2f}  "
                  f"min={s['min']:.2f}  max={s['max']:.2f}")

if __name__ == "__main__":
    main()
