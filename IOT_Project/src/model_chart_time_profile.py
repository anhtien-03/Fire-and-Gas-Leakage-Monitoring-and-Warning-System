#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vẽ biểu đồ profile từ time_profile.pkl (improved).
- Input : model/time_profile.pkl
- Output: plots_profile/*.png + dashboard.pdf
"""

import os
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

MODEL_PATH = "model/time_profile.pkl"
OUTPUT_DIR = "plots_profile"

def plot_profile(sensor, profile, pdf=None):
    hours = sorted(profile.keys())
    median_vals = [profile[h]["median"] for h in hours]
    q10_vals = [profile[h]["q10"] for h in hours]
    q90_vals = [profile[h]["q90"] for h in hours]
    mad_vals = [profile[h]["mad"] for h in hours]

    # MAD band
    mad_low = [m - mad for m, mad in zip(median_vals, mad_vals)]
    mad_high = [m + mad for m, mad in zip(median_vals, mad_vals)]

    plt.figure(figsize=(10, 5))
    plt.plot(hours, median_vals, label="Median", color="blue", marker="o")
    plt.fill_between(hours, q10_vals, q90_vals,
                     color="lightblue", alpha=0.3, label="Q10-Q90 range")
    plt.fill_between(hours, mad_low, mad_high,
                     color="orange", alpha=0.2, label="±MAD")
    plt.xlabel("Hour of day")
    plt.ylabel(sensor)
    plt.title(f"Profile for {sensor}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{sensor}.png")
    plt.savefig(out_path)
    print(f"✅ Saved plot: {out_path}")

    if pdf is not None:
        pdf.savefig()
    plt.close()

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Không tìm thấy file {MODEL_PATH}")
        return

    bundle = joblib.load(MODEL_PATH)
    profiles = bundle["profiles"]

    sensors = ["Temp", "Humidity", "Mq2", "Mq5", "Mq9"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUTPUT_DIR, "dashboard.pdf")
    with PdfPages(pdf_path) as pdf:
        for sensor in sensors:
            if sensor in profiles:
                plot_profile(sensor, profiles[sensor], pdf=pdf)

    print(f"📊 Dashboard tổng hợp lưu tại: {pdf_path}")

if __name__ == "__main__":
    main()
