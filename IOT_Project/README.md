IOT_Project/
├── data/                # Dữ liệu CSV, key Firebase JSON
│   └── sample_data_trend.csv
├── model/               # Mô hình đã huấn luyện (.h5, .pkl)
├── src/                 # Code Python
│   ├── train_time_profile.py
│   ├── train_lstm_model.py
│   ├── train_lstm_multistep.py
│   ├── monitor_realtime.py
│   └── plot_time_models.py
├── logs/                # Lưu log anomaly + forecast
├── plots_profile/       # Biểu đồ profile + dashboard
├── requirements.txt     # Danh sách thư viện cần cài
└── README.md            # Tài liệu hướng dẫn


Huấn luyện luyện mô hình/
py src/train_time_profile.py
py src/train_lstm_model.py
py src/train_lstm_multistep.py


Chạy giám sát tập lệnh từ Firebase/
py src/monitor_realtime.py


Cài đặt thư viện/
pip install -r requirements.txt


Giờ bạn chỉ cần dùng 1 file duy nhất:

python train_mysql.py one-step

python train_mysql.py multi-step

python train_mysql.py time-profile