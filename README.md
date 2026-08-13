# Fire-and-Gas-Leakage-Monitoring-and-Warning-System

An IoT-based fire and gas leakage monitoring system designed for real-time environmental monitoring and early risk detection.

The system uses an ESP32 with MQ-series gas sensors and DHT22 to collect environmental data, control warning devices, and send real-time data to Firebase.

The main focus of this project is to **integrate Machine Learning into the existing monitoring system** to analyze sensor trends and predict future environmental conditions, providing an additional layer for early risk detection.

## System Overview

The system consists of:

- **ESP32** – Central controller and sensor data acquisition
- **MQ-series sensors** – Gas and smoke detection
- **DHT22** – Temperature and humidity monitoring
- **Firebase** – Real-time data storage and communication
- **Warning devices** – Buzzer, warning light, exhaust fan, and pump
- **Mobile application** – Remote monitoring and control
- **Machine Learning module** – Trend analysis and sensor prediction

![System Overview](https://github.com/user-attachments/assets/41bae605-1ae7-4abc-9b48-b510045b9030)

## Existing System

The original system monitors environmental conditions using sensors connected to the ESP32.

Sensor data is processed by the ESP32 and synchronized with Firebase through Wi-Fi. When dangerous conditions are detected, the system can activate warning and control devices.

```text
Sensors
   ↓
ESP32
   ↓
Firebase
   ↓
Mobile Application

This approach mainly relies on sensor values and predefined thresholds for real-time warning.

Machine Learning Integration

Machine Learning is added as an additional analysis layer on top of the existing IoT system.

Instead of relying only on instantaneous sensor values, historical and real-time data are analyzed to identify trends and predict future sensor values.

Sensor Data
     ↓
Firebase
     ↓
Data Preprocessing
     ↓
Feature Extraction
     ↓
Machine Learning Model
     ↓
Prediction / Trend Analysis
     ↓
Firebase
     ↓
Mobile Application

The Machine Learning workflow includes:

Collecting historical sensor data
Data preprocessing and normalization
Noise filtering
Time-based resampling
Time-series feature extraction
Training and testing the model
Predicting future sensor values
Analyzing increasing or decreasing trends

A Random Forest model is used for sensor value prediction.

Why Machine Learning?

A conventional threshold-based system can detect a dangerous condition when a sensor value exceeds a predefined limit.

Machine Learning provides additional information about how the environment is changing over time.

Traditional Approach
Current Sensor Value
        ↓
Compare with Threshold
        ↓
Warning
ML-assisted Approach
Historical + Real-time Data
            ↓
      Machine Learning
            ↓
    Prediction / Trend
            ↓
      Early Warning

The Machine Learning module is designed as a supporting analysis layer. Real-time safety control and warning functions remain handled by the main monitoring system.

Prediction Process

During operation, the ML module retrieves historical and real-time sensor data from Firebase.

The data is processed and transformed into the required features before being passed to the trained Random Forest model.

The predicted values and trends are then stored back in Firebase and made available to the monitoring application.

Firebase
   ↓
Read Sensor Data
   ↓
Data Validation
   ↓
Feature Generation
   ↓
Random Forest Model
   ↓
Future Value Prediction
   ↓
Trend Analysis
   ↓
Firebase
   ↓
Mobile Application
Results

The Machine Learning module can:

Analyze sensor trends
Identify increasing or decreasing environmental conditions
Predict upcoming sensor values
Provide additional information for early warning
Support users in monitoring abnormal environmental changes

The current model is intended primarily for monitoring and prediction support. Its performance is limited by the amount and diversity of available training data.

Technologies
ESP32
MQ-series Gas Sensors
DHT22
Firebase Realtime Database
Python
Machine Learning
Random Forest
Arduino IDE
What I Learned

Through this project, I gained practical experience in:

ESP32-based IoT development
Sensor data acquisition
Firebase real-time communication
Time-series data preprocessing
Feature engineering
Machine Learning model training
Random Forest regression
Sensor value prediction
Trend analysis
Integrating Machine Learning into an IoT system
Future Improvements
Collect a larger and more diverse dataset
Improve feature engineering
Compare Random Forest with other Machine Learning models
Improve anomaly detection
Evaluate the model using additional metrics
Improve prediction accuracy
Explore edge-based Machine Learning
Improve the connection between ML predictions and early-warning decisions
Project Purpose

This project was developed as a learning and research project to explore how Machine Learning can be integrated into an existing IoT safety monitoring system.

The goal is to move beyond simple threshold-based monitoring by using historical and real-time sensor data to analyze environmental trends and support early risk detection.

Tools
Arduino IDE
Python
Firebase
ESP32
Machine Learning
Author

Sâu

Personal IoT and Machine Learning project.


Phần **Random Forest, preprocessing, feature extraction, luồng Firebase → ML → Firebase**, cũng như việc ML đóng vai trò **hỗ trợ phân tích/dự báo chứ không trực tiếp thay thế cơ chế điều khiển an toàn realtime**, đều khớp với nội dung trong tài liệu project của bạn. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

Mình cố tình **không nhồi quá nhiều nội dung IoT/ESP32** vào README, vì điểm bạn muốn showcase ở project này là:

**IoT system → Data collection → Data processing → Machine Learning → Prediction → Early warning.**

Như vậy khi recruiter mở GitHub, họ sẽ nhìn thấy ngay phần **Machine Learning là phần mở rộng có chủ đích**, chứ không bị chìm giữa đống mô tả phần cứng.
ý là hãy viết toàn bộ nội dung cho phần readme để tôi dán vào

À, hiểu ý bạn rồi 😄 Bạn muốn một khối README hoàn chỉnh từ đầu đến cuối, không giải thích thêm, để copy toàn bộ vào README.md.

Mình viết lại theo hướng gọn, rõ, tập trung vào Machine Learning, và chỉ sử dụng những nội dung project của bạn đã có.

# Fire-and-Gas-Leakage-Monitoring-and-Warning-System

An IoT-based fire and gas leakage monitoring system designed for real-time environmental monitoring and early risk detection.

The system uses an **ESP32**, MQ-series gas sensors, and a DHT22 sensor to collect environmental data, process sensor information, control warning devices, and transmit data to Firebase for remote monitoring.

The main focus of this project is to **integrate Machine Learning into the existing IoT system** to analyze sensor trends and predict future environmental conditions, providing additional support for early risk detection.

---

## System Overview

The system consists of several main components:

- **ESP32** – Central controller and sensor data acquisition
- **MQ-series sensors** – Gas and smoke detection
- **DHT22** – Temperature and humidity monitoring
- **Firebase** – Real-time data storage and communication
- **Warning devices** – Buzzer, warning light, exhaust fan, and pump
- **Mobile application** – Remote monitoring and system control
- **Machine Learning module** – Sensor trend analysis and prediction

![System Overview](https://github.com/user-attachments/assets/41bae605-1ae7-4abc-9b48-b510045b9030)

---

## Existing IoT System

The original system continuously collects environmental data from the sensors through the ESP32.

The sensor data is processed locally and synchronized with Firebase through Wi-Fi. When dangerous conditions are detected, the ESP32 can activate the corresponding warning and control devices.

The basic data flow is:

```text
Sensors
   ↓
ESP32
   ↓
Firebase
   ↓
Mobile Application

The original warning mechanism mainly relies on sensor values and predefined thresholds.

Machine Learning Integration

Machine Learning is introduced as an additional analysis layer for the existing monitoring system.

Instead of relying only on instantaneous sensor values, historical and real-time sensor data are analyzed to identify environmental trends and predict future sensor values.

The overall workflow is:

Sensor Data
     ↓
Firebase
     ↓
Data Preprocessing
     ↓
Feature Extraction
     ↓
Machine Learning Model
     ↓
Prediction / Trend Analysis
     ↓
Firebase
     ↓
Mobile Application

The Machine Learning process includes:

Collecting historical sensor data
Data preprocessing
Time normalization
Noise filtering
Resampling sensor data
Time-series feature extraction
Training and testing the model
Predicting future sensor values
Analyzing sensor trends
Machine Learning Model

A Random Forest model is used to predict future sensor values.

The training process starts from historical sensor data stored in CSV format. The data is preprocessed, filtered, resampled, and transformed into time-series features before being divided into training and testing datasets.

The trained model is evaluated using Mean Absolute Error (MAE) and then saved for future prediction.

Historical Sensor Data
        ↓
Data Preprocessing
        ↓
Noise Filtering
        ↓
Time-based Resampling
        ↓
Feature Extraction
        ↓
Train / Test Split
        ↓
Random Forest
        ↓
Model Evaluation
        ↓
Saved Model
Prediction Process

During operation, the Machine Learning module retrieves historical and real-time sensor data from Firebase.

The data is validated and processed to generate the required features. The trained Random Forest model is then used to predict future sensor values and determine the current trend.

The prediction results are synchronized back to Firebase for monitoring through the application.

Firebase
   ↓
Read Sensor Data
   ↓
Data Validation
   ↓
Feature Generation
   ↓
Random Forest Model
   ↓
Future Value Prediction
   ↓
Trend Analysis
   ↓
Firebase
   ↓
Mobile Application
Why Machine Learning?

A conventional monitoring system mainly reacts when a sensor value exceeds a predefined threshold.

Machine Learning provides an additional way to analyze how environmental parameters are changing over time.

Traditional Approach
Current Sensor Value
        ↓
Compare with Threshold
        ↓
Warning
ML-assisted Approach
Historical + Real-time Data
            ↓
      Machine Learning
            ↓
     Prediction / Trend
            ↓
      Early Warning

The Machine Learning module is designed as a supporting analysis layer. The main real-time monitoring and safety control mechanisms remain independent of the prediction model.

Results

The Machine Learning module can be used to:

Analyze increasing and decreasing sensor trends
Predict upcoming sensor values
Identify unusual changes in environmental conditions
Provide additional information for early warning
Support users in monitoring environmental changes

The experimental results show that Machine Learning can support early identification of potentially abnormal environmental trends.

However, the prediction performance is still affected by the amount and diversity of available training data.

Technologies
ESP32
MQ-series Gas Sensors
DHT22
Firebase
Python
Random Forest
Machine Learning
Arduino IDE
What I Learned

Through this project, I gained practical experience in:

ESP32-based IoT development
Sensor data acquisition
Firebase real-time communication
Time-series data preprocessing
Feature extraction
Machine Learning model training
Random Forest regression
Sensor value prediction
Trend analysis
Integrating Machine Learning into an IoT system
Future Improvements

Several improvements can be explored in future versions:

Collect a larger and more diverse sensor dataset
Improve feature engineering
Compare Random Forest with other Machine Learning models
Improve anomaly detection
Evaluate the model using additional metrics
Improve prediction accuracy
Explore lightweight Machine Learning models for edge devices
Improve the integration between ML predictions and early-warning mechanisms
Project Purpose

This project was developed as a learning and research project to explore the integration of Machine Learning with an existing IoT safety monitoring system.

The main objective is to move beyond simple threshold-based monitoring by using historical and real-time sensor data to analyze environmental trends and support early risk detection.

The project combines:

Embedded System
      +
      IoT
      +
 Sensor Data
      +
Machine Learning
      ↓
Early Risk Detection
Tools
Arduino IDE
Python
Firebase
ESP32
Machine Learning
Git / GitHub
Author

Sâu
