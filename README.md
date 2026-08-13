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
