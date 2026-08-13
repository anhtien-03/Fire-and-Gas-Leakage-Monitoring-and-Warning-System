Fire-and-Gas-Leakage-Monitoring-and-Warning-System

An IoT-based fire and gas leakage monitoring system developed for real-time environmental monitoring and early risk detection. The system uses an ESP32, MQ-series gas sensors, and DHT22 to collect environmental data, process sensor information, control warning devices, and transmit data to Firebase for remote monitoring. The main focus of this project is to integrate Machine Learning into the existing IoT system to analyze sensor trends and predict future environmental conditions, providing additional support for early risk detection.

System Overview

The system consists of an ESP32 as the central controller, MQ-series sensors for gas and smoke detection, a DHT22 for temperature and humidity monitoring, Firebase for real-time data storage and communication, warning devices including a buzzer, warning light, exhaust fan and pump, a mobile application for remote monitoring and control, and a Machine Learning module for sensor trend analysis and prediction.

The original system continuously collects environmental data from the sensors through the ESP32. The sensor data is processed locally and synchronized with Firebase through Wi-Fi. When dangerous conditions are detected, the ESP32 can activate the corresponding warning and control devices.

Sensors
   ↓
ESP32
   ↓
Firebase
   ↓
Mobile Application
Machine Learning Integration

Machine Learning is introduced as an additional analysis layer for the existing monitoring system. Instead of relying only on instantaneous sensor values and predefined thresholds, historical and real-time sensor data are analyzed to identify environmental trends and predict future sensor values.

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

The Machine Learning process includes collecting historical sensor data, preprocessing and normalizing the data, filtering noise, resampling sensor data, extracting time-series features, training and testing the model, predicting future sensor values, and analyzing sensor trends.

Machine Learning Model

A Random Forest model is used to predict future sensor values. Historical sensor data is processed before training, including time normalization, noise filtering, resampling, and time-series feature extraction. The dataset is then divided into training and testing sets. The trained model is evaluated using Mean Absolute Error (MAE) and saved for future prediction.

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
Prediction and Early Warning

During operation, the Machine Learning module retrieves historical and real-time sensor data from Firebase, validates and processes the data, generates the required features, and uses the trained Random Forest model to predict future sensor values and analyze trends. The prediction results are then synchronized back to Firebase and displayed through the monitoring application.

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

The conventional monitoring system mainly reacts when sensor values exceed predefined thresholds. The Machine Learning module provides an additional layer by analyzing how environmental parameters change over time, helping identify increasing or decreasing trends and supporting early warning. The ML module is designed as a supporting analysis layer, while the main real-time monitoring and safety control mechanisms remain independent of the prediction model.

Results

The Machine Learning module can analyze sensor trends, predict upcoming sensor values, identify unusual changes in environmental conditions, and provide additional information for early warning. Experimental results indicate that the ML module can support early identification of potentially abnormal environmental trends. However, prediction performance is still affected by the amount and diversity of available training data.

Technologies
ESP32
MQ-series Gas Sensors
DHT22
Firebase
Python
Random Forest
Machine Learning
Arduino IDE
Git / GitHub
What I Learned

Through this project, I gained practical experience in ESP32-based IoT development, sensor data acquisition, Firebase real-time communication, time-series data preprocessing, feature extraction, Machine Learning model training, Random Forest regression, sensor value prediction, trend analysis, and integrating Machine Learning into an existing IoT system.

Future Improvements

Future improvements include collecting a larger and more diverse sensor dataset, improving feature engineering, comparing Random Forest with other Machine Learning models, improving anomaly detection, evaluating the model using additional metrics, improving prediction accuracy, exploring lightweight Machine Learning models for edge devices, and improving the integration between ML predictions and early-warning mechanisms.

Project Purpose

This project was developed as a learning and research project to explore how Machine Learning can be integrated into an existing IoT safety monitoring system. The main objective is to move beyond simple threshold-based monitoring by using historical and real-time sensor data to analyze environmental trends and support early risk detection.

Embedded System
      +
      IoT
      +
 Sensor Data
      +
Machine Learning
      ↓
Early Risk Detection
Author

Sâu
