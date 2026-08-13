# Fire-and-Gas-Leakage-Monitoring-and-Warning-System

An IoT-based fire and gas leakage monitoring system developed for real-time environmental monitoring and early risk detection.

The system uses an **ESP32**, MQ-series gas sensors, DHT22, Firebase, and a mobile application. The main focus of this project is integrating **Machine Learning** to analyze sensor data, identify trends, and predict future environmental conditions.

## System Architecture

The system consists of four main layers: Perception, Network, Processing, and Application.

![System Architecture](images/system-architecture.png)

## Hardware Architecture

The hardware includes sensors for gas, temperature, and humidity, an ESP32 as the main controller, and output devices for warning and control.

![Hardware Architecture](images/hardware-architecture.png)

## Mobile Application

The mobile application provides real-time monitoring, warning notifications, and remote interaction with the system.

![Mobile Application](images/mobile-application.png)

## Machine Learning

Machine Learning is added as an analysis layer on top of the existing IoT system.

Historical and real-time sensor data are processed to extract time-series features and predict future sensor values using a **Random Forest** model.

```text
Sensor Data
     ↓
Firebase
     ↓
Data Preprocessing
     ↓
Feature Extraction
     ↓
Random Forest
     ↓
Prediction / Trend Analysis
     ↓
Firebase
     ↓
Mobile Application
Sensor Data Analysis
Temperature

Humidity

The collected data is analyzed over time to identify environmental changes and support the prediction process.

Results

The Machine Learning module can:

Analyze sensor trends
Predict future sensor values
Identify unusual environmental changes
Support early warning

The prediction performance depends on the quality, amount, and diversity of the available training data.

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
ESP32 and IoT system development
Sensor data acquisition and processing
Firebase communication
Time-series data preprocessing
Feature engineering
Random Forest regression
Sensor prediction and trend analysis
Integrating Machine Learning into an IoT system
Future Improvements
Collect a larger dataset
Improve feature engineering
Compare different ML models
Improve anomaly detection
Explore lightweight ML for edge devices
Improve prediction accuracy
Purpose

This project was developed to explore the integration of Machine Learning with an IoT safety monitoring system, moving beyond simple threshold-based monitoring toward data-driven trend analysis and early risk detection.

Author

Sâu
