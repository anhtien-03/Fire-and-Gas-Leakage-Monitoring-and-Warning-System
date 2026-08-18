# Fire-and-Gas-Leakage-Monitoring-and-Warning-System

An IoT-based fire and gas leakage monitoring system developed for real-time environmental monitoring and early risk detection.

The system uses an **ESP32**, MQ-series gas sensors, DHT22, Firebase, and a mobile application. The main focus of this project is integrating **Machine Learning** to analyze sensor data, identify trends, and predict future environmental conditions.

## System Architecture

The system consists of four main layers: Perception, Network, Processing, and Application.

<img width="1162" height="475" alt="image" src="https://github.com/user-attachments/assets/6968a798-c1a5-4ef9-9bda-89cfa4e21127" />


## Hardware Architecture

The hardware includes sensors for gas, temperature, and humidity, an ESP32 as the main controller, and output devices for warning and control.

<img width="1157" height="514" alt="image" src="https://github.com/user-attachments/assets/434d3a82-9e5c-4f85-9115-4b5a15b6aeeb" />


## Mobile Application

The mobile application provides real-time monitoring, warning notifications, and remote interaction with the system.

<img width="1597" height="642" alt="image" src="https://github.com/user-attachments/assets/6d82ec12-00be-45f0-beea-c561fe1cdf8b" />


## Results

The Machine Learning module can:

- Analyze sensor trends
- Predict future sensor values
- Identify unusual environmental changes
- Support early warning
<img width="986" height="543" alt="image" src="https://github.com/user-attachments/assets/f0adca0c-c1f9-4fca-8b7c-4771e65a0a95" />
<img width="1047" height="601" alt="image" src="https://github.com/user-attachments/assets/554f5488-c5c2-4b7e-b721-ac6afc0fc2d2" />
<img width="1025" height="590" alt="image" src="https://github.com/user-attachments/assets/1144dd96-91da-4095-851b-2b8655b7bdbf" />


Prediction performance depends on the quality and amount of available training data.

## Technologies

- ESP32
- MQ-series Gas Sensors
- DHT22
- Firebase
- Python
- Random Forest
- Machine Learning
- Arduino IDE

## What I Learned

- ESP32 and IoT development
- Sensor data processing
- Firebase communication
- Time-series preprocessing
- Feature engineering
- Random Forest regression
- Sensor prediction and trend analysis
- ML integration with IoT systems

## Future Improvements

- Collect more sensor data
- Improve feature engineering
- Compare different ML models
- Improve anomaly detection
- Explore edge Machine Learning

## Purpose

This project explores the integration of **Machine Learning into an IoT safety monitoring system** to analyze environmental trends and support **early risk detection**.
