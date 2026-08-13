# Fire-and-Gas-Leakage-Monitoring-and-Warning-System

An IoT-based fire and gas leakage monitoring system developed for real-time environmental monitoring and early risk detection.

The system uses an **ESP32**, MQ-series gas sensors, and DHT22 to collect environmental data, control warning devices, and transmit data to Firebase for remote monitoring.

The main focus of this project is to **integrate Machine Learning into the existing IoT system** to analyze sensor trends and predict future environmental conditions, providing additional support for early risk detection.

## System Architecture

The system is organized into four main layers: perception, network, processing, and application.

![System Architecture](images/system-architecture.png)

At the device level, sensors collect environmental data and send it to the ESP32. The ESP32 processes the data and communicates with Firebase through Wi-Fi. Firebase acts as the central data platform connecting the embedded system, processing system, and mobile application.

## Hardware Architecture

The hardware consists of three main blocks: input, processing, and output.

![Hardware Architecture](images/hardware-architecture.png)

The input block includes the power supply, MQ-series gas sensor, and DHT22 temperature and humidity sensor. The ESP32 is used as the main processing unit. The output block includes the warning buzzer and light, exhaust fan, pump, and TFT display.

## Mobile Application

The mobile application provides real-time monitoring and remote interaction with the system.

![Mobile Application](images/mobile-application.png)

The application allows users to monitor environmental parameters, view system status, receive warning notifications, and interact with the system remotely.

## Machine Learning Integration

Machine Learning is introduced as an additional analysis layer on top of the existing IoT monitoring system.

Instead of relying only on instantaneous sensor values and predefined thresholds, historical and real-time sensor data are processed to analyze environmental trends and predict future sensor values.

```text
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

Historical sensor data collection
Data preprocessing
Time normalization
Noise filtering
Time-based resampling
Time-series feature extraction
Model training and testing
Sensor value prediction
Trend analysis
Machine Learning Model

A Random Forest model is used for sensor value prediction.

Historical sensor data is processed before training by filtering and resampling the data and extracting time-series features. The dataset is then divided into training and testing sets.

The model is evaluated using Mean Absolute Error (MAE) before being used for prediction.

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
Prediction
Sensor Data Analysis

The collected environmental data is analyzed over different time periods to identify changes and trends.

Temperature

The temperature data is visualized using statistical values such as median, minimum, maximum, and distribution ranges over time.

Humidity

Humidity data is analyzed in a similar way to observe environmental variations and support the prediction process.

Prediction and Early Warning

During operation, the Machine Learning module retrieves historical and real-time sensor data from Firebase.

The data is validated and transformed into the required features before being processed by the trained Random Forest model. The predicted values and trends are then synchronized back to Firebase for monitoring through the application.

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

The Machine Learning module provides an additional layer of analysis by identifying increasing or decreasing environmental trends and predicting upcoming sensor values.

The ML model is designed as a supporting analysis layer, while the main real-time monitoring and safety control mechanisms remain independent of the prediction model.

Results

The Machine Learning module can be used to:

Analyze environmental trends
Predict upcoming sensor values
Identify unusual changes in sensor data
Support early warning
Provide additional information for environmental monitoring

The results show that Machine Learning can support the identification of potentially abnormal environmental trends. However, prediction performance is still influenced by the amount and diversity of available training data.

Technologies
ESP32
MQ-series Gas Sensor
DHT22
Firebase
Python
Random Forest
Machine Learning
Arduino IDE
Git / GitHub
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
Collect a larger and more diverse sensor dataset
Improve feature engineering
Compare Random Forest with other Machine Learning models
Improve anomaly detection
Evaluate the model using additional metrics
Improve prediction accuracy
Explore lightweight Machine Learning models for edge devices
Improve the integration between ML predictions and early-warning mechanisms
Project Purpose

This project was developed as a learning and research project to explore how Machine Learning can be integrated into an existing IoT safety monitoring system.

The main objective is to move beyond simple threshold-based monitoring by using historical and real-time sensor data to analyze environmental trends and support early risk detection.

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
