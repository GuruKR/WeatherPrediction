

# 🌤️ WeatherAI – AI Powered Weather Prediction System

WeatherAI is a **machine learning powered web application** that predicts **weather conditions 2 hours ahead** using an **LSTM (Long Short-Term Memory) deep learning model**.

The system analyzes **voltage, current, temperature, and humidity data** to forecast weather conditions and automatically trigger **system actions such as solar charging**.

---

# 🚀 Features

✅ **LSTM Deep Learning Model**
Predicts upcoming weather conditions based on time-series environmental data.

✅ **2-Hour Ahead Forecasting**
Uses sequential historical inputs to estimate near-future weather conditions.

✅ **Real-Time Prediction Interface**
Users can input current sensor readings and instantly receive predictions.

✅ **Automated System Response**
Based on predictions, the system can trigger actions such as:

* Solar charging activation
* Energy management decisions

✅ **Prediction Dashboard**

The dashboard provides:

* Total predictions
* Sunny days
* Rainy days
* Cloudy days
* Prediction statistics graph

✅ **Historical Buffer System**

Stores recent weather inputs including:

* Temperature
* Humidity
* Sequential buffer inputs for LSTM

✅ **Battery Simulation**

Demonstrates system energy management states:

* Charging
* Discharging
* Full
* Low

---

# 🧠 AI Model

The system uses an **LSTM (Long Short-Term Memory) neural network**, which is ideal for **time-series prediction problems**.

### Model Inputs

The model processes the following environmental parameters:

* Voltage
* Current
* Temperature
* Humidity

### Sequence Processing

The LSTM model processes a **5-step sequence buffer** of previous readings to predict weather conditions.

### Model Output

The model predicts one of the following:

* ☀️ **Sunny**
* 🌧️ **Rainy**
* ☁️ **Cloudy**

---

# 🖥️ System Workflow

```
Sensor Data Input
       │
       ▼
Historical Buffer (5-step sequence)
       │
       ▼
LSTM Model Prediction
       │
       ▼
Weather Forecast (2 hours ahead)
       │
       ▼
Automation Action Trigger
```

---

# 📊 Dashboard

The dashboard includes:

* Weather prediction statistics
* Prediction history
* Graph visualization
* Quick action controls

Users can:

* Filter predictions by date
* View prediction trends
* Monitor system behavior

---

# 📦 Project Structure

Example project structure:

```
WeatherAI/
│
├── app.py
├── model/
│   └── lstm_model.h5
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   └── predict.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── dataset/
│   └── weather_data.csv
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/weather-ai.git
cd weather-ai
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Mac**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the application

```bash
python app.py
```

---

### 5️⃣ Open in browser

```
http://127.0.0.1:5000
```

---

# 📈 Example Prediction

Input values:

```
Voltage: 5.84
Current: 2.92
Temperature: 24.9°C
Humidity: 53.5%
```

Predicted result:

```
Weather: Sunny
Forecast: 2 hours ahead
Automation: Activating solar charging system
```

---

# 🛠️ Technologies Used

### Backend

* Python
* Flask

### Machine Learning

* TensorFlow / Keras
* LSTM Neural Network
* NumPy
* Pandas
* Scikit-learn

### Frontend

* HTML
* CSS
* JavaScript
* Bootstrap

### Visualization

* Chart.js

---

# 🔋 Automation Logic

The system can automatically trigger actions based on predictions.

Example:

| Weather | Action                  |
| ------- | ----------------------- |
| Sunny   | Activate Solar Charging |
| Rainy   | Reduce solar dependency |
| Cloudy  | Optimize energy storage |

---



# 👨‍💻 Author

Developed by **Guru**

AI-based smart energy and weather prediction system.

Screen Shots

