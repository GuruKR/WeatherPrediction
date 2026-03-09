# weather_predict_2hr.py
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ---------------- Load Saved Model and Tools ----------------
model = load_model("weather_lstm_model.h5")
scaler = joblib.load("scaler.pkl")
label_enc = joblib.load("label_encoder.pkl")

# ---------------- Function to Predict Weather ----------------
def predict_future_weather(temp, humidity, history_buffer):
    """
    Predicts weather after 2 hours based on temperature, humidity, and past sequence.
    :param temp: current temperature (float)
    :param humidity: current humidity (float)
    :param history_buffer: previous (time_steps - 1) samples of [temp, humidity]
    :return: predicted future weather condition (string)
    """

    # Append current reading to buffer
    history_buffer.append([temp, humidity])

    # Keep only last 5 readings (same as training time_steps)
    if len(history_buffer) > 5:
        history_buffer.pop(0)

    # Ensure we have 5 timesteps
    if len(history_buffer) < 5:
        print("⚠️ Need at least 5 past readings to predict.")
        return None

    # Scale input
    input_seq = scaler.transform(np.array(history_buffer))
    input_seq = np.expand_dims(input_seq, axis=0)  # shape: (1, 5, 2)

    # Predict
    prediction = model.predict(input_seq)
    predicted_label = np.argmax(prediction)
    future_condition = label_enc.inverse_transform([predicted_label])[0]

    return future_condition


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    # Example: last 4 past readings of [temperature, humidity]
    # You can keep updating this list dynamically in real applications.
    history_buffer = [
        [30.2, 62.5],
        [31.0, 61.0],
        [30.8, 60.0],
        [31.5, 59.2]
    ]

    current_temp = float(input("Enter current temperature (°C): "))
    current_humidity = float(input("Enter current humidity (%): "))

    future_weather = predict_future_weather(current_temp, current_humidity, history_buffer)
    
    if future_weather:
        print(f"🌦️ Predicted Weather After 2 Hours: {future_weather}")

        # Example of switching operation
        if future_weather == "Rainy":
            print("🔁 Switching Operation: Turning ON protective cover system.")
        elif future_weather == "Sunny":
            print("🔆 Switching Operation: Activating solar charging system.")
        else:
            print("☁️ Switching Operation: Keeping system in standby mode.")
