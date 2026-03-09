# weather_lstm_train.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib

# ---------------- Load Dataset ----------------
df = pd.read_csv("data.csv")

# Encode categorical labels
label_enc = LabelEncoder()
df['future_condition_after_2hr'] = label_enc.fit_transform(df['future_condition_after_2hr'])

# Save label encoder for later use
joblib.dump(label_enc, "label_encoder.pkl")

# Select features
features = df[['current_temperature', 'current_humidity']]
targets = df['future_condition_after_2hr']

# Normalize features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# ---------------- Create Sequences for LSTM ----------------
def create_sequences(X, y, time_steps=5):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 5  # Using past 5 timesteps to predict next condition
X, y = create_sequences(features_scaled, targets, time_steps)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# One-hot encode labels
num_classes = len(label_enc.classes_)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# ---------------- Build LSTM Model ----------------
model = Sequential([
    InputLayer(input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------------- Train Model ----------------
checkpoint = ModelCheckpoint('weather_lstm_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=30,
    batch_size=32,
    callbacks=[checkpoint],
    verbose=1
)

# ---------------- Plot Training Performance ----------------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("training_performance.png")
plt.show()

print("✅ Model training completed and saved as 'weather_lstm_model.h5'")
print("✅ Scaler and label encoder saved as 'scaler.pkl' and 'label_encoder.pkl'")
print("✅ Training performance graph saved as 'training_performance.png'")
