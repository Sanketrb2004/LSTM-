import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv('tilapia growth dataset - 1.csv')

print("Dataset loaded successfully")
print(df.head())

# =========================
# 2. PREPROCESSING
# =========================
df.columns = df.columns.str.strip()
df = df.ffill()

print("\nColumns:", df.columns)

df = df.sort_values(by=['Year', 'Month', 'Timestamp'])

# =========================
# 3. FEATURE SELECTION
# =========================
features = [
    'Temperature_C',
    'Dissolved_oxygen_mg_L',
    'pH',
    'Turbidity (NTU)'
]

target = 'Fish_length'
data = df[features + [target]]

# =========================
# 4. NORMALIZATION
# =========================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

print("Data normalized")

# =========================
# 5. CREATE SEQUENCES
# =========================
def create_sequences(data, seq_length=96):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :-1])
        y.append(data[i+seq_length, -1])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)

print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================
# 6. TRAIN-TEST SPLIT
# =========================
split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =========================
# 7. BUILD MODEL
# =========================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

print(model.summary())

# =========================
# 8. EARLY STOPPING
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# =========================
# 9. TRAIN MODEL
# =========================
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# =========================
# 10. LOSS GRAPH
# =========================
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()

# =========================
# 11. PREDICTIONS
# =========================
predictions = model.predict(X_test)

# =========================
# 12. INVERSE SCALING
# =========================
dummy = np.zeros((len(predictions), len(features)+1))

dummy[:, -1] = predictions[:, 0]
predicted_length = scaler.inverse_transform(dummy)[:, -1]

dummy[:, -1] = y_test
actual_length = scaler.inverse_transform(dummy)[:, -1]

# =========================
# 13. MAE (FIXED POSITION)
# =========================
mae = mean_absolute_error(actual_length, predicted_length)
print("\nMean Absolute Error:", mae)

# =========================
# 14. PLOT RESULTS
# =========================
plt.figure(figsize=(10,5))
plt.plot(actual_length[:100], label='Actual')
plt.plot(predicted_length[:100], label='Predicted')
plt.legend()
plt.title("Actual vs Predicted Fish Length")
plt.show()

# =========================
# 15. GROWTH ANALYSIS
# =========================
print("\n--- Growth Analysis ---")

if predicted_length[-1] > actual_length[-1]:
    print("Fish growth is improving 📈")
elif predicted_length[-1] < actual_length[-1]:
    print("Fish growth is declining ⚠️ Check water conditions")
else:
    print("Fish growth is stable ✅")

print(f"Predicted Length: {predicted_length[-1]:.2f}")
print(f"Actual Length: {actual_length[-1]:.2f}")

# =========================
# 16. SAVE MODEL
# =========================
model.save("fish_growth_model.h5")

print("\nModel saved successfully!")