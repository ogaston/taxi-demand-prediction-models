import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("../Data/taxis_dataset.csv")

# Select features and target
input_data = df[['pickup_longitude', 'pickup_latitude', 'year', 'month', 'day', 'hour']]
expected_prediction = df['demand']

# Normalize features
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(input_data_scaled, expected_prediction, test_size=0.2, random_state=42)

# Build Neural Network
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)  # Regression output
])

# Compile Model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train Model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate Model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae}")

# Predict
sample_location = np.array([['-73.990', '40.750', '2015', '01', '15', '19']])
sample_location_scaled = scaler.transform(sample_location)
predicted_taxis = model.predict(sample_location_scaled)
print(f"Predicted number of taxis: {predicted_taxis[0][0]}")
