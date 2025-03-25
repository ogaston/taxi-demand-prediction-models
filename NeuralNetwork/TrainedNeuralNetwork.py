import joblib
import numpy as np
from tensorflow import keras

model_storage_location = '../Data/taxi_model.keras'
model = keras.models.load_model(model_storage_location)

scaler_storage_location = '../Data/scaler.pkl'
scaler = joblib.load(scaler_storage_location)

# Predict
sample_location = np.array([['-73.990', '40.750', '2015', '01', '15', '19', '1', '-4.231']])
sample_location_scaled = scaler.transform(sample_location)
predicted_taxis = model.predict(sample_location_scaled)
print(f"Predicted number of taxis: {predicted_taxis[0][0]}")
