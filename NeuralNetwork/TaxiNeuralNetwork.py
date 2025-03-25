import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# Load dataset
df = pd.read_csv("../Data/taxis_dataset.csv")

# Select features and target
input_data = df[['pickup_longitude', 'pickup_latitude', 'year', 'month', 'day', 'hour']]
expected_prediction = df['demand']

# Normalize features
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(input_data_scaled, expected_prediction, test_size=0.2,
                                                    random_state=42)

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

# Save the model to use it later
model_storage_location = '../Data/taxi_model.keras'
model.save(model_storage_location)

# Save the scaler
scaler_storage_location = '../Data/scaler.pkl'
joblib.dump(scaler, scaler_storage_location)

# Graph of prediction vs value
y_pred = model.predict(X_test).flatten()
y_true = y_test.values

plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.title('Comparación entre valores reales y predichos')
plt.xlabel('Demanda real (y_test)')
plt.ylabel('Demanda predicha (y_pred)')

# Línea de referencia y = x para ver la perfección de predicción
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.show()
