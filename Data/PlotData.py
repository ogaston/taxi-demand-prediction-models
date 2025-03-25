import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("../Data/taxis_dataset.csv")

# Compute Data statistics
# Compute statistics
stats = {
    "Latitude": {
        "Min": df["pickup_latitude"].min(),
        "Max": df["pickup_latitude"].max(),
        "Mean": df["pickup_latitude"].mean(),
    },
    "Longitude": {
        "Min": df["pickup_longitude"].min(),
        "Max": df["pickup_longitude"].max(),
        "Mean": df["pickup_longitude"].mean(),
    },
    "Num Trips": {
        "Min": df["demand"].min(),
        "Max": df["demand"].max(),
        "Mean": df["demand"].mean(),
    },
}

count_negative = (df["pickup_latitude"] <= 0).sum()
print(f"Count of pickup_latitude less than 0: {count_negative}")

count_negative = (df["pickup_longitude"] >= 0).sum()
print(f"Count of pickup_longitude more than 0: {count_negative}")

# Print results
for coord, values in stats.items():
    print(f"{coord}:")
    for stat, value in values.items():
        print(f"  {stat}: {value:.6f}")

# Extract relevant columns
latitudes = df["pickup_latitude"]
longitudes = df["pickup_longitude"]
num_trips = df["demand"]

# Create hexbin plot
plt.figure(figsize=(10, 6))
hb = plt.hexbin(longitudes, latitudes, C=num_trips, gridsize=50, cmap="coolwarm", reduce_C_function=np.sum)
plt.colorbar(hb, label="Number of Trips")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Taxi Trip Distribution in NYC")
plt.show()
