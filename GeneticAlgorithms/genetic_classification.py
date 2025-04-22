import pandas as pd
import numpy as np
import pygad
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('../Data/taxis_dataset_filtered.csv')  # Replace with your actual dataset path

# Preprocess
df['is_holiday'] = df['is_holiday'].astype(int)
df['lat_bin'] = (df['pickup_latitude'] * 100).astype(int)
df['lon_bin'] = (df['pickup_longitude'] * 100).astype(int)

# Features and target
features = ['hour', 'is_holiday', 'temperature', 'pickup_latitude', 'pickup_longitude']
X = df[features].values
y = df['demand'].values

# Fitness function
def fitness_func(ga_instance, solution, solution_idx):
    predictions = np.dot(X, solution[1:]) + solution[0]
    rmse = sqrt(mean_squared_error(y, predictions))
    return -rmse

# Genetic Algorithm setup
ga_instance = pygad.GA(
    num_generations=100,
    num_parents_mating=10,
    fitness_func=fitness_func,
    sol_per_pop=20,
    num_genes=len(features) + 1,
    gene_space={'low': -10.0, 'high': 10.0},
    mutation_percent_genes=20,
    mutation_type="random",
    crossover_type="single_point",
    random_mutation_min_val=-1.0,
    random_mutation_max_val=1.0,
    stop_criteria=["saturate_10"]
)

# Run GA
ga_instance.run()
solution, _, _ = ga_instance.best_solution()

# Predict demand
df['predicted_demand'] = np.dot(X, solution[1:]) + solution[0]

# Heatmap
heatmap_data = df.groupby(['lat_bin', 'lon_bin'])['predicted_demand'].mean().unstack().fillna(0)
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, cmap='YlOrRd')
plt.title("Predicted Demand Heatmap by Grid")
plt.xlabel("Longitude Bin")
plt.ylabel("Latitude Bin")
plt.tight_layout()
plt.savefig("predicted_demand_heatmap.png")
plt.show()