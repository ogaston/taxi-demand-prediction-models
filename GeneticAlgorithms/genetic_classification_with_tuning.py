import pandas as pd
import numpy as np
import pygad
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import itertools
import time

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
def fitness_func(model, solution, solution_idx):
    predictions = np.dot(X, solution[1:]) + solution[0]
    rmse = sqrt(mean_squared_error(y, predictions))
    return -rmse

# Hyperparameter grid
num_generations_list = [50, 100, 150]
num_parents_mating_list = [5, 10, 15]
mutation_percent_genes_list = [10, 20, 30]

# Generate combinations
param_grid = list(itertools.product(num_generations_list, num_parents_mating_list, mutation_percent_genes_list))

# Run experiments
results = []

for num_generations, num_parents_mating, mutation_percent_genes in param_grid:
    print(f"Testing: Generations={num_generations}, Parents={num_parents_mating}, Mutation={mutation_percent_genes}%")
    start_time = time.time()

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=20,
        num_genes=len(features) + 1,
        gene_space={'low': -10.0, 'high': 10.0},
        mutation_percent_genes=mutation_percent_genes,
        mutation_type="random",
        crossover_type="single_point",
        random_mutation_min_val=-1.0,
        random_mutation_max_val=1.0,
        stop_criteria=["saturate_10"]
    )

    ga_instance.run()
    solution, fitness, _ = ga_instance.best_solution()
    duration = time.time() - start_time

    results.append({
        "num_generations": num_generations,
        "num_parents_mating": num_parents_mating,
        "mutation_percent_genes": mutation_percent_genes,
        "fitness": fitness,
        "rmse": -fitness,
        "time_sec": duration
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)
results_df.sort_values(by="rmse", inplace=True)

# Save to CSV
results_df.to_csv("ga_tuning_results.csv", index=False)

# Display as table
print("\nAll Results Sorted by RMSE:")
print(results_df)

# Plot fitness trends
plt.figure(figsize=(12, 6))
plt.plot(range(len(results_df)), results_df["rmse"], marker='o')
plt.xticks(range(len(results_df)), results_df.index, rotation=45)
plt.title("RMSE by Hyperparameter Combination")
plt.xlabel("Experiment Index")
plt.ylabel("RMSE (Lower is Better)")
plt.grid(True)
plt.tight_layout()
plt.savefig("fitness_trend.png")
plt.show()

# Best configuration
best = results_df.iloc[0]
print("\nBest Configuration:")
print("Params:", best[["num_generations", "num_parents_mating", "mutation_percent_genes"]].to_dict())
print("Best RMSE:", best["rmse"])
print("Training Time (sec):", best["time_sec"])

# Print everything to check the results
print("\nSorted results:")
print("Results ", results)