import numpy as np
import json
import os

# Define base path
base_path = "data/dummy_dataset/stats"
os.makedirs(base_path, exist_ok=True)

# d_features = 17 (after removing one original feature)
# N_grid = 10 * 10 = 100
d_features = 17
n_grid = 100

# Data stats (per-feature, assuming they will be broadcast or tiled to N_grid by the dataset/loader if needed)
# Or, if the loader expects it to be (N_grid, d_features), we should make it so.
# The WeatherDataset standardizes sample as (sample - self.data_mean) / self.data_std
# where sample is (sample_length, N_grid, d_features).
# So, data_mean and data_std should be (N_grid, d_features) or broadcastable.
# For simplicity, let's make them (1, d_features) to be broadcastable.
data_mean = np.random.rand(1, d_features).astype(np.float32).tolist()
data_std = (np.random.rand(1, d_features) + 0.1).astype(np.float32).tolist() # ensure std > 0

# Flux stats: flux is (sample_len, N_grid, 1)
# So, flux_mean, flux_std should be (N_grid, 1) or broadcastable.
# Let's make them (1,1) for simplicity.
flux_mean = [np.random.rand(1).astype(np.float32).item()] # scalar in a list
flux_std = [(np.random.rand(1) + 0.1).astype(np.float32).item()] # scalar in a list, >0

stats = {
    "data_mean": data_mean,
    "data_std": data_std,
    "flux_mean": flux_mean,
    "flux_std": flux_std,
}

with open(os.path.join(base_path, "dataset_stats.json"), "w") as f:
    json.dump(stats, f, indent=4)

print("Dummy stats JSON file created.")
