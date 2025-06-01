import numpy as np
import os

# Define base path
base_path = "data/dummy_dataset/samples/train"
os.makedirs(base_path, exist_ok=True)

# Dummy main sample data (nwp_test01_mbr000.npy)
# Shape (N_t', dim_x, dim_y, d_features') -> (65, 10, 10, 18)
nwp_data = np.random.rand(65, 10, 10, 18).astype(np.float32)
# Make one feature identifiable for testing
nwp_data[:, :, :, 0] = 1.0
np.save(os.path.join(base_path, "nwp_test01_mbr000.npy"), nwp_data)

# Dummy water cover data (wtr_test01.npy)
# Shape (dim_x, dim_y) -> (10, 10)
water_data = np.random.rand(10, 10).astype(np.float32)
water_data[:, 0] = 2.0 # Make identifiable
np.save(os.path.join(base_path, "wtr_test01.npy"), water_data)

# Dummy TOA flux data (nwp_toa_downwelling_shortwave_flux_test01.npy)
# Shape (N_t', dim_x, dim_y) -> (65, 10, 10)
flux_data = np.random.rand(65, 10, 10).astype(np.float32)
flux_data[:, :, 0] = 3.0 # Make identifiable
np.save(os.path.join(base_path, "nwp_toa_downwelling_shortwave_flux_test01.npy"), flux_data)

print("Dummy NPY files created.")
