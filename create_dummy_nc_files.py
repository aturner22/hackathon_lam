import numpy as np
import xarray as xr
import os

# Define base path
base_path = "data/dummy_dataset/samples/train"
os.makedirs(base_path, exist_ok=True)

# --- Dummy main sample data (nwp_test02_mbr000.nc) ---
# Shape (time, y, x, var_dim) -> (65, 10, 10, 18) - xarray usually has (time, y, x) or (time, lat, lon)
# Let's use these dimension names for clarity in the NetCDF file.
# The WeatherDataset's .flatten(1,2) will flatten y,x so order shouldn't matter too much there.
nwp_data_np = np.random.rand(65, 10, 10, 18).astype(np.float32)
nwp_data_np[:, :, :, 0] = 4.0 # Identifiable data

nwp_ds = xr.Dataset(
    {"data_var": (("time", "y", "x", "feature_idx"), nwp_data_np)},
    coords={
        "time": np.arange(65),
        "y": np.arange(10),
        "x": np.arange(10),
        "feature_idx": np.arange(18),
    },
)
nwp_ds.to_netcdf(os.path.join(base_path, "nwp_test02_mbr000.nc"))

# --- Dummy water cover data (wtr_test02.nc) ---
# Shape (y, x) -> (10, 10)
water_data_np = np.random.rand(10, 10).astype(np.float32)
water_data_np[:, 0] = 5.0 # Identifiable data

water_ds = xr.Dataset(
    {"water_var": (("y", "x"), water_data_np)},
    coords={"y": np.arange(10), "x": np.arange(10)},
)
water_ds.to_netcdf(os.path.join(base_path, "wtr_test02.nc"))

# --- Dummy TOA flux data (nwp_toa_downwelling_shortwave_flux_test02.nc) ---
# Shape (time, y, x) -> (65, 10, 10)
flux_data_np = np.random.rand(65, 10, 10).astype(np.float32)
flux_data_np[:, :, 0] = 6.0 # Identifiable data

flux_ds = xr.Dataset(
    {"flux_var": (("time", "y", "x"), flux_data_np)},
    coords={"time": np.arange(65), "y": np.arange(10), "x": np.arange(10)},
)
flux_ds.to_netcdf(os.path.join(base_path, "nwp_toa_downwelling_shortwave_flux_test02.nc"))

print("Dummy NC files created.")
