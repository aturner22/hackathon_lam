# Data Handling with WeatherDataset

This document provides an overview of the `WeatherDataset` class in the `neural_lam.weather_dataset` module, detailing its usage, supported data formats, and configuration options.

## Overview

The `WeatherDataset` class is a PyTorch `Dataset` designed to load and preprocess weather simulation data for use in machine learning models. It handles various data formats, subsampling, standardization, and allows for custom preprocessing steps.

Key functionalities include:
- Loading data from multiple file formats: NumPy (`.npy`), GRIB (`.grib`), and NetCDF (`.nc`).
- Discovering samples based on file naming conventions.
- Handling main forecast data as well as auxiliary data like water coverage and Top of Atmosphere (TOA) flux.
- Configurable subsampling of time series data.
- Data standardization using either pre-calculated statistics or user-provided values.
- Support for custom preprocessing pipelines to further transform the data.

## Supported Data Formats

The `WeatherDataset` supports the following data formats for the main forecast variables, water coverage, and TOA flux data:

1.  **NumPy (`.npy`)**:
    *   Binary files storing NumPy arrays.
    *   Assumed to be directly loadable by `numpy.load()` and convertible to PyTorch tensors.

2.  **GRIB (`.grib`)**:
    *   Commonly used format in meteorology for storing gridded weather data.
    *   Requires the `cfgrib` library and its underlying `eccodes` dependency.
    *   The dataset attempts to load GRIB files using `cfgrib.open_datasets()` and convert the first dataset's variables into a single PyTorch tensor using `xarray.Dataset.to_array().data`. This implies that the GRIB file should ideally be structured such that its primary data can be represented as a single multi-dimensional array. For complex GRIB files with many disparate variables, you might need to pre-process them into a more uniform structure or customize the loading logic.

3.  **NetCDF (`.nc`)**:
    *   A self-describing, machine-independent data format for array-oriented scientific data.
    *   Requires the `xarray` and `netCDF4` libraries.
    *   Similar to GRIB, NetCDF files are loaded using `xarray.open_dataset()`, and the data is converted to a PyTorch tensor using `xarray.Dataset.to_array().data`. This works best if the NetCDF file contains one primary data variable or variables that can be meaningfully combined into a single array.

### Naming Conventions and Auxiliary Files

The dataset discovers samples in the specified `sample_dir_path` (e.g., `data/<dataset_name>/samples/<split>`). It expects specific naming conventions:

*   **Main forecast files**: `nwp_<timestamp>_mbr<member_id>.<format_extension>`
    *   Example: `nwp_2023010100_mbr000.npy`
    *   The `<timestamp>` is typically in `YYYYMMDDHH` format.
    *   The loader extracts the base name (e.g., `2023010100_mbr000`) and format.

*   **Water coverage files**: `wtr_<timestamp>.<format_extension>`
    *   Example: `wtr_2023010100.npy`
    *   Must correspond to a main forecast file's timestamp.
    *   Must use the *same file format extension* as the corresponding main forecast file for that sample.

*   **TOA flux files**: `nwp_toa_downwelling_shortwave_flux_<timestamp>.<format_extension>`
    *   Example: `nwp_toa_downwelling_shortwave_flux_2023010100.npy`
    *   Must correspond to a main forecast file's timestamp.
    *   Must use the *same file format extension* as the corresponding main forecast file for that sample.

The dataset expects the internal structure of these files to be compatible with the processing steps (e.g., specific dimensions for time, grid, features). Default expected dimensions for the raw loaded `full_sample` (before most processing) are `(N_t', dim_x, dim_y, d_features')`. Water coverage is `(dim_x, dim_y)` and TOA flux is `(N_t', dim_x, dim_y)`.

## Configuration Options

The `WeatherDataset` is initialized with several arguments to control its behavior:

```python
dataset = WeatherDataset(
    dataset_name="my_dataset",
    pred_length=19, # Number of time steps to predict
    split="train",  # 'train', 'val', or 'test'
    # ... other options ...
)
```

### Standardization

-   **`standardize`**: (bool, default: `True`) Whether to apply standardization.
-   **`data_mean`, `data_std`**: (torch.Tensor, optional) User-provided mean and standard deviation for the main forecast data. Expected to be broadcastable to `(N_grid, d_features)`. If not provided and `standardize` is `True`, these are loaded using `utils.load_dataset_stats(dataset_name, "cpu")`.
-   **`flux_mean`, `flux_std`**: (torch.Tensor, optional) User-provided mean and standard deviation for the TOA flux data. Expected to be broadcastable to `(N_grid, 1)`. If not provided and `standardize` is `True`, these are loaded similarly.

**Example:**
```python
# Using pre-calculated stats from utils
dataset = WeatherDataset(dataset_name="era5_europe", standardize=True)

# Providing custom stats
my_data_mean = torch.load("path/to/data_mean.pt")
my_data_std = torch.load("path/to/data_std.pt")
dataset = WeatherDataset(
    dataset_name="custom_data",
    standardize=True,
    data_mean=my_data_mean,
    data_std=my_data_std,
    # flux_mean and flux_std can also be provided
)
```

### Subsampling

-   **`subsample_step`**: (int, default: 3) The default step for subsampling the time series if `subsample_cfg` is not provided.
-   **`subsample_cfg`**: (dict, optional) A dictionary to configure subsampling behavior.
    -   `'type'`: `'random'` (default for 'train' split) or `'fixed'` (default for 'val'/'test' splits).
    -   `'step'`: The subsampling step (e.g., `3` for taking every 3rd time step). Overrides `subsample_step` if provided here.
    -   `'index'`: The starting index for subsampling when `type` is `'fixed'` (default: `0`).

**Example:**
```python
# Default behavior (random for train, step 3)
dataset_train = WeatherDataset(dataset_name="my_dataset", split="train")

# Fixed subsampling starting at index 0, every 2nd time step
cfg_fixed = {'type': 'fixed', 'index': 0, 'step': 2}
dataset_val = WeatherDataset(dataset_name="my_dataset", split="val", subsample_cfg=cfg_fixed)

# Random subsampling, every 4th time step
cfg_random = {'type': 'random', 'step': 4}
dataset_custom_train = WeatherDataset(dataset_name="my_dataset", split="train", subsample_cfg=cfg_random)
```

### Custom Preprocessing Pipeline

-   **`preprocessing_pipeline`**: (list of callables, optional) A list of functions to be applied sequentially to the data after all standard preprocessing is done. Each function in the pipeline should accept three arguments: `(init_states, target_states, forcing)` and return a tuple of the same structure `(modified_init_states, modified_target_states, modified_forcing)`.

**Example:**
```python
def multiply_by_two(init_states, target_states, forcing):
    return init_states * 2, target_states * 2, forcing * 2

def add_small_noise(init_states, target_states, forcing):
    noise = torch.randn_like(init_states) * 0.01
    return init_states + noise, target_states + noise, forcing # Selectively apply noise

custom_pipeline = [multiply_by_two, add_small_noise]

dataset = WeatherDataset(
    dataset_name="my_dataset",
    preprocessing_pipeline=custom_pipeline
)

# In __getitem__, after normal processing, the data will pass through:
# 1. multiply_by_two
# 2. add_small_noise
```

## Dependencies

To use all features of `WeatherDataset`, particularly for GRIB and NetCDF support, you may need to install additional libraries:

-   **`numpy`**: For handling `.npy` files and general numerical operations.
-   **`torch`**: The core PyTorch library.
-   **`xarray`**: For reading NetCDF files (and used internally by `cfgrib`).
-   **`netCDF4`**: Low-level library for NetCDF, often a dependency of `xarray`.
-   **`cfgrib`**: For reading GRIB files.
    -   This library depends on ECMWF's `eccodes` library. You might need to install `eccodes` separately on your system (e.g., `sudo apt-get install libeccodes-dev` on Debian/Ubuntu) before installing `cfgrib` via pip.

Ensure these are included in your project's `requirements.txt` or Python environment. The `requirements.txt` in this project should already list them.

```
numpy>=1.24.2, <2.0.0
# ... other dependencies ...
cfgrib
netCDF4
xarray
```

## Example Usage

```python
from neural_lam.weather_dataset import WeatherDataset
import torch

# Define a simple augmentation function for the pipeline
def augment_brightness(init_states, target_states, forcing):
    # Example: increase brightness of first feature in target_states
    target_states_aug = target_states.clone()
    target_states_aug[:, :, 0] = target_states_aug[:, :, 0] * 1.1
    return init_states, target_states_aug, forcing

# Configure and instantiate the dataset
dataset_args = {
    "dataset_name": "dummy_dataset", # Use a real dataset name
    "pred_length": 10,
    "split": "train",
    "standardize": True,
    "subsample_cfg": {'type': 'random', 'step': 2},
    "preprocessing_pipeline": [augment_brightness]
}
my_weather_dataset = WeatherDataset(**dataset_args)

# Create a DataLoader
data_loader = torch.utils.data.DataLoader(
    my_weather_dataset,
    batch_size=4,
    shuffle=True
)

# Iterate through data
for batch_idx, (init_states, target_states, forcing) in enumerate(data_loader):
    print(f"Batch {batch_idx+1}")
    print("Init states shape:", init_states.shape)
    print("Target states shape:", target_states.shape)
    print("Forcing shape:", forcing.shape)
    # Your model training/evaluation logic here
    if batch_idx == 0: # Stop after first batch for brevity
        break
```
This provides a comprehensive guide to using the `WeatherDataset`.
