# Standard library
import datetime as dt
import glob
import os

# Third-party
import numpy as np
import torch
import xarray as xr # For NetCDF
import cfgrib # For GRIB

# First-party
from neural_lam import constants, utils


class WeatherDataset(torch.utils.data.Dataset):
    """
    For our dataset:
    N_t' = 65
    N_t = 65//subsample_step (= 21 for 3h steps)
    dim_x = 268
    dim_y = 238
    N_grid = 268x238 = 63784
    d_features = 17 (d_features' = 18)
    d_forcing = 5
    """

    def __init__(
        self,
        dataset_name,
        pred_length=19,
        split="train",
        subsample_step=3, # Default, will be overridden by subsample_cfg if provided
        standardize=True,
        subset=False,
        control_only=False,
        data_mean=None,
        data_std=None,
        flux_mean=None,
        flux_std=None,
        subsample_cfg=None,
        preprocessing_pipeline=None,
    ):
        super().__init__()

        assert split in ("train", "val", "test"), "Unknown dataset split"
        self.preprocessing_pipeline = preprocessing_pipeline if preprocessing_pipeline else []
        self.dataset_name = dataset_name
        self.split = split
        self.sample_dir_path = os.path.join(
            "data", dataset_name, "samples", split
        )
        self.control_only = control_only

        # Discover samples
        # self.sample_names now becomes self.samples_info = [(name, format), ...]
        self.samples_info = self._discover_samples()

        if subset:
            self.samples_info = self.samples_info[:50]  # Limit to 50 samples

        self.sample_length = pred_length + 2  # 2 init states

        # Initialize subsampling configuration
        if subsample_cfg:
            self.subsample_cfg = subsample_cfg
            self.subsample_step = subsample_cfg.get('step', subsample_step) # Use provided step or default
        else:
            # Default subsampling behavior
            self.subsample_cfg = {'type': 'random' if split == 'train' else 'fixed',
                                  'step': subsample_step}
            if self.subsample_cfg['type'] == 'fixed':
                self.subsample_cfg['index'] = 0 # Default fixed index
            self.subsample_step = subsample_step

        self.original_sample_length = (
            65 // self.subsample_step
        )  # e.g., 21 for 3h steps if original is 65 timepoints
        assert (
            self.sample_length <= self.original_sample_length
        ), "Requesting too long time series samples"

        # Set up for standardization
        self.standardize = standardize
        if standardize:
            if data_mean is not None and data_std is not None:
                self.data_mean = data_mean
                self.data_std = data_std
            else:
                ds_stats = utils.load_dataset_stats(dataset_name, "cpu")
                self.data_mean = ds_stats["data_mean"]
                self.data_std = ds_stats["data_std"]

            if flux_mean is not None and flux_std is not None:
                self.flux_mean = flux_mean
                self.flux_std = flux_std
            else:
                # Assuming flux stats might also come from ds_stats or be separately handled
                # If not provided and not in ds_stats, this might need adjustment
                ds_stats = utils.load_dataset_stats(dataset_name, "cpu") # Reload if not all provided
                self.flux_mean = ds_stats.get("flux_mean") # Use .get for safety
                self.flux_std = ds_stats.get("flux_std")

        # random_subsample is now part of subsample_cfg
        # self.random_subsample = split == "train" # Old logic

    def __len__(self):
        return len(self.samples_info)

    def _load_npy_data(self, path):
        """Loads data from a .npy file."""
        try:
            return torch.tensor(np.load(path), dtype=torch.float32)
        except Exception as e: # Catching a broader exception for robustness
            print(f"Failed to load NPY {path}: {e}")
            return None

    def _load_grib_data(self, path):
        """Loads data from a GRIB file."""
        try:
            # cfgrib.open_datasets returns a list of xarray.Dataset objects
            # Assuming we need to merge them or select the relevant one.
            # For simplicity, let's assume the first dataset is the one we want
            # and it can be converted to a tensor. This part might need
            # adjustment based on the actual GRIB file structure.
            ds_list = cfgrib.open_datasets(path)
            if not ds_list:
                print(f"No dataset found in GRIB file: {path}")
                return None
            # Assuming data variables in the dataset can be combined into a single tensor.
            # This is a simplification; actual GRIB files may need specific variable selection
            # and merging logic. For now, convert the first dataset to a NumPy array.
            # This might need careful selection of variables.
            # Example: ds_list[0].to_array().values
            # For now, let's assume a primary data variable exists and convert it.
            # This part is highly dependent on GRIB structure and might need specific keys.
            # For now, a placeholder for conversion:
            # data_array = ds_list[0][DATA_VARIABLE_NAME].data
            # This needs to be adapted to the actual data structure.
            # As a placeholder, attempting to convert the whole dataset to an array
            data_array = ds_list[0].to_array().data
            return torch.tensor(data_array, dtype=torch.float32)
        except Exception as e:
            print(f"Failed to load GRIB {path}: {e}")
            return None

    def _load_netcdf_data(self, path):
        """Loads data from a NetCDF file."""
        try:
            # Similar to GRIB, NetCDF files can have complex structures.
            # Using xarray to open the dataset.
            ds = xr.open_dataset(path)
            # Again, assuming a primary data variable or a way to convert to a single array.
            # This is a simplification.
            # Example: ds[DATA_VARIABLE_NAME].data
            # As a placeholder, attempting to convert the whole dataset to an array
            data_array = ds.to_array().data
            return torch.tensor(data_array, dtype=torch.float32)
        except Exception as e:
            print(f"Failed to load NetCDF {path}: {e}")
            return None

    def _discover_samples(self):
        """Discovers samples of supported formats (.npy, .grib, .nc) in the dataset directory."""
        samples_info = []
        for fmt_ext, member_pattern_suffix in [
            ("npy", ".npy"),
            ("grib", ".grib"),
            ("nc", ".nc")
        ]:
            member_file_regexp = (
                f"nwp*mbr000{member_pattern_suffix}" if self.control_only else f"nwp*mbr*{member_pattern_suffix}"
            )
            sample_paths = glob.glob(
                os.path.join(self.sample_dir_path, member_file_regexp)
            )
            for path in sample_paths:
                # Extract name like "yyymmddhh_mbrXXX"
                name_with_ext = path.split("/")[-1]
                # Remove "nwp_" prefix and extension
                name = name_with_ext[4 : -len(fmt_ext)-1]
                samples_info.append({"name": name, "format": fmt_ext, "prefix": name_with_ext[:4]}) # Store prefix like "nwp_"
        return samples_info

    def __getitem__(self, idx):
        # === Sample ===
        sample_info = self.samples_info[idx]
        sample_name = sample_info["name"]
        sample_format = sample_info["format"]
        file_prefix = sample_info["prefix"] # e.g. "nwp_"

        sample_filename = f"{file_prefix}{sample_name}.{sample_format}"
        sample_path = os.path.join(self.sample_dir_path, sample_filename)

        loader_map = {
            "npy": self._load_npy_data,
            "grib": self._load_grib_data,
            "nc": self._load_netcdf_data,
        }

        load_func = loader_map.get(sample_format)
        if not load_func:
            print(f"Unsupported file format: {sample_format} for {sample_path}")
            return None # Or raise error

        full_sample = load_func(sample_path)
        if full_sample is None:
            # Error message already printed by loader
            return None

        # Only use every ss_step:th time step, sample which of ss_step
        # possible such time series
        subsample_type = self.subsample_cfg.get('type', 'random' if self.split == 'train' else 'fixed')
        if subsample_type == 'random':
            subsample_index = torch.randint(0, self.subsample_step, ()).item()
        elif subsample_type == 'fixed':
            subsample_index = self.subsample_cfg.get('index', 0)
        else: # Default or error
            subsample_index = 0
            print(f"Warning: Unknown subsample type {subsample_type}. Defaulting to fixed index 0.")

        subsample_end_index = self.original_sample_length * self.subsample_step
        sample = full_sample[
            subsample_index : subsample_end_index : self.subsample_step
        ]
        # (N_t, dim_x, dim_y, d_features')

        # Remove feature 15, "z_height_above_ground"
        sample = torch.cat(
            (sample[:, :, :, :15], sample[:, :, :, 16:]), dim=3
        )  # (N_t, dim_x, dim_y, d_features)

        # Accumulate solar radiation instead of just subsampling
        rad_features = full_sample[:, :, :, 2:4]  # (N_t', dim_x, dim_y, 2)
        # Accumulate for first time step
        init_accum_rad = torch.sum(
            rad_features[: (subsample_index + 1)], dim=0, keepdim=True
        )  # (1, dim_x, dim_y, 2)
        # Accumulate for rest of subsampled sequence
        in_subsample_len = (
            subsample_end_index - self.subsample_step + subsample_index + 1
        )
        rad_features_in_subsample = rad_features[
            (subsample_index + 1) : in_subsample_len
        ]  # (N_t*, dim_x, dim_y, 2), N_t* = (N_t-1)*ss_step
        _, dim_x, dim_y, _ = sample.shape
        rest_accum_rad = torch.sum(
            rad_features_in_subsample.view(
                self.original_sample_length - 1,
                self.subsample_step,
                dim_x,
                dim_y,
                2,
            ),
            dim=1,
        )  # (N_t-1, dim_x, dim_y, 2)
        accum_rad = torch.cat(
            (init_accum_rad, rest_accum_rad), dim=0
        )  # (N_t, dim_x, dim_y, 2)
        # Replace in sample
        sample[:, :, :, 2:4] = accum_rad

        # Flatten spatial dim
        sample = sample.flatten(1, 2)  # (N_t, N_grid, d_features)

        # Uniformly sample time id to start sample from
        init_id = torch.randint(
            0, 1 + self.original_sample_length - self.sample_length, ()
        )
        sample = sample[init_id : (init_id + self.sample_length)]
        # (sample_length, N_grid, d_features)

        if self.standardize:
            # Standardize sample
            sample = (sample - self.data_mean) / self.data_std

        # Split up sample in init. states and target states
        init_states = sample[:2]  # (2, N_grid, d_features)
        target_states = sample[2:]  # (sample_length-2, N_grid, d_features)

        # === Forcing features ===
        # Now batch-static features are just part of forcing,
        # repeated over temporal dimension
        # Load water coverage
        # Assuming water/flux files follow the same format and naming convention as the main sample
        sample_datetime = sample_name[:10] # "yyyymmddhh"

        water_filename = f"wtr_{sample_datetime}.{sample_format}"
        water_path = os.path.join(self.sample_dir_path, water_filename)
        water_cover_features = load_func(water_path) # Use the same loader as the main sample

        if water_cover_features is None:
            # Error message already printed by loader
            return None
        water_cover_features = water_cover_features.unsqueeze(-1) # (dim_x, dim_y, 1)
        # Flatten
        water_cover_features = water_cover_features.flatten(0, 1)  # (N_grid, 1)
        # Expand over temporal dimension
        water_cover_expanded = water_cover_features.unsqueeze(0).expand(
            self.sample_length - 2, -1, -1  # -2 as added on after windowing
        )  # (sample_len, N_grid, 1)

        # TOA flux
        # Assuming flux files follow the same format and naming convention
        flux_filename = f"nwp_toa_downwelling_shortwave_flux_{sample_datetime}.{sample_format}"
        flux_path = os.path.join(
            self.sample_dir_path,
            flux_filename,
        )
        flux = load_func(flux_path) # Use the same loader as the main sample
        if flux is None:
            # Error message already printed by loader
            return None
        flux = flux.unsqueeze(-1)  # (N_t', dim_x, dim_y, 1)

        if self.standardize:
            flux = (flux - self.flux_mean) / self.flux_std

        # Flatten and subsample flux forcing
        flux = flux.flatten(1, 2)  # (N_t, N_grid, 1)
        flux = flux[subsample_index :: self.subsample_step]  # (N_t, N_grid, 1)
        flux = flux[
            init_id : (init_id + self.sample_length)
        ]  # (sample_len, N_grid, 1)

        # Time of day and year
        dt_obj = dt.datetime.strptime(sample_datetime, "%Y%m%d%H")
        dt_obj = dt_obj + dt.timedelta(
            hours=2 + subsample_index
        )  # Offset for first index
        # Extract for initial step
        init_hour_in_day = dt_obj.hour
        start_of_year = dt.datetime(dt_obj.year, 1, 1)
        init_seconds_into_year = (dt_obj - start_of_year).total_seconds()

        # Add increments for all steps
        hour_inc = (
            torch.arange(self.sample_length) * self.subsample_step
        )  # (sample_len,)
        hour_of_day = (
            init_hour_in_day + hour_inc
        )  # (sample_len,), Can be > 24 but ok
        second_into_year = (
            init_seconds_into_year + hour_inc * 3600
        )  # (sample_len,)
        # can roll over to next year, ok because periodicity

        # Encode as sin/cos
        hour_angle = (hour_of_day / 12) * torch.pi  # (sample_len,)
        year_angle = (
            (second_into_year / constants.SECONDS_IN_YEAR) * 2 * torch.pi
        )  # (sample_len,)
        datetime_forcing = torch.stack(
            (
                torch.sin(hour_angle),
                torch.cos(hour_angle),
                torch.sin(year_angle),
                torch.cos(year_angle),
            ),
            dim=1,
        )  # (N_t, 4)
        datetime_forcing = (datetime_forcing + 1) / 2  # Rescale to [0,1]
        datetime_forcing = datetime_forcing.unsqueeze(1).expand(
            -1, flux.shape[1], -1
        )  # (sample_len, N_grid, 4)

        # Put forcing features together
        forcing_features = torch.cat(
            (flux, datetime_forcing), dim=-1
        )  # (sample_len, N_grid, d_forcing)

        # Combine forcing over each window of 3 time steps
        forcing_windowed = torch.cat(
            (
                forcing_features[:-2],
                forcing_features[1:-1],
                forcing_features[2:],
            ),
            dim=2,
        )  # (sample_len-2, N_grid, 3*d_forcing)
        # Now index 0 of ^ corresponds to forcing at index 0-2 of sample

        # batch-static water cover is added after windowing,
        # as it is static over time
        forcing = torch.cat((water_cover_expanded, forcing_windowed), dim=2)
        # (sample_len-2, N_grid, forcing_dim)

        # Apply custom preprocessing pipeline
        for func in self.preprocessing_pipeline:
            # Functions in the pipeline are expected to take
            # (init_states, target_states, forcing) and return the modified tuple
            try:
                init_states, target_states, forcing = func(init_states, target_states, forcing)
            except Exception as e:
                print(f"Error during custom preprocessing function {func.__name__}: {e}")
                # Depending on desired robustness, either skip this function or re-raise

        return init_states, target_states, forcing
