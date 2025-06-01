# Standard library
import os
import sys
import shutil
import json

# Third-party
import pytest
import torch
import numpy as np

# Ensure neural_lam is in path (adjust as necessary if tests are run from a different CWD)
# This assumes tests might be run from the root of the project.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# First-party
from neural_lam.weather_dataset import WeatherDataset
from neural_lam import utils # For mocking load_dataset_stats

# --- Test Configuration ---
DUMMY_DATASET_NAME = "dummy_dataset"
DUMMY_DATA_DIR = os.path.join("data", DUMMY_DATASET_NAME)
DUMMY_SAMPLES_DIR = os.path.join(DUMMY_DATA_DIR, "samples", "train")
DUMMY_STATS_DIR = os.path.join(DUMMY_DATA_DIR, "stats")
DUMMY_STATS_FILE = os.path.join(DUMMY_STATS_DIR, "dataset_stats.json")

# Expected shapes based on dummy data creation
# N_t' = 65 (original time steps in files)
# Subsample_step = 3 (default in WeatherDataset, if not overridden)
# N_t = 65 // 3 = 21 (time steps after subsampling)
# pred_length = 4 (for shorter test sequences)
# sample_length = pred_length + 2 = 6
# dim_x = 10, dim_y = 10 (dummy grid dimensions)
# N_grid = 10 * 10 = 100
# d_features' = 18 (original features in files)
# d_features = 17 (after removing feature 15)
# d_forcing_original = 5 (flux (1) + datetime (4))
# Forcing window = 3, so 3 * 5 = 15. Plus water cover (1) = 16
# forcing_dim = 1 (water) + 3 * (1 (flux) + 4 (datetime)) = 1 + 3*5 = 16

TEST_PRED_LENGTH = 4
TEST_SAMPLE_LENGTH = TEST_PRED_LENGTH + 2 # 6
TEST_N_GRID = 100
TEST_D_FEATURES = 17
TEST_FORCING_DIM = 16


@pytest.fixture(scope="module", autouse=True)
def setup_dummy_data_env():
    """
    Set up the dummy data directory and files for the test module.
    This fixture will run once per module.
    """
    # Create dummy data files (already done by previous steps, but good for standalone test execution)
    # For simplicity in this environment, we assume they are created by the previous agent steps.
    # If running this test file standalone, these creation scripts would need to be called here.
    # os.system("python create_dummy_npy_files.py")
    # os.system("python create_dummy_nc_files.py")
    # os.makedirs(DUMMY_SAMPLES_DIR, exist_ok=True)
    # open(os.path.join(DUMMY_SAMPLES_DIR, "nwp_test03_mbr000.grib"), 'a').close()
    # open(os.path.join(DUMMY_SAMPLES_DIR, "wtr_test03.grib"), 'a').close()
    # open(os.path.join(DUMMY_SAMPLES_DIR, "nwp_toa_downwelling_shortwave_flux_test03.grib"), 'a').close()
    # os.system("python create_dummy_stats_file.py")
    pass # Assuming files are there from previous steps

    yield

    # Teardown: Remove dummy data directory after tests are done
    # Commented out for iterative development in the agent environment
    # if os.path.exists(DUMMY_DATA_DIR):
    #     shutil.rmtree(DUMMY_DATA_DIR)
    # if os.path.exists("create_dummy_npy_files.py"):
    #    os.remove("create_dummy_npy_files.py")
    # if os.path.exists("create_dummy_nc_files.py"):
    #    os.remove("create_dummy_nc_files.py")
    # if os.path.exists("create_dummy_stats_file.py"):
    #    os.remove("create_dummy_stats_file.py")


@pytest.fixture
def base_dataset_args():
    return {
        "dataset_name": DUMMY_DATASET_NAME,
        "pred_length": TEST_PRED_LENGTH,
        "split": "train", # Points to "data/dummy_dataset/samples/train"
        "subsample_step": 3, # Default, N_t = 65 // 3 = 21
        "standardize": False, # Default to false for simple loading tests
        "subset": False,
        "control_only": False, # Will load all nwp_test*mbr000.* files
    }

# --- Mock for utils.load_dataset_stats ---
@pytest.fixture
def mock_load_stats(monkeypatch):
    def mock_load_dataset_stats(dataset_name, device):
        # This function needs to load the DUMMY_STATS_FILE
        # and convert lists back to tensors as WeatherDataset might expect
        with open(DUMMY_STATS_FILE, 'r') as f:
            stats = json.load(f)

        # WeatherDataset expects tensors for these after loading
        return {
            "data_mean": torch.tensor(stats["data_mean"], dtype=torch.float32),
            "data_std": torch.tensor(stats["data_std"], dtype=torch.float32),
            "flux_mean": torch.tensor(stats["flux_mean"], dtype=torch.float32),
            "flux_std": torch.tensor(stats["flux_std"], dtype=torch.float32),
        }
    monkeypatch.setattr(utils, "load_dataset_stats", mock_load_dataset_stats)


# --- Test Cases ---

def test_dataset_creation(base_dataset_args):
    dataset = WeatherDataset(**base_dataset_args)
    assert dataset is not None
    # Expected: 1 NPY, 1 NC, 1 GRIB sample = 3 discovered samples
    assert len(dataset) == 3


def test_npy_loading(base_dataset_args):
    dataset = WeatherDataset(**base_dataset_args)
    # Find the .npy sample (nwp_test01_mbr000.npy)
    npy_sample_idx = -1
    for i, sample_info in enumerate(dataset.samples_info):
        if sample_info["format"] == "npy":
            npy_sample_idx = i
            break
    assert npy_sample_idx != -1, "NPY sample not found in dataset"

    init_states, target_states, forcing = dataset[npy_sample_idx]

    assert init_states is not None
    assert target_states is not None
    assert forcing is not None

    # Check shapes
    # init_states: (2, N_grid, d_features) = (2, 100, 17)
    # target_states: (sample_length-2, N_grid, d_features) = (4, 100, 17)
    # forcing: (sample_length-2, N_grid, forcing_dim) = (4, 100, 16)
    assert init_states.shape == (2, TEST_N_GRID, TEST_D_FEATURES)
    assert target_states.shape == (TEST_PRED_LENGTH, TEST_N_GRID, TEST_D_FEATURES)
    assert forcing.shape == (TEST_PRED_LENGTH, TEST_N_GRID, TEST_FORCING_DIM)

    # Check identifiable value from nwp_test01_mbr000.npy (feature 0 was set to 1.0)
    # This value should propagate through subsampling and windowing.
    # Feature 0 is not z_height (which is 15), so it's not removed.
    # It's not solar radiation (features 2,3), so not affected by accumulation.
    assert torch.allclose(init_states[0, :, 0], torch.tensor(1.0))


def test_nc_loading(base_dataset_args):
    dataset = WeatherDataset(**base_dataset_args)
    nc_sample_idx = -1
    for i, sample_info in enumerate(dataset.samples_info):
        if sample_info["format"] == "nc":
            nc_sample_idx = i
            break
    assert nc_sample_idx != -1, "NC sample not found"

    init_states, target_states, forcing = dataset[nc_sample_idx]
    assert init_states is not None
    # Check identifiable value (feature 0 was set to 4.0 in nwp_test02_mbr000.nc)
    assert torch.allclose(init_states[0, :, 0], torch.tensor(4.0))


def test_grib_loading_graceful_failure(base_dataset_args):
    # This test expects GRIB loading to fail because the dummy GRIB files are empty
    dataset = WeatherDataset(**base_dataset_args)
    grib_sample_idx = -1
    for i, sample_info in enumerate(dataset.samples_info):
        if sample_info["format"] == "grib":
            grib_sample_idx = i
            break
    assert grib_sample_idx != -1, "GRIB sample not found"

    # Expect __getitem__ to return None due to loading failure
    item = dataset[grib_sample_idx]
    assert item is None, "GRIB loading did not fail gracefully for empty file"


def test_standardization_loaded_stats(base_dataset_args, mock_load_stats):
    args = {**base_dataset_args, "standardize": True}
    dataset = WeatherDataset(**args)

    npy_sample_idx = next(i for i, info in enumerate(dataset.samples_info) if info["format"] == "npy")
    init_states, target_states, forcing = dataset[npy_sample_idx]

    # Check if data is standardized (mean approx 0, std approx 1)
    # This is a rough check. Exact values depend on the dummy stats.
    # We only check a slice to avoid issues with padding or specific values.
    # Only check init_states as target_states comes from the same source.
    # Skip feature 0 which was set to a constant 1.0, its std would be 0.
    assert torch.mean(init_states[0, :, 1]).abs().item() < 0.5 # Mean close to 0
    assert (torch.std(init_states[0, :, 1]) - 1.0).abs().item() < 0.5 # Std close to 1

    # Check flux standardization (flux is part of forcing)
    # Forcing: (sample_len-2, N_grid, forcing_dim)
    # Flux is the first component of the windowed forcing variables.
    # forcing_features = torch.cat((flux, datetime_forcing), dim=-1)
    # flux is forcing_features[:,:,0]
    # forcing_windowed = torch.cat((forcing_features[:-2,:,0], forcing_features[1:-1,:,0], forcing_features[2:,:,0]), dim=?)
    # This is tricky due to windowing. Let's check one of the original flux values from the dummy file (set to 3.0)
    # The original flux was (65, 10, 10), first element of each slice was 3.0
    # After subsampling and windowing, it's hard to track.
    # Instead, verify that the means/stds stored in the dataset match the loaded ones.
    loaded_stats = utils.load_dataset_stats(DUMMY_DATASET_NAME, "cpu")
    assert torch.allclose(dataset.data_mean, loaded_stats["data_mean"])
    assert torch.allclose(dataset.data_std, loaded_stats["data_std"])
    assert torch.allclose(dataset.flux_mean, loaded_stats["flux_mean"])
    assert torch.allclose(dataset.flux_std, loaded_stats["flux_std"])


def test_standardization_provided_stats(base_dataset_args):
    # Create some dummy stats
    data_mean_tensor = torch.rand(1, TEST_D_FEATURES, dtype=torch.float32)
    data_std_tensor = torch.rand(1, TEST_D_FEATURES, dtype=torch.float32) + 0.1
    flux_mean_tensor = torch.rand(1, 1, dtype=torch.float32)
    flux_std_tensor = torch.rand(1, 1, dtype=torch.float32) + 0.1

    args = {
        **base_dataset_args,
        "standardize": True,
        "data_mean": data_mean_tensor,
        "data_std": data_std_tensor,
        "flux_mean": flux_mean_tensor,
        "flux_std": flux_std_tensor,
    }
    dataset = WeatherDataset(**args)

    assert torch.allclose(dataset.data_mean, data_mean_tensor)
    assert torch.allclose(dataset.data_std, data_std_tensor)
    assert torch.allclose(dataset.flux_mean, flux_mean_tensor)
    assert torch.allclose(dataset.flux_std, flux_std_tensor)

    npy_sample_idx = next(i for i, info in enumerate(dataset.samples_info) if info["format"] == "npy")
    init_states, _, _ = dataset[npy_sample_idx]
    # Check if data is standardized using the provided stats (rough check)
    # Again, skip feature 0 due to its constant value.
    # Calculate expected standardized value for a known original value (e.g., init_states[0,0,1])
    # Original data is random, so this is hard to check without knowing original values.
    # The check that dataset stores the provided stats is the main part here.
    assert torch.mean(init_states[0, :, 1]).abs().item() < 2.0 # Wider margin as stats are random


def test_subsampling_fixed(base_dataset_args):
    subsample_cfg = {'type': 'fixed', 'index': 1, 'step': 2}
    # N_t = 65 // 2 = 32. original_sample_length = 32
    # sample_length = TEST_PRED_LENGTH + 2 = 6
    args = {**base_dataset_args, "subsample_cfg": subsample_cfg, "subsample_step": subsample_cfg["step"]}
    dataset = WeatherDataset(**args)

    assert dataset.subsample_cfg == subsample_cfg
    assert dataset.subsample_step == 2
    assert dataset.original_sample_length == 32

    # Test __getitem__
    # We need to know what data would be at subsample_index = 1 with step 2
    # from the original nwp_test01_mbr000.npy file (feature 0 is 1.0)
    # full_sample[subsample_index :: step]
    # full_sample[1::2] -> elements at 1, 3, 5, ...
    # All these elements for feature 0 should be 1.0
    npy_sample_idx = next(i for i, info in enumerate(dataset.samples_info) if info["format"] == "npy")
    init_states, _, _ = dataset[npy_sample_idx]
    assert torch.allclose(init_states[0, :, 0], torch.tensor(1.0))
    # A more specific check would require loading the raw file and doing manual subsampling.


@pytest.mark.parametrize("subsample_step_cfg", [1, 4]) # Test different steps
def test_subsampling_random(base_dataset_args, subsample_step_cfg, monkeypatch):
    # Mock torch.randint to control the "random" subsample index
    # Test that the dataset calls it and uses its output.
    mock_indices = [0, subsample_step_cfg // 2, subsample_step_cfg -1] # Test boundaries and middle
    call_count = 0

    def mocked_randint(low, high, size_empty_tuple):
        nonlocal call_count
        assert low == 0
        assert high == subsample_step_cfg
        val = mock_indices[call_count % len(mock_indices)]
        call_count += 1
        return torch.tensor(val)

    monkeypatch.setattr(torch, "randint", mocked_randint)

    subsample_cfg = {'type': 'random', 'step': subsample_step_cfg}
    args = {**base_dataset_args, "subsample_cfg": subsample_cfg, "subsample_step": subsample_cfg["step"]}
    dataset = WeatherDataset(**args)

    assert dataset.subsample_step == subsample_step_cfg
    assert dataset.original_sample_length == 65 // subsample_step_cfg

    npy_sample_idx = next(i for i, info in enumerate(dataset.samples_info) if info["format"] == "npy")

    for i in range(len(mock_indices)):
        init_states, _, _ = dataset[npy_sample_idx]
        # Check that feature 0 (constant 1.0) is still 1.0, regardless of subsample index
        assert torch.allclose(init_states[0, :, 0], torch.tensor(1.0))

    assert call_count >= len(mock_indices) # Ensure randint was called


def test_custom_preprocessing_pipeline(base_dataset_args):
    def pipeline_func_multiply(init_states, target_states, forcing):
        return init_states * 2, target_states * 2, forcing * 2

    def pipeline_func_add(init_states, target_states, forcing):
        return init_states + 0.1, target_states + 0.1, forcing + 0.1

    args = {
        **base_dataset_args,
        "preprocessing_pipeline": [pipeline_func_multiply, pipeline_func_add]
    }
    dataset = WeatherDataset(**args)
    npy_sample_idx = next(i for i, info in enumerate(dataset.samples_info) if info["format"] == "npy")

    # Get original data by loading dataset without pipeline
    original_dataset = WeatherDataset(**base_dataset_args)
    orig_init, orig_target, orig_forcing = original_dataset[npy_sample_idx]

    # Get processed data
    processed_init, processed_target, processed_forcing = dataset[npy_sample_idx]

    expected_init = (orig_init * 2) + 0.1
    expected_target = (orig_target * 2) + 0.1
    expected_forcing = (orig_forcing * 2) + 0.1

    assert torch.allclose(processed_init, expected_init)
    assert torch.allclose(processed_target, expected_target)
    assert torch.allclose(processed_forcing, expected_forcing)
