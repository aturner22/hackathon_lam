import hydra
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
import pytest

def test_load_default_config():
    """Test that the default Hydra configuration loads and has the main keys."""
    GlobalHydra.instance().clear()  # Clear any previous Hydra state
    # Initialize Hydra - relative path to config directory
    # The path is relative to this test file, assuming 'neural_lam' is a sibling of 'tests'
    # or that the CWD for tests will be the repo root.
    # For robustness, it's often better to use absolute paths or a clear anchor point.
    # However, `../neural_lam/configs` is standard if tests are run from `tests/` or repo root with `python -m pytest`.
    # Hydra's typical search path is `conf` or specified. Let's try relative to a hypothetical CWD of repo root.
    initialize(config_path="../neural_lam/configs", version_base=None)
    cfg = compose(config_name="config")

    assert cfg is not None
    assert isinstance(cfg, DictConfig)

    # Check for main keys
    assert "model" in cfg
    assert "data" in cfg
    assert "training" in cfg

    # Check if defaults are loaded
    assert cfg.model.name == "graphcast" # From defaults list in config.yaml
    assert cfg.data.dataset_name == "cosmo_de_npy" # From defaults list
    assert cfg.training.lr == 0.0005 # From defaults list

    # Check a specific value from a sub-config
    assert cfg.model.hidden_dim == 256 # From model/graphcast.yaml
    assert cfg.data.batch_size == 2 # From data/weather.yaml
    assert cfg.training.epochs == 100 # From training/default.yaml

if __name__ == "__main__":
    pytest.main()
