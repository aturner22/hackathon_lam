import pytest
import hydra
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

# Import the model to be tested
from neural_lam.models.ar_model import ARModel
# Import GraphCast to ensure it can also be part_of the test later if needed
from neural_lam.models.graphcast import GraphCast

@pytest.fixture(scope="function")
def hydra_initialize_for_model():
    """Fixture to initialize Hydra and clear its state after the test."""
    if not GlobalHydra.instance().is_initialized():
        # Initialize with the path to your config directory
        initialize(config_path="../neural_lam/configs", version_base=None)
    yield
    GlobalHydra.instance().clear()

def test_ar_model_instantiation(hydra_initialize_for_model):
    """Test that ARModel can be instantiated with a composed Hydra config."""
    # Compose a configuration. We can override parts of it for the test.
    # For ARModel, it needs `model_cfg`, `training_cfg`, `data_cfg`.
    # We'll use the defaults defined in `config.yaml` which point to
    # `model: graphcast`, `data: weather`, `training: default`.
    # ARModel's __init__ expects certain fields in these cfgs.
    cfg = compose(config_name="config", overrides=[
        "model=graphcast", # graphcast inherits from ARModel or similar enough for base ARModel part
        "data=weather",
        "training=default",
        # Minimal overrides for quick testing if needed
        "data.dataset_name=test_dummy_model_dataset", # To avoid issues with real data paths in static feature loading
        "data.subsample_step=3", # Example value ARModel might use from data_cfg
        "model.output_std=False", # Example value ARModel might use from model_cfg
        "training.lr=0.001", # Example value ARModel might use from training_cfg
        "training.n_example_pred=1", # Example
        # ARModel loads static data like 'grid_static_features.pt', 'param_weights.pt'
        # These are usually in data/<dataset_name>/static/.
        # If 'test_dummy_model_dataset/static' doesn't exist or is empty, this will fail.
        # For a true unit test, utils.load_static_data might need mocking or
        # a dedicated small test dataset fixture.
    ])

    # Instantiate the model
    # ARModel expects model_cfg, training_cfg, data_cfg
    try:
        model = ARModel(
            model_cfg=cfg.model,
            training_cfg=cfg.training,
            data_cfg=cfg.data
        )
        assert model is not None, "Model should be instantiated."
        # Add more assertions here if needed, e.g., checking some initialized parameters
        assert model.lr == cfg.training.lr
        assert model.output_std == cfg.model.output_std

    except FileNotFoundError as e:
        # This might happen if static data files are missing for the dummy dataset
        pytest.skip(f"Skipping model instantiation test due to missing static data for dummy dataset: {e}")
    except Exception as e:
        if "No such file or directory" in str(e) and "static_features.pt" in str(e):
             pytest.skip(f"Skipping model instantiation test due to missing static data for dummy dataset: {e}")
        else:
            # Re-raise other exceptions to fail the test
            pytest.fail(f"ARModel instantiation failed with an unexpected error: {e}")


# Example for GraphCast (more complex, might need more specific overrides)
# def test_graphcast_model_instantiation(hydra_initialize_for_model):
#     cfg = compose(config_name="config", overrides=[
#         "model=graphcast",
#         "data=weather",
#         "training=default",
#         "data.dataset_name=test_dummy_graphcast_dataset",
#         # GraphCast specific model params that might be in model_cfg.graph_name, etc.
#         # e.g. model.graph_name if it's used by load_graph
#         "model.graph_name=test_dummy_graph", # if load_graph needs a name
#         "model.hidden_layers=1", # BaseGraphModel needs this
#         "model.vertical_propnets=False", # BaseGraphModel needs this
#         "model.mesh_aggr='sum'", # GraphCast needs this
#     ])
#
#     try:
#         # GraphCast inherits from BaseGraphModel -> ARModel
#         model = GraphCast(
#             model_cfg=cfg.model,
#             training_cfg=cfg.training,
#             data_cfg=cfg.data
#         )
#         assert model is not None
#     except FileNotFoundError as e:
#         pytest.skip(f"Skipping GraphCast instantiation due to missing static/graph data: {e}")
#     except Exception as e:
#         if "No such file or directory" in str(e): # more specific checks if possible
#              pytest.skip(f"Skipping GraphCast instantiation due to missing data: {e}")
#         else:
#             pytest.fail(f"GraphCast instantiation failed with an unexpected error: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
