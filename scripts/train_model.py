# Standard library
import random
import time

# Third-party
import hydra
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import seed
from omegaconf import DictConfig, OmegaConf

# First-party
from neural_lam.configs import constants # Updated
from neural_lam.utils import utils as project_utils # Updated
from neural_lam.models.ar_model import ARModel # Added for type checking
from neural_lam.models.graph_efm import GraphEFM
from neural_lam.models.graph_fm import GraphFM
from neural_lam.models.graphcast import GraphCast
from neural_lam.data.weather_dataset import WeatherDataModule

MODELS = {
    "ar_model": ARModel, # Assuming ARModel might be selectable directly
    "graphcast": GraphCast,
    "graph_fm": GraphFM,
    "graph_efm": GraphEFM,
}


@hydra.main(config_path="../neural_lam/configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main function for training and evaluating models using Hydra.
    """
    # Set seed
    seed.seed_everything(cfg.seed)

    # Print config
    print(OmegaConf.to_yaml(cfg))

    # Load data module
    data_module = WeatherDataModule(
        dataset_name=cfg.data.dataset_name,
        pred_length=cfg.data.pred_length, # Used for val/test, train uses training.ar_steps
        subsample_step=cfg.data.subsample_step,
        standardize=cfg.data.standardize,
        subset=cfg.data.subset,
        control_only=cfg.data.control_only,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    # Instantiate model
    model_class = MODELS[cfg.model.name]
    checkpoint_path = None # Initialize checkpoint_path

    # Check if the model is a subclass of ARModel (which has the new __init__ signature)
    # This assumes GraphCast, GraphFM, GraphEFM are children of ARModel or have adopted the new signature.
    # If a model is not a child of ARModel AND has not adopted the new signature, it will fail here.
    # A more robust solution might involve a registry or a try-except block for instantiation.
    if issubclass(model_class, ARModel):
        if cfg.resume_run_id and cfg.resume_run_id.strip():
            wandb_id = cfg.resume_run_id.split("/")[-1]
            checkpoint_path = project_utils.get_checkpoint_path_from_wandb_id(wandb_id, project_name=cfg.project_name)
            if checkpoint_path:
                print(f"Resuming {cfg.model.name} from checkpoint: {checkpoint_path}")
                # For models with new signature, pass config groups to load_from_checkpoint
                # This requires load_from_checkpoint to be able to handle these.
                # PyTorch Lightning's default load_from_checkpoint saves hparams passed to __init__.
                # If __init__ now takes these config groups, it should work assuming
                # the checkpoint was saved with a version of the model that had this new __init__.
                # If loading older checkpoints, this might need a custom load_from_checkpoint or a conversion.
                model = model_class.load_from_checkpoint(
                    checkpoint_path,
                    map_location=torch.device('cpu'),
                    model_cfg=cfg.model,
                    training_cfg=cfg.training,
                    data_cfg=cfg.data
                )
            else:
                print(f"Could not find checkpoint for run_id {cfg.resume_run_id}. Starting new {cfg.model.name} run.")
                model = model_class(cfg.model, cfg.training, cfg.data)
        else:
            model = model_class(cfg.model, cfg.training, cfg.data)
    else:
        # This block is for models that DO NOT inherit from ARModel and have NOT been updated
        # to the new __init__(model_cfg, training_cfg, data_cfg) signature.
        # It attempts to use the old `model_args` style.
        print(f"Warning: Model {cfg.model.name} does not seem to be ARModel child. Using legacy args constructor.")
        model_args = OmegaConf.create({
            "lr": cfg.training.lr,
            "output_std": cfg.model.output_std,
            "loss": cfg.training.loss_func,
            "step_length": cfg.data.subsample_step,
            "n_example_pred": cfg.training.n_example_pred,
            "dataset": cfg.data.dataset_name,
            # Common model params (ensure these cover what old models need)
            "hidden_dim": cfg.model.get("hidden_dim", 64), # Default if not in model_cfg
            "processor_layers": cfg.model.get("processor_layers", 4), # Default
            "graph_name": cfg.model.get("graph_name", "multiscale"),
            "hidden_layers": cfg.model.get("hidden_layers", 1),
            "mesh_aggr": cfg.model.get("mesh_aggr", "sum"),
            "vertical_propnets": cfg.model.get("vertical_propnets", False),
            "latent_dim": cfg.model.get("latent_dim", None),
            "prior_processor_layers": cfg.model.get("prior_processor_layers", 2),
            "encoder_processor_layers": cfg.model.get("encoder_processor_layers", 2),
            "learn_prior": cfg.model.get("learn_prior", True),
            "prior_dist": cfg.model.get("prior_dist", "isotropic"),
            # EFM specific args (these might be better placed in model_cfg or training_cfg explicitly)
            "sample_obs_noise": cfg.training.get("sample_obs_noise", False),
            "ensemble_size": cfg.training.get("ensemble_size", 5),
            "kl_beta": cfg.training.get("kl_beta", 1.0),
            "crps_weight": cfg.training.get("crps_weight", 0.0),
        })

        if cfg.resume_run_id and cfg.resume_run_id.strip():
            wandb_id = cfg.resume_run_id.split("/")[-1]
            checkpoint_path = project_utils.get_checkpoint_path_from_wandb_id(wandb_id, project_name=cfg.project_name)
            if checkpoint_path:
                print(f"Resuming {cfg.model.name} from checkpoint (legacy args): {checkpoint_path}")
                # This assumes old models' load_from_checkpoint expects `args`
                model = model_class.load_from_checkpoint(checkpoint_path, map_location=torch.device('cpu'), args=model_args)
            else:
                print(f"Could not find checkpoint for run_id {cfg.resume_run_id}. Starting new {cfg.model.name} run.")
                model = model_class(model_args)
        else:
            model = model_class(model_args)


    # Set up logger
    run_name = f"{cfg.model.name}-{cfg.data.dataset_name}-{time.strftime('%Y%m%d_%H%M%S')}"
    if cfg.run_mode == "debug":
        run_name = "debug-" + run_name

    logger = pl.loggers.WandbLogger(
        project=cfg.project_name,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        notes=cfg.notes if cfg.notes else None,
        resume="allow" if cfg.resume_run_id else None, # Allow resuming if id is provided
        id=cfg.resume_run_id.split("/")[-1] if cfg.resume_run_id and cfg.resume_run_id.strip() else None,
    )

    # Callbacks
    callbacks = []
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            monitor=cfg.training.monitor_metric,
            mode=cfg.training.mode,
            save_top_k=cfg.training.save_top_k,
            save_last=True, # Always save last checkpoint
            filename='{epoch}-{' + cfg.training.monitor_metric + ':.2f}'
        )
    )
    # Add other callbacks from constants.VAL_STEP_CHECKPOINTS if needed, adapting to cfg
    # Example:
    # for unroll_time in constants.VAL_STEP_CHECKPOINTS:
    #     metric_name = f"val_loss_unroll{unroll_time}"
    #     callbacks.append(
    #         pl.callbacks.ModelCheckpoint(
    #             monitor=metric_name,
    #             mode="min",
    #             filename=f'{{epoch}}-{{{metric_name}:.2f}}'
    #         )
    #     )

    # Training strategy (example, may need adjustment)
    # strategy = "ddp" if cfg.training.kl_beta > 0 else "ddp_find_unused_parameters_true"
    # Simplified for now, assuming standard DDP if multiple GPUs
    strategy = "ddp" if cfg.training.gpus > 1 else None


    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        max_steps=cfg.training.max_steps if cfg.training.max_steps > 0 else -1,
        deterministic=True, # Keep deterministic for reproducibility
        accelerator="gpu" if cfg.training.gpus > 0 else "cpu",
        devices=cfg.training.gpus if cfg.training.gpus > 0 else "auto",
        logger=logger,
        callbacks=callbacks,
        check_val_every_n_epoch=int(cfg.training.val_check_interval) if isinstance(cfg.training.val_check_interval, float) and cfg.training.val_check_interval.is_integer() else cfg.training.val_check_interval,
        precision=str(cfg.training.precision), # PL expects string for precision
        log_every_n_steps=cfg.training.log_every_n_steps,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        strategy=strategy,
        # Add other trainer args from cfg.training as needed
    )

    if trainer.global_rank == 0:
        # utils.init_wandb_metrics(logger) # May need to be adapted or removed
        pass


    if cfg.run_mode == "test":
        print("Running evaluation on test set...")
        trainer.test(model=model, datamodule=data_module)
    elif cfg.run_mode == "train" or cfg.run_mode == "debug":
        print("Starting training...")
        trainer.fit(model=model, datamodule=data_module, ckpt_path=checkpoint_path if cfg.resume_run_id and checkpoint_path else None)
        # After training, optionally run test
        # print("Running evaluation on test set after training...")
        # trainer.test(model=model, datamodule=data_module) # best=True could be an option
    else:
        raise ValueError(f"Unknown run_mode: {cfg.run_mode}")


if __name__ == "__main__":
    main()
