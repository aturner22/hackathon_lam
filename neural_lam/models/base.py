import pytorch_lightning as pl
import torch

class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError
