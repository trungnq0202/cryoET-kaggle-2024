import torch
import lightning as pl
from monai.networks.nets import UNet
from monai.losses import TverskyLoss
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
import mlflow

class Model(pl.LightningModule):
    def __init__(
        self,
        spatial_dims=3,
        in_channels=1,
        out_channels=7,
        channels=(48, 64, 80, 80),
        strides=(2, 2, 1),
        num_res_units=1,
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(
            spatial_dims=self.hparams.spatial_dims,
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.out_channels,
            channels=self.hparams.channels,
            strides=self.hparams.strides,
            num_res_units=self.hparams.num_res_units,
        )

        self.loss_fn = TverskyLoss(
            include_background=True, to_onehot_y=True, softmax=True,beta=0.7,alpha=0.3
        )
        self.metric_fn = DiceMetric(
            include_background=True, reduction="mean", ignore_empty=True
        )

        self.train_loss = 0
        self.val_metric = 0
        self.num_train_batch = 0
        self.num_val_batch = 0
        self.epoch = 1

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.train_loss += loss
        self.num_train_batch += 1
        torch.cuda.empty_cache()
        return loss


    def on_train_epoch_end(self) -> None:
        if self.num_train_batch > 0:
            loss_per_epoch = self.train_loss / self.num_train_batch
            mlflow.log_metric("train_loss", loss_per_epoch,step=self.epoch)
            self.log("train_loss", loss_per_epoch, prog_bar=True)
            self.train_loss = 0
            self.num_train_batch = 0

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch["image"], batch["label"]
            y_hat = self(x)
            metrix_val_outputs = [
                AsDiscrete(argmax=True, to_onehot=self.hparams.out_channels)(i)
                for i in decollate_batch(y_hat)
            ]
            metrix_val_labels = [
                AsDiscrete(argmax=True, to_onehot=self.hparams.out_channels)(i)
                for i in decollate_batch(y)
            ]
            self.metric_fn(y_pred=metrix_val_outputs, y=metrix_val_labels)
            metrics = self.metric_fn.aggregate(reduction="mean_batch")
            val_metric = torch.mean(metrics)
            self.val_metric += val_metric
            self.num_val_batch += 1
        torch.cuda.empty_cache()
        return {"val_metric": val_metric}


    def on_validation_epoch_end(self) -> None:
        if self.num_val_batch > 0:
            loss_per_epoch = self.val_metric / self.num_val_batch
            mlflow.log_metric("val_metric", loss_per_epoch, step=self.epoch)
            self.log("val_metric", loss_per_epoch, prog_bar=True)
            self.val_metric = 0
            self.num_val = 0
        self.epoch += 1


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
