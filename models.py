import torch
import lightning as pl
from monai.networks.nets import UNet
from monai.losses import TverskyLoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
import mlflow
import torch.nn.functional as F
from torch import nn

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0, smooth=1e-7, 
                 include_background=True, to_onehot_y=True, softmax=False):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.softmax = softmax

    def forward(self, y_pred, y_true):
        # print("y_true shape:", y_true.shape)
        # print("y_pred shape:", y_pred.shape)
        # Softmax 或 Sigmoid 激活
        if self.softmax:
            y_pred = torch.softmax(y_pred, dim=1)
        else:
            y_pred = torch.sigmoid(y_pred)
        
        # 转换为 one-hot 格式
        if self.to_onehot_y:
            num_classes = y_pred.shape[1]
            y_true = torch.squeeze(y_true, dim=1) 
            y_true = F.one_hot(y_true.long(), num_classes=num_classes)  # 转换为 one-hot
            y_true = y_true.permute(0, 4, 1, 2, 3).float()  # 调整为 NCHWD 格式
        
        # 忽略背景类别
        if not self.include_background:
            y_pred = y_pred[:, 1:]
            y_true = y_true[:, 1:]

        # 计算 TP, FP, FN
        tp = torch.sum(y_true * y_pred, dim=(2, 3, 4))  # 真正例
        fp = torch.sum((1 - y_true) * y_pred, dim=(2, 3, 4))  # 假正例
        fn = torch.sum(y_true * (1 - y_pred), dim=(2, 3, 4))  # 假反例

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Focal 调制
        focal_tversky_loss = torch.mean((1 - tversky_index) ** self.gamma)
        
        return focal_tversky_loss

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
        beta=0.8,
        alpha=0.2
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

        # self.loss_fn = TverskyLoss(
        #     include_background=True, to_onehot_y=True, softmax=True, beta=beta, alpha=alpha
        # )
        self.loss_fn = FocalTverskyLoss(
            include_background=True, to_onehot_y=True, softmax=True, beta=beta, alpha=alpha
        )
        self.metric_fn = DiceMetric(
            include_background=False, reduction="mean", ignore_empty=True
        )
        # self.metric_fn = ConfusionMatrixMetric(include_background=False, metric_name="recall", reduction="None")
        self.weight = torch.Tensor([1,0,2,1,2,1]).to('cuda')
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
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        # torch.cuda.empty_cache()
        return loss


    def on_train_epoch_end(self) -> None:
        if self.num_train_batch > 0:
           
            train_loss = self.trainer.callback_metrics["train_loss_epoch"]
            mlflow.log_metric("train_loss", train_loss, step=self.epoch)
            # self.log("train_loss", loss_per_epoch, prog_bar=True)
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
                AsDiscrete(to_onehot=self.hparams.out_channels)(i)
                for i in decollate_batch(y)
            ]

            self.metric_fn(y_pred=metrix_val_outputs, y=metrix_val_labels)
            metrics = self.metric_fn.aggregate(reduction="mean_batch")
            val_metric = torch.sum(metrics*self.weight)/torch.sum(self.weight)
            
            for i,m in enumerate(metrics):
                mlflow.log_metric(f"validation metric class {i+1}", m, step=self.epoch)
                self.log(f"class {i+1}", m*self.weight[i]/7, on_epoch=True, prog_bar=True)
            self.val_metric += val_metric
            self.num_val_batch += 1
        
        return {"val_metric": val_metric}


    def on_validation_epoch_end(self) -> None:

        loss_per_epoch = self.val_metric / self.num_val_batch
        mlflow.log_metric("val_metric", loss_per_epoch, step=self.epoch)
        self.log("val_metric", loss_per_epoch, prog_bar=True)
        self.val_metric = 0
        self.num_val_batch = 0

        self.epoch += 1


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
