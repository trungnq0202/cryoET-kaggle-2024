import torch
from monai.networks.nets import UNet
from monai.losses import DiceLoss, TverskyLoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric


class CryoETUNet(UNet):
    def __init__(
        self,
        spatial_dims=3,
        in_channels=1,
        out_channels=8,  # Number of output classes, adjustable
        channels=(48, 64, 80, 80),
        strides=(2, 2, 1),
        num_res_units=1,
        learning_rate=1e-3,
        device=None,
    ):
        """
        Initializes the CryoET UNet model, optimizer, loss function, and metrics.

        Args:
            spatial_dims (int): Number of spatial dimensions (e.g., 3 for 3D).
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels (e.g., number of classes).
            channels (tuple): Number of filters in each layer of the UNet.
            strides (tuple): Stride values for downsampling in each layer.
            num_res_units (int): Number of residual units per layer.
            learning_rate (float): Learning rate for the optimizer.
            device (torch.device or str): Device to run the model on (e.g., "cuda" or "cpu").
        """
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
        )

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # Move the model to the specified device

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_function = TverskyLoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,  # Use softmax for multiclass segmentation
        )
        self.metrics_function = DiceMetric(
            include_background=False,
            reduction="mean",
            ignore_empty=True,
        )
        self.recall_metric = ConfusionMatrixMetric(
            include_background=False,
            metric_name="recall",
            reduction="none",
        )


    def forward(self, x):
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model output.
        """
        return self.model(x)

    # def compute_loss(self, predictions, targets):
    #     """
    #     Computes the loss between predictions and targets.

    #     Args:
    #         predictions (torch.Tensor): Model predictions.
    #         targets (torch.Tensor): Ground truth labels.

    #     Returns:
    #         torch.Tensor: Loss value.
    #     """
    #     return self.loss_function(predictions, targets)

    # def update_metrics(self, predictions, targets):
    #     """
    #     Updates metrics based on predictions and targets.

    #     Args:
    #         predictions (torch.Tensor): Model predictions.
    #         targets (torch.Tensor): Ground truth labels.

    #     Returns:
    #         dict: Updated metric values.
    #     """
    #     # Convert predictions to one-hot if necessary
    #     self.dice_metric(y_pred=predictions, y=targets)
    #     self.recall_metric(y_pred=predictions, y=targets)

    #     return {
    #         "dice": self.dice_metric.aggregate().item(),
    #         "recall": self.recall_metric.aggregate(),
    #     }

    # def reset_metrics(self):
    #     """Resets all metrics."""
    #     self.dice_metric.reset()
    #     self.recall_metric.reset()

    # def save_model(self, path):
    #     """
    #     Saves the model to the specified path.

    #     Args:
    #         path (str): Path to save the model.
    #     """
    #     torch.save(self.model.state_dict(), path)

    # def load_model(self, path):
    #     """
    #     Loads the model from the specified path.

    #     Args:
    #         path (str): Path to the model checkpoint.
    #     """
    #     self.model.load_state_dict(torch.load(path, map_location=self.device))
