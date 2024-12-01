import logging
import os
import random
import time
import argparse

import torch
from torchinfo import summary
import hydra
import copick
from omegaconf import OmegaConf
from monai.transforms import AsDiscrete
import mlflow

from datasets import CryoETDataset
from models import CryoETUNet
from train import train

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="")
def run(config):
    # --------------------- Load configs ---------------------
    cfg = OmegaConf.to_container(config, resolve=True)


    # --------------------- Setup logger ---------------------
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    def print_line():
        prefix, unit, suffix = "#", "~~", "#"
        logger.info(prefix + unit*50 + suffix)


    # --------------------- Seed configs ---------------------
    print_line()
    logger.info("Seed: {}".format(cfg['seed']))
    

    # --------------------- Load data ---------------------
    print_line()
    logger.info("Loading data...")
    root = copick.from_file(cfg['config_blob'])

    dataset = CryoETDataset(root, voxel_size=10, tomo_type="denoised")
    train_loader, val_loader = dataset.get_dataloaders(train_batch_size=1, val_batch_size=1)
    logger.info(f"Number of classes: {len(root.pickable_objects)+1}")

    # --------------------- Load model ---------------------
    model = CryoETUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=len(root.pickable_objects)+1,
        channels=(48, 64, 80, 80),
        strides=(2, 2, 1),
        num_res_units=1,
    )


    # --------------------- Train model ---------------------
    post_pred = AsDiscrete(argmax=True, to_onehot=len(root.pickable_objects)+1)
    post_label = AsDiscrete(to_onehot=len(root.pickable_objects)+1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlflow.end_run()
    mlflow.set_experiment('Baseline 3D UNet model')
    with mlflow.start_run():
        params = {
            "epochs": cfg['train_params']['epochs'],
            "learning_rate": cfg['optimizer']['lr'],
            "loss_function": model.loss_function.__class__.__name__,
            "metric_function": model.recall_metric.__class__.__name__,
            "optimizer": "Adam",
        }
        # Log training parameters.
        mlflow.log_params(params)

        # Log model summary.
        model_output_dir = os.path.join(cfg['outputs']['model_dir'], "model_summary.txt")
        os.makedirs(cfg['outputs']['model_dir'], exist_ok=True)

        with open(model_output_dir, "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact(model_output_dir)

        train(
            model, train_loader, val_loader, post_pred, post_label,
            output_dir=cfg['outputs']['model_dir'], 
            eval_frequency=cfg['train_params']['eval_frequency'], 
            max_epochs=cfg['train_params']['epochs'], 
            device=device
        )

        # Save the trained model to MLflow.
        mlflow.pytorch.log_model(model, "baseline_model")



if __name__ == "__main__":
    run()
    
    

