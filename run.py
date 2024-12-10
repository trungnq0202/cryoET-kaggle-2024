import logging
import os
import torch
import hydra
from omegaconf import OmegaConf
import mlflow
from datasets import CryoETDataset
from models import Model
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.model_summary import ModelSummary

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
        logger.info(prefix + unit * 50 + suffix)

    # --------------------- Seed configs ---------------------
    print_line()
    logger.info("Seed: {}".format(cfg['seed']))

    # --------------------- Load data ---------------------
    print_line()
    logger.info("Loading data...")
    source_dir = "/home/xie.chang/cryo_kaggle/czii-cryo-et-object-identification/train/overlay"
    destination_dir = "/home/xie.chang/cryoET-kaggle-2024/data/train/overlay_v1.0"
    CryoETDataset.copy_from_source(source_dir, destination_dir)
    dataset = CryoETDataset(
        cfg['config_blob'],
        train_batch_size=1,
        val_batch_size=2,
        num_training_dataset=6)

    # --------------------- Load model ---------------------
    model = Model(
        spatial_dims=3,
        in_channels=1,
        out_channels=7,
        channels=cfg['train_params']['channels'],
        strides=cfg['train_params']['strides'],
        num_res_units=cfg['train_params']['num_res_units'],
        lr=cfg['optimizer']['lr'],
        beta=cfg['train_params']['alpha'],
        alpha=cfg['train_params']['beta']
    )
    
    # logger.info(model.parameters)
    # --------------------- Train model ---------------------

    torch.set_float32_matmul_precision("medium")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_metric",
        dirpath="checkpoints/",
        filename="best_model",
        save_top_k=1,
        mode="max",
    )
    trainer = pl.Trainer(
        max_epochs=cfg['train_params']['epochs'],
        accelerator="gpu",
        devices=[0],
        num_nodes=1,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
    )
    mlflow.end_run()
    mlflow.set_experiment('Baseline 3D UNet model')
    with mlflow.start_run():
        params = {
            "epochs": cfg['train_params']['epochs'],
            "channels": cfg['train_params']['channels'],
            "strides": cfg['train_params']['strides'],
            "num_res_units": cfg['train_params']['num_res_units'],
            "learning_rate": cfg['optimizer']['lr'],
            "alpha": cfg['train_params']['alpha'],
            "beta": cfg['train_params']['beta'],
            "loss_function": model.loss_fn.__class__.__name__,
            "metric_function": model.metric_fn.__class__.__name__,
            "optimizer": "Adam",

        }
        # Log training parameters.
        mlflow.log_params(params)

        # Log model summary.
        model_output_dir = os.path.join(cfg['outputs']['model_dir'], "model_summary.txt")
        os.makedirs(cfg['outputs']['model_dir'], exist_ok=True)

        with open(model_output_dir, "w") as f:
            f.write(str(ModelSummary(model)))
        mlflow.log_artifact(model_output_dir)
        trainer.fit(model, dataset)

        # Save the trained model to MLflow.
        mlflow.pytorch.log_model(model, "baseline_model_v1")


if __name__ == "__main__":
    run()
