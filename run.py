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
    dataset = CryoETDataset(
        cfg['config_blob'],
        train_batch_size=1,
        val_batch_size=16,
        num_training_dataset=5,
        generate_masks=False)

    # --------------------- Load model ---------------------
    model = Model(
        spatial_dims=3,
        in_channels=1,
        out_channels=7,
        channels=(48, 64, 80, 80),
        strides=(2, 2, 1),
        num_res_units=1,
        lr=cfg['optimizer']['lr']
    )

    # --------------------- Train model ---------------------

    torch.set_float32_matmul_precision("medium")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_metric",
        dirpath="checkpoints/",
        filename="best_model",
        save_top_k=1,
        mode="min",
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
            "learning_rate": cfg['optimizer']['lr'],
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
        mlflow.pytorch.log_model(model, "baseline_model")


if __name__ == "__main__":
    run()
