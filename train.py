import logging
import os
import random
import time
import argparse

import torch
import hydra
import copick
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="")
def train(config):
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
    
    


if __name__ == "__main__":
    train()
    
    

