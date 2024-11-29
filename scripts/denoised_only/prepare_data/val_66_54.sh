#!/bin/bash

python prepare_data.py \
    --data_dir data/ \
    --split_name denoised_only/val_66_54  \
    --val_experiments TS_6_6 TS_5_4