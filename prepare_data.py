import os
import sys
import argparse
from utils import create_multiple_dirs

def create_output_dirs(args):
    train_data_dir = os.path.join(args.data_dir, "czii-cryo-et-object-identification/train")
    ouput_data_dir = os.path.join(args.data_dir, "splits", args.split_name)

    output_train_data_dir = os.path.join(ouput_data_dir, "train")
    ouput_train_imgs_dir = os.path.join(output_train_data_dir, "static/ExperimentRuns")
    output_train_labels_dir = os.path.join(output_train_data_dir, "overlay/ExperimentRuns")
    
    output_val_data_dir = os.path.join(ouput_data_dir, "val")
    ouput_val_imgs_dir = os.path.join(output_val_data_dir, "static/ExperimentRuns")
    output_val_labels_dir = os.path.join(output_val_data_dir, "overlay/ExperimentRuns")

    create_multiple_dirs([
        ouput_train_imgs_dir,
        output_train_labels_dir,
        ouput_val_imgs_dir,
        output_val_labels_dir
    ])

    return train_data_dir, output_train_data_dir, output_val_data_dir

def copy_data(source_dir, dest_dir, val_experiments, is_train_split):
    target_experiments = val_experiments if not is_train_split \
                                else [exp for exp in os.listdir(os.path.join(source_dir, "overlay/ExperimentRuns")) if exp not in val_experiments]

    dest_imgs_dir = os.path.join(dest_dir, "static/ExperimentRuns")
    dest_labels_dir = os.path.join(dest_dir, "overlay/ExperimentRuns")    

    for exp in target_experiments:
        exp_imgs_dir = os.path.join(source_dir, f"static/ExperimentRuns/{exp}")
        exp_labels_dir = os.path.join(source_dir, f"overlay/ExperimentRuns/{exp}")

        os.system(f"cp -r {exp_imgs_dir} {dest_imgs_dir}")
        os.system(f"cp -r {exp_labels_dir} {dest_labels_dir}")
    
def prepare_data(args):
    train_data_dir, ouput_train_imgs_dir, output_val_data_dir  = create_output_dirs(args)

    # Prep train data
    copy_data(train_data_dir, ouput_train_imgs_dir, args.val_experiments, is_train_split=True)
    
    # Prep val data
    copy_data(train_data_dir, output_val_data_dir, args.val_experiments, is_train_split=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to the data directory."
    )
    parser.add_argument(
        "--split_name", 
        type=str,
        help="Name of the split to prepare data for."
    )
    parser.add_argument(
        "--val_experiments",
        nargs="+",
        default=["TS_6_6", "TS_5_4"],
        help="List of experiment folders to include in the validation split (default: TS_6_6, TS_5_4)."
    )

    args = parser.parse_args()

    # config = craft_copick_configs(args)
    prepare_data(args)
    print("Data preparation complete.")