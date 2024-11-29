import os
import sys
import argparse
import shutil

def prepare_config_blob(args):
    full_data_dir = os.path.join(os.path.abspath(args.data_dir), "train/static")
    full_new_overlay_dir = os.path.abspath(args.new_overlay_dir)
    
    config_blob = f"""{{
        "name": "czii_cryoet_mlchallenge_2024",
        "description": "2024 CZII CryoET ML Challenge training data.",
        "version": "1.0.0",

        "pickable_objects": [
            {{
                "name": "apo-ferritin",
                "is_particle": true,
                "pdb_id": "4V1W",
                "label": 1,
                "color": [0, 117, 220, 128],
                "radius": 60,
                "map_threshold": 0.0418
            }},
            {{
                "name": "beta-galactosidase",
                "is_particle": true,
                "pdb_id": "6X1Q",
                "label": 3,
                "color": [76, 0, 92, 128],
                "radius": 90,
                "map_threshold": 0.0578
            }},
            {{
                "name": "ribosome",
                "is_particle": true,
                "pdb_id": "6EK0",
                "label": 4,
                "color": [0, 92, 49, 128],
                "radius": 150,
                "map_threshold": 0.0374
            }},
            {{
                "name": "thyroglobulin",
                "is_particle": true,
                "pdb_id": "6SCJ",
                "label": 5,
                "color": [43, 206, 72, 128],
                "radius": 130,
                "map_threshold": 0.0278
            }},
            {{
                "name": "virus-like-particle",
                "is_particle": true,
                "label": 6,
                "color": [255, 204, 153, 128],
                "radius": 135,
                "map_threshold": 0.201
            }},
            {{
                "name": "membrane",
                "is_particle": false,
                "label": 8,
                "color": [100, 100, 100, 128]
            }},
            {{
                "name": "background",
                "is_particle": false,
                "label": 9,
                "color": [10, 150, 200, 128]
            }}
        ],

        "overlay_root": "{full_new_overlay_dir}",

        "overlay_fs_args": {{
            "auto_mkdir": true
        }},

        "static_root": "{full_data_dir}"
    }}"""
    
    with open(args.config_blob_dir, "w") as f:
        f.write(config_blob)
    

def prepare_overlay_data(args):
    for root, dirs, files in os.walk(args.data_dir):
        relative_path = os.path.relpath(root, args.data_dir)
        target_dir = os.path.join(args.new_overlay_dir, relative_path)
        os.makedirs(target_dir, exist_ok=True)

        for file in files:
            if file.startswith("curation_0_"):
                new_filename = file
            else:
                new_filename = f"curation_0_{file}"
            
            # Define full paths for the source and destination files
            source_file = os.path.join(root, file)
            destination_file = os.path.join(target_dir, new_filename)

            # Copy the file with the new name
            shutil.copy2(source_file, destination_file)
            print(f"Copied {source_file} to {destination_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to the data directory."
    )
    parser.add_argument(
        "--new_overlay_dir", 
        type=str,
        help="Path to the overlay directory."
    )
    parser.add_argument(
        "--config_blob_dir",
        type=str,
        help="Path to the config blob"
    )

    args = parser.parse_args()

    prepare_overlay_data(args)
    prepare_config_blob(args)
    print("Data preparation complete.")