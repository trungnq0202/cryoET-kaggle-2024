import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
from monai.data import Dataset, CacheDataset, DataLoader
from monai.transforms import (
    Compose, 
    EnsureChannelFirstd, 
    Orientationd,  
    NormalizeIntensityd, 
    RandFlipd, 
    RandRotate90d, 
    RandCropByLabelClassesd
)
from copick_utils.segmentation import segmentation_from_picks
import copick_utils.writers.write as write


class CryoETDataset:
    def __init__(
        self,
        root,
        voxel_size=10,
        tomo_type="denoised",
        copick_user_name="copickUtils",
        copick_segmentation_name="paintedPicks",
        generate_masks=True,
        num_classes=8,
        train_split=0.8,
        spatial_size=(96, 96, 96),
        num_samples=16,
    ):
        """
        Initializes the CryoETDataset class with the given parameters.

        Args:
            root (object): The root object from the dataset (e.g., from CoPick).
            voxel_size (int): The voxel size for tomogram data.
            tomo_type (str): The type of tomogram (e.g., "denoised").
            copick_user_name (str): User name for segmentation generation.
            copick_segmentation_name (str): Name for segmentation mask.
            num_classes (int): Number of label classes for segmentation.
            train_split (float): Proportion of the dataset used for training.
            spatial_size (tuple): Dimensions for cropping data.
            num_samples (int): Number of samples for random crops.
        """
        self.root = root
        self.voxel_size = voxel_size
        self.tomo_type = tomo_type
        self.copick_user_name = copick_user_name
        self.copick_segmentation_name = copick_segmentation_name
        self.num_classes = num_classes
        self.train_split = train_split
        self.spatial_size = spatial_size
        self.num_samples = num_samples
        self.generate_masks = generate_masks
        self.data_dicts = []
        self.train_files = []
        self.val_files = []
        self._generate_mask()
        self._prepare_data_dicts()
        self._split_data()

    def _generate_mask(self):
        if self.generate_masks:
            target_objects = defaultdict(dict)
            for object in self.root.pickable_objects:
                if object.is_particle:
                    target_objects[object.name]['label'] = object.label
                    target_objects[object.name]['radius'] = object.radius


            for run in tqdm(self.root.runs):
                tomo = run.get_voxel_spacing(10)
                tomo = tomo.get_tomogram(self.tomo_type).numpy()
                target = np.zeros(tomo.shape, dtype=np.uint8)
                for pickable_object in self.root.pickable_objects:
                    pick = run.get_picks(object_name=pickable_object.name, user_id="curation")
                    if len(pick):  
                        target = segmentation_from_picks.from_picks(
                            pick[0], 
                            target, 
                            target_objects[pickable_object.name]['radius'] * 0.8,
                            target_objects[pickable_object.name]['label']
                        )
                write.segmentation(run, target, self.copick_user_name, name=self.copick_segmentation_name)

    def _prepare_data_dicts(self):
        """Prepares the data dictionaries by loading tomograms and segmentations."""
        for run in tqdm(self.root.runs, desc="Preparing Data Dicts"):
            tomogram = run.get_voxel_spacing(self.voxel_size).get_tomogram(self.tomo_type).numpy()
            segmentation = run.get_segmentations(
                name=self.copick_segmentation_name,
                user_id=self.copick_user_name,
                voxel_size=self.voxel_size,
                is_multilabel=True
            )[0].numpy()
            self.data_dicts.append({"image": tomogram, "label": segmentation})

    def _split_data(self):
        """Splits the dataset into training and validation sets."""
        num_train = int(len(self.data_dicts) * self.train_split)
        self.train_files = self.data_dicts[:num_train]
        self.val_files = self.data_dicts[num_train:]

    def get_transforms(self, random=False):
        """Creates data transforms.

        Args:
            random (bool): Whether to include random augmentations.

        Returns:
            Compose: A MONAI Compose object with the desired transformations.
        """
        transforms = [
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
        ]
        if random:
            transforms += [
                RandCropByLabelClassesd(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=self.spatial_size,
                    num_classes=self.num_classes,
                    num_samples=self.num_samples,
                ),
                RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 2]),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            ]
        return Compose(transforms)

    def get_dataloaders(self, train_batch_size=1, val_batch_size=1, num_workers=4):
        """Creates dataloaders for training and validation datasets.

        Args:
            train_batch_size (int): Batch size for training.
            val_batch_size (int): Batch size for validation.
            num_workers (int): Number of workers for DataLoader.

        Returns:
            tuple: Training and validation DataLoader objects.
        """
        # Training dataset and DataLoader
        train_ds = CacheDataset(data=self.train_files, transform=self.get_transforms(), cache_rate=1.0)
        train_ds = Dataset(data=train_ds, transform=self.get_transforms(random=True))
        train_loader = DataLoader(
            train_ds,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        # Validation dataset and DataLoader
        val_ds = CacheDataset(data=self.val_files, transform=self.get_transforms(), cache_rate=1.0)
        val_loader = DataLoader(
            val_ds,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        return train_loader, val_loader
