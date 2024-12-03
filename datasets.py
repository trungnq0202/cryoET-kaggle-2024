import lightning as pl
from transforms import (get_non_random_transforms, get_random_transforms)
import copick
from tqdm import tqdm
from collections import defaultdict
from copick_utils.segmentation import segmentation_from_picks
import copick_utils.writers.write as write
import numpy as np
from monai.data import CacheDataset, DataLoader, Dataset

class CryoETDataset(pl.LightningDataModule):

    def __init__(self,
                 copick_config_path,
                 train_batch_size,
                 val_batch_size,
                 num_training_dataset,
                 spatial_size = [96,96,96],
                 num_samples=16,
                 copick_segmentation_name="paintedPicks",
                 copick_user_name="copickUtils",
                 voxel_size=10,
                 tomo_type="denoised",
                 generate_masks=True):

        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.root = copick.from_file(copick_config_path)
        self.copick_segmentation_name = copick_segmentation_name
        self.copick_user_name = copick_user_name
        self.voxel_size = voxel_size
        self.tomo_type = tomo_type
        self.classes_num = len(self.root.pickable_objects) + 1
        if generate_masks:
            self._generate_mask()
        self.non_random_transforms = get_non_random_transforms()
        self.random_transforms = get_random_transforms(spatial_size, self.classes_num, num_samples)
        self.data_dicts = self._data_from_copick()
        self.train_dicts = self.data_dicts[:num_training_dataset]
        self.val_dicts = self.data_dicts[num_training_dataset:]

    def setup(self, stage):
        self.train_ds = CacheDataset(data=self.train_dicts, transform=self.non_random_transforms, cache_rate=1.0)
        self.train_ds = Dataset(data=self.train_ds, transform=self.random_transforms)
        self.val_ds = CacheDataset(data=self.val_dicts, transform=self.non_random_transforms, cache_rate=1.0)
        self.val_ds = Dataset(data=self.val_ds, transform=self.random_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.train_batch_size, shuffle=True, num_workers=1,
                          persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.val_batch_size, shuffle=False, num_workers=15,
                          persistent_workers=True)

    def _data_from_copick(self):
        data_dict = []
        for run in tqdm(self.root.runs):
            tomogram = run.get_voxel_spacing(self.voxel_size).get_tomogram(self.tomo_type).numpy()
            segmentation = run.get_segmentations(
                name=self.copick_segmentation_name,
                user_id=self.copick_user_name,
                voxel_size=self.voxel_size,
                is_multilabel=True,
            )[0].numpy()
            data_dict.append({'image': tomogram, 'label': segmentation})
        return data_dict

    def _generate_mask(self):
        target_objects = defaultdict(dict)
        for obj in self.root.pickable_objects:
            if obj.is_particle:
                target_objects[obj.name]["label"] = obj.label
                target_objects[obj.name]["radius"] = obj.radius
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
                        target_objects[pickable_object.name]["radius"] * 0.8,
                        target_objects[pickable_object.name]["label"],
                    )
            write.segmentation(run, target, self.copick_user_name, name=self.copick_segmentation_name)
