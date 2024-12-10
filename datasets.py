import lightning as pl
from transforms import (get_non_random_transforms, get_random_transforms, get_val_transforms)
import copick
from tqdm import tqdm
from collections import defaultdict
from copick_utils.segmentation import segmentation_from_picks
import copick_utils.writers.write as write
import numpy as np
from monai.data import CacheDataset, DataLoader, Dataset
from utils import create_dir, extract_3d_patches_minimal_overlap
import shutil
import os
class CryoETDataset(pl.LightningDataModule):

    def __init__(self,
                 copick_config_path,
                 train_batch_size,
                 val_batch_size,
                 num_training_dataset,
                 spatial_size = [96,96,96],
                 num_samples=8,
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
        self.train_dicts = self.process_data(self.train_dicts, spatial_size, 12)
        val_dicts = self.data_dicts[num_training_dataset:]
        val_image = [val["image"] for val in val_dicts]
        val_label = [val["label"] for val in val_dicts]
        val_img_patches, img_coordinates = extract_3d_patches_minimal_overlap(val_image, 96)
        val_label_patches, label_coordinates = extract_3d_patches_minimal_overlap(val_label, 96)
        self.val_dicts = [
            {"image": image, "label": label}
            for (image, label) in zip(val_img_patches, val_label_patches)
        ]
        


    def setup(self, stage):
        self.train_ds = CacheDataset(data=self.train_dicts, transform=self.non_random_transforms, cache_rate=1.0)
        self.train_ds = Dataset(data=self.train_ds, transform=self.random_transforms)


        self.val_ds = CacheDataset(data=self.val_dicts, transform=self.non_random_transforms, cache_rate=1.0)
        # self.val_ds = Dataset(data=self.val_ds, transform=self.val_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.train_batch_size, shuffle=True, num_workers=12,
                          persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.val_batch_size, shuffle=False, num_workers=12,
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
            unique, counts = np.unique(segmentation, return_counts=True)
            # print("Label distribution:", dict(zip(unique, counts)))
            data_dict.append({'image': tomogram, 'label': segmentation})
        return data_dict
    
    @staticmethod
    def copy_from_source(source_dir, destination_dir):
        
        # print(source_dir)
        # print(destination_dir)
        for root, dirs, files in os.walk(source_dir):
            # print(root)
            # Create corresponding subdirectories in the destination
            relative_path = os.path.relpath(root, source_dir)
            target_dir = os.path.join(destination_dir, relative_path)
            os.makedirs(target_dir, exist_ok=True)

            # Copy and rename each file
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
                # print(f"Copied {source_file} to {destination_file}")


    def random_crop_3d(self, image, label, crop_size):
        """
        对三维图像和标签进行随机裁剪
        :param image: numpy数组, 代表三维图像
        :param label: numpy数组, 代表对应的标签
        :param crop_size: tuple, 裁剪的尺寸 (depth, height, width)
        :return: 裁剪后的图像和标签
        """
        # 获取图像尺寸
        depth, height, width = image.shape

        # 确保裁剪尺寸小于图像尺寸
        crop_d, crop_h, crop_w = crop_size
        assert crop_d <= depth and crop_h <= height and crop_w <= width, "裁剪尺寸不能大于图像尺寸"

        # 随机生成起始点
        start_d = np.random.randint(0, depth - crop_d)
        start_h = np.random.randint(0, height - crop_h)
        start_w = np.random.randint(0, width - crop_w)

        # 裁剪图像和标签
        cropped_image = image[start_d:start_d + crop_d, start_h:start_h + crop_h, start_w:start_w + crop_w]
        cropped_label = label[start_d:start_d + crop_d, start_h:start_h + crop_h, start_w:start_w + crop_w]

        return cropped_image, cropped_label

    def process_data(self, data_list, crop_size, num_crops=1):
        """
        
        :param data_list: list[dict{'image', 'label'}], 输入数据
        :param crop_size: tuple, 
        :param num_crops: int, 
        :return: list[dict{'image', 'label'}]
        """
        cropped_data = []
        for data in data_list:
            image = data['image']
            label = data['label']
            for _ in range(num_crops):
                cropped_image, cropped_label = self.random_crop_3d(image, label, crop_size)
                cropped_data.append({'image': cropped_image, 'label': cropped_label})
        return cropped_data

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
