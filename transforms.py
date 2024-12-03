from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    Orientationd, RandCropByLabelClassesd, RandRotate90d, RandFlipd,
)


def get_random_transforms(spatial_size, num_classes, num_samples):
    return Compose(
        [
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                num_classes=num_classes,
                num_samples=num_samples,
            ),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(1, 2)),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
        ]
    )


def get_non_random_transforms():
    return Compose(
        [
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            NormalizeIntensityd(keys="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
        ]
    )
