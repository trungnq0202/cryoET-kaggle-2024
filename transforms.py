from monai.transforms import (
    Compose, 
    EnsureChannelFirstd, 
    Orientationd,  
    NormalizeIntensityd, 
    RandFlipd, 
    RandRotate90d, 
    RandCropByLabelClassesd
)


def get_random_transforms(n_samples):
    """Creates data transforms.

    Returns:
        Compose: A MONAI Compose object with the desired transformations.
    """
    random_transforms = Compose([
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=[96, 96, 96],
            num_classes=8,
            num_samples=n_samples
        ),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 2]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),    
    ])
    return random_transforms


def get_non_random_transforms():
    non_random_transforms = Compose([
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS")
    ])

    return non_random_transforms


def get_validation_transforms(n_samples):
    val_transforms = Compose([
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image"),
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=[96, 96, 96],
            num_classes=8,
            num_samples=n_samples,  # Use 1 to get a single, consistent crop per image
        ),
    ])

    return val_transforms