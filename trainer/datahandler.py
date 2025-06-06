import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os


both_transform = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5)
    ],
    additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

class DataHandler(Dataset):
    def __init__(self, root_dir: str, target_side: str = "right"):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        self.size = self._infer_shape()
        self.target_side = target_side

    def _infer_shape(self) -> int:
        image_path = os.path.join(self.root_dir, self.list_files[0])
        image = Image.open(image_path)
        return int(image.size[0] / 2)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :self.size, :]
        target_image = image[:, self.size:, :]

        augmentations = both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = transform_only_input(image=input_image)["image"]
        target_image = transform_only_mask(image=target_image)["image"]

        if self.target_side == "right":
            return input_image, target_image
        
        return target_image, input_image # change the order