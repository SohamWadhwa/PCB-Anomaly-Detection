from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from Data.transformations import getImageNetTransforms, getMaskTransforms
from torchvision import transforms
import torch

class MVTecDataset(Dataset):
    def __init__(self, root_dir, split="train", image_size=256, return_mask=False):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.transform = getImageNetTransforms(image_size)
        self.mask_transform = getMaskTransforms(image_size)
        self.return_mask = return_mask

        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        self.defect_types = []
        
        if split == "train":
            train_dir = self.root_dir / "train" / "good"
            self.image_paths = sorted(train_dir.glob("*.png"))
            self.defect_types = ["good"] * len(self.image_paths)
            self.labels = [0] * len(self.image_paths)
            self.mask_paths = [None] * len(self.image_paths)
        elif split == "test":
            test_dir = self.root_dir / "test"

            for defect_dir in sorted(test_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue

                defect_name = defect_dir.name
                for img_path in sorted(defect_dir.glob("*.png")):
                    self.image_paths.append(img_path)
                    self.defect_types.append(defect_name)

                    if defect_name == "good":
                        self.labels.append(0)
                        self.mask_paths.append(None)
                    else:
                        self.labels.append(1)
                        mask_name = img_path.stem + "_mask.png"
                        mask_path = self.root_dir / "ground_truth" / defect_name / mask_name
                        self.mask_paths.append(mask_path if mask_path.exists() else None)  
        else:
            raise ValueError(
                "split must be 'train' or 'test'"
            )
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        sample = {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(img_path),
            "defect_type": self.defect_types[idx],
            "idx": idx
        }

        if self.return_mask:
            mask_path = self.mask_paths[idx]
            if mask_path is not None:
                mask = Image.open(mask_path).convert("L")
                mask = self.mask_transform(mask)
            else:
                mask = torch.zeros((1, self.image_size, self.image_size))

            sample["mask"] = mask

        return sample

