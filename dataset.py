import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ICLEVRDataset(Dataset):
    def __init__(self, data_dir, json_file, objects_file, mode="train"):
        """
        Args:
            data_dir (str): Directory containing the images (used in train mode).
            json_file (str): Path to train.json, test.json, or new_test.json.
            objects_file (str): Path to objects.json to construct label mappings.
            mode (str): 'train' for loading images and labels, 'test' for labels only.
        """
        super().__init__()
        self.data_dir = data_dir
        self.mode = mode

        with open(objects_file, "r") as f:
            self.obj2idx = json.load(f)

        self.num_classes = len(self.obj2idx)
        if self.num_classes != 24:
            raise ValueError(f"Expected 24 classes, found {self.num_classes}.")

        with open(json_file, "r") as f:
            data = json.load(f)

        self.samples = []

        # Train data structure: Dict[filename, List[labels]]
        if self.mode == "train":
            if not isinstance(data, dict):
                raise TypeError("Train JSON must be a dictionary.")
            for img_name, labels in data.items():
                img_path = os.path.join(data_dir, img_name)
                multi_hot = self._to_multi_hot(labels)
                self.samples.append((img_path, multi_hot))

        # Test data structure: List[List[labels]]
        elif self.mode == "test":
            if not isinstance(data, list):
                raise TypeError("Test JSON must be a list.")
            for labels in data:
                multi_hot = self._to_multi_hot(labels)
                self.samples.append(multi_hot)
        else:
            raise ValueError("Mode must be 'train' or 'test'.")

        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def _to_multi_hot(self, labels):
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for label in labels:
            target[self.obj2idx[label]] = 1.0
        return target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.mode == "train":
            img_path, label = self.samples[idx]
            # convert('RGB') ensures consistency if some grayscale images exist
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            return img, label

        return self.samples[idx]


def denormalize(tensor):
    """
    Reverts transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    for saving and visual evaluation.
    """
    return torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)
