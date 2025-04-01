import os
import random
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from config import BaseTDCD25Config

# -----------------------------------------------------------------------------
# TiO2CaptionDataset: Reads image, XYZ, and caption files for the captioning task.
# -----------------------------------------------------------------------------
class TiO2CaptionDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        temperature_filter: Optional[Callable[[int], bool]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            base_dir (str): Root directory of the dataset.
            temperature_filter (callable, optional): Function to filter temperatures.
            transform (callable, optional): Transformations for images.
        """
        self.base_dir = base_dir
        self.temperature_filter = temperature_filter
        self.transform = transform
        self.data = self._prepare_dataset()

    def _prepare_dataset(self) -> List[Dict[str, Any]]:
        data = []
        phases = ["anatase", "brookite", "rutile"]

        for phase in phases:
            for temp in range(0, 1001, 50):
                if self.temperature_filter and not self.temperature_filter(temp):
                    continue

                temp_dir = os.path.join(self.base_dir, phase, f"{temp}K")
                if not os.path.exists(temp_dir):
                    continue

                images_dir = os.path.join(temp_dir, "images")
                xyz_dir = os.path.join(temp_dir, "xyz")
                text_dir = os.path.join(temp_dir, "text")

                # Process the original configuration
                orig_image = os.path.join(images_dir, "original.png")
                orig_xyz = os.path.join(xyz_dir, "original_structure.xyz")
                orig_text = os.path.join(text_dir, "original.txt")
                if all(os.path.exists(f) for f in [orig_image, orig_xyz, orig_text]):
                    with open(orig_text, "r") as f:
                        caption = f.read().strip()
                    data_entry = {
                        "phase": phase,
                        "temperature": temp,
                        "image_path": orig_image,
                        "xyz_path": orig_xyz,
                        "caption": caption,
                        "is_original": True,
                    }
                    data.append(data_entry)

                # Process rotated configurations (example: rotations 1 to 9)
                for rot_idx in range(1, 10):
                    rot_image = os.path.join(images_dir, f"rot_{rot_idx}.png")
                    rot_xyz = os.path.join(xyz_dir, f"rot_{rot_idx}.xyz")
                    rot_text = os.path.join(text_dir, f"rot_{rot_idx}.txt")
                    if all(os.path.exists(f) for f in [rot_image, rot_xyz, rot_text]):
                        with open(rot_text, "r") as f:
                            caption = f.read().strip()
                        data_entry = {
                            "phase": phase,
                            "temperature": temp,
                            "image_path": rot_image,
                            "xyz_path": rot_xyz,
                            "caption": caption,
                            "is_original": False,
                        }
                        data.append(data_entry)
        print(f"Loaded {len(data)} total samples")
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.data[idx]

        # Load and transform image
        image = Image.open(entry["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load XYZ data and extract elements and coordinates
        with open(entry["xyz_path"], "r") as f:
            xyz_lines = f.readlines()[2:]  # Skip header lines

        elements = []
        coords = []
        for line in xyz_lines:
            parts = line.strip().split()
            if len(parts) == 4:
                elem, x, y, z = parts
                elements.append(elem)
                coords.append([float(x), float(y), float(z)])
        xyz_coords = torch.tensor(coords, dtype=torch.float)

        return {
            "image": image,
            "xyz": xyz_coords,
            "elements": elements,
            "caption": entry["caption"],
            "phase": entry["phase"],
            "temperature": entry["temperature"],
            "is_original": entry["is_original"],
            "image_path": entry["image_path"],
            "xyz_path": entry["xyz_path"],
        }

# -----------------------------------------------------------------------------
# Custom collate function for captioning task.
# -----------------------------------------------------------------------------
def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function that batches images, preserves variable-length XYZ data,
    and aggregates metadata including captions and file paths.
    """
    images = torch.stack([item["image"] for item in batch])
    captions = [item["caption"] for item in batch]
    phases = [item["phase"] for item in batch]
    temperatures = [item["temperature"] for item in batch]
    is_original = [item["is_original"] for item in batch]
    image_paths = [item["image_path"] for item in batch]
    xyz_paths = [item["xyz_path"] for item in batch]
    xyz_coords = [item["xyz"] for item in batch]
    elements = [item["elements"] for item in batch]

    return {
        "image": images,
        "xyz": xyz_coords,
        "elements": elements,
        "caption": captions,
        "phase": phases,
        "temperature": temperatures,
        "is_original": is_original,
        "image_path": image_paths,
        "xyz_path": xyz_paths,
    }

# -----------------------------------------------------------------------------
# TiO2CaptionDataModule: Manages dataset splits and DataLoader creation.
# -----------------------------------------------------------------------------
class TiO2CaptionDataModule:
    def __init__(
        self,
        base_dir: str,
        batch_size: int = 32,
        num_workers: int = 0,
        transform: Optional[Callable] = None,
        temperature_filters: Optional[Dict[str, Callable[[int], bool]]] = None,
    ) -> None:
        """
        Args:
            base_dir (str): Root directory of the dataset.
            batch_size (int): Batch size for DataLoaders.
            num_workers (int): Number of worker threads.
            transform (callable, optional): Image transformations.
            temperature_filters (dict, optional): Dictionary with keys "train_val", "id_test",
                and "ood_test" mapping to functions that filter temperatures.
        """
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        # Default temperature filters if not provided.
        self.temperature_filters = temperature_filters or {
            "train_val": lambda temp: (0 <= temp < 400) or (600 < temp <= 800),
            "id_test": lambda temp: 400 <= temp <= 600,
            "ood_test": lambda temp: 800 < temp <= 1000,
        }
        self.train_dataset: Optional[Subset] = None
        self.val_dataset: Optional[Subset] = None
        self.id_test_dataset: Optional[TiO2CaptionDataset] = None
        self.ood_test_dataset: Optional[TiO2CaptionDataset] = None

    def setup(self) -> None:
        # Build dataset for train/validation split using the train_val filter.
        train_val_dataset = TiO2CaptionDataset(
            base_dir=self.base_dir,
            temperature_filter=self.temperature_filters["train_val"],
            transform=self.transform,
        )
        total_size = len(train_val_dataset)
        train_size = int(0.8 * total_size)
        indices = list(range(total_size))
        random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        self.train_dataset = Subset(train_val_dataset, train_indices)
        self.val_dataset = Subset(train_val_dataset, val_indices)

        # Build ID Test and OOD Test datasets.
        self.id_test_dataset = TiO2CaptionDataset(
            base_dir=self.base_dir,
            temperature_filter=self.temperature_filters["id_test"],
            transform=self.transform,
        )
        self.ood_test_dataset = TiO2CaptionDataset(
            base_dir=self.base_dir,
            temperature_filter=self.temperature_filters["ood_test"],
            transform=self.transform,
        )

    def _create_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
        )

    @property
    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before accessing dataloaders.")
        return self._create_dataloader(self.train_dataset, shuffle=True)

    @property
    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before accessing dataloaders.")
        return self._create_dataloader(self.val_dataset, shuffle=False)

    @property
    def id_test_dataloader(self) -> DataLoader:
        if self.id_test_dataset is None:
            raise RuntimeError("Call setup() before accessing dataloaders.")
        return self._create_dataloader(self.id_test_dataset, shuffle=False)

    @property
    def ood_test_dataloader(self) -> DataLoader:
        if self.ood_test_dataset is None:
            raise RuntimeError("Call setup() before accessing dataloaders.")
        return self._create_dataloader(self.ood_test_dataset, shuffle=False)

# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    BASE_DIR = BaseTDCD25Config.base_dir

    # Initialize and set up the caption data module.
    data_module = TiO2CaptionDataModule(base_dir=BASE_DIR, batch_size=32, num_workers=0)
    data_module.setup()

    train_loader = data_module.train_dataloader
    val_loader = data_module.val_dataloader
    id_test_loader = data_module.id_test_dataloader
    ood_test_loader = data_module.ood_test_dataloader

    # Print a sample batch from the training loader to verify contents.
    for batch in train_loader:
        print("\nSample batch contents:")
        print(f"Images shape: {batch['image'].shape}")
        print(f"Number of captions: {len(batch['caption'])}")
        print(f"Sample caption: {batch['caption'][0]}")
        print(f"Sample image path: {batch['image_path'][0]}")
        print(f"Sample XYZ path: {batch['xyz_path'][0]}")
        print(f"Sample XYZ shape: {batch['xyz'][0].shape}")
        print(f"Sample elements length: {len(batch['elements'][0])}")
        break
