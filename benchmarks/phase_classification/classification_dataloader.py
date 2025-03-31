import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

# Import the new configuration class
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import ClassificationConfig

# -----------------------------------------------------------------------------
# Global seed setup for reproducibility using config
# -----------------------------------------------------------------------------
# Instantiate the configuration. This sets the seed, base directory, number of rotations, etc.
config = ClassificationConfig()
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------
class TDCD25ClassificationDataset(Dataset):
    """
    PyTorch Dataset for loading TiO2 data with associated image, structure, and text data.
    """
    def __init__(
        self,
        base_dir: str,
        num_rotations: int,
        temperature_filter: Optional[Callable[[int], bool]] = None,
        transform: Optional[Callable] = None
    ) -> None:
        """
        Args:
            base_dir (str): Root directory of the dataset.
            num_rotations (int): Number of rotations (transformed images) to include.
            temperature_filter (callable, optional): Function to filter temperatures.
            transform (callable, optional): Transformations to apply on images.
        """
        self.base_dir = base_dir
        self.num_rotations = num_rotations
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
                    print(f"Temperature directory {temp_dir} does not exist. Skipping.")
                    continue

                images_dir = os.path.join(temp_dir, "images")
                xyz_dir = os.path.join(temp_dir, "xyz")
                text_dir = os.path.join(temp_dir, "text")

                if not (os.path.isdir(images_dir) and os.path.isdir(xyz_dir) and os.path.isdir(text_dir)):
                    print(f"Missing subdirectories in {temp_dir}. Skipping.")
                    continue

                for rotation in range(self.num_rotations):
                    if rotation == 0:
                        image_file = "original.png"
                        xyz_file = "original_structure.xyz"
                        text_file = "original.txt"
                    else:
                        image_file = f"rot_{rotation}.png"
                        xyz_file = f"rot_{rotation}.xyz"
                        text_file = f"rot_{rotation}.txt"

                    entry = {
                        "phase": phase,
                        "temperature": temp,
                        "image_path": os.path.join(images_dir, image_file),
                        "xyz_path": os.path.join(xyz_dir, xyz_file),
                        "text_path": os.path.join(text_dir, text_file),
                        "label": phase  # the phase acts as the class label
                    }
                    data.append(entry)
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.data[idx]

        # Load image and apply transforms if provided
        image = Image.open(entry["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load XYZ data (skipping the first two header lines)
        with open(entry["xyz_path"], "r") as f:
            xyz_lines = f.readlines()[2:]
        element_symbols = []
        xyz_coords = []
        for line in xyz_lines:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            elem, x, y, z = parts
            element_symbols.append(elem)
            xyz_coords.append([float(x), float(y), float(z)])
        xyz_tensor = torch.tensor(xyz_coords, dtype=torch.float)

        # Load text data
        with open(entry["text_path"], "r") as f:
            text_data = f.read()

        # Encode label using a fixed mapping
        label_map = {"anatase": 0, "brookite": 1, "rutile": 2}
        label = label_map[entry["phase"]]

        return {
            "image": image,
            "xyz": xyz_tensor,
            "element": element_symbols,
            "text": text_data,
            "label": label
        }


# -----------------------------------------------------------------------------
# Data Module Class
# -----------------------------------------------------------------------------
class TDCD25ClassificationDataModule:
    """
    Data module for TiO2 classification tasks.
    Handles dataset setup, splitting, saving indices, and DataLoader creation.
    """
    def __init__(
        self,
        config: ClassificationConfig,
        batch_size: int = 64,
        num_workers: int = 0,
        output_dir: str = "splits",
    ) -> None:
        """
        Args:
            config (ClassificationConfig): Configuration for the task.
            batch_size (int): Batch size for DataLoaders.
            num_workers (int): Number of worker threads.
            output_dir (str): Directory to save split indices.
        """
        self.config = config
        self.base_dir = config.base_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.output_dir = output_dir
        self.seed = config.seed
        self.transform = config.classification_transform
        self.num_rotations = config.num_rotations

        # Datasets for different splits
        self.train_dataset: Optional[Subset] = None
        self.val_dataset: Optional[Subset] = None
        self.id_test_dataset: Optional[TDCD25ClassificationDataset] = None
        self.ood_test_dataset: Optional[TDCD25ClassificationDataset] = None

    def setup(self) -> None:
        """
        Set up the datasets for train/validation and test splits.
        """
        # Create train/validation dataset (0-400K and 600-800K)
        train_val_dataset = TDCD25ClassificationDataset(
            base_dir=self.base_dir,
            num_rotations=self.num_rotations,
            temperature_filter=lambda temp: (0 <= temp < 400) or (600 < temp <= 800),
            transform=self.transform
        )
        if not train_val_dataset.data:
            raise ValueError("Train/Validation dataset is empty. Check temperature filters and directory structure.")

        # Split indices for training and validation
        train_indices, val_indices = self._split_indices(train_val_dataset.data, train_ratio=0.8)
        self.train_dataset = Subset(train_val_dataset, train_indices)
        self.val_dataset = Subset(train_val_dataset, val_indices)

        # Create ID test dataset (400K to 600K)
        self.id_test_dataset = TDCD25ClassificationDataset(
            base_dir=self.base_dir,
            num_rotations=self.num_rotations,
            temperature_filter=lambda temp: 400 <= temp <= 600,
            transform=self.transform
        )
        if self.id_test_dataset.data:
            self._save_indices(list(range(len(self.id_test_dataset.data))), "id_test_indices.txt")
            print(f"ID Test set: {len(self.id_test_dataset.data)} samples.")
        else:
            print("Warning: ID Test dataset is empty. Check temperature filters and directory structure.")

        # Create OOD test dataset (850K to 1000K)
        self.ood_test_dataset = TDCD25ClassificationDataset(
            base_dir=self.base_dir,
            num_rotations=self.num_rotations,
            temperature_filter=lambda temp: 800 < temp <= 1000,
            transform=self.transform
        )
        if self.ood_test_dataset.data:
            self._save_indices(list(range(len(self.ood_test_dataset.data))), "ood_test_indices.txt")
            print(f"OOD Test set: {len(self.ood_test_dataset.data)} samples.")
        else:
            print("Warning: OOD Test dataset is empty. Check temperature filters and directory structure.")

    def _split_indices(self, data: List[Any], train_ratio: float) -> Tuple[List[int], List[int]]:
        """
        Shuffle and split indices based on the given training ratio.
        Also saves the indices to disk.

        Returns:
            Tuple containing training indices and validation indices.
        """
        indices = list(range(len(data)))
        random.shuffle(indices)
        split_idx = int(len(indices) * train_ratio)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        self._save_indices(train_indices, "train_indices.txt")
        self._save_indices(val_indices, "val_indices.txt")
        print(f"Split dataset into {len(train_indices)} training and {len(val_indices)} validation samples.")
        return train_indices, val_indices

    def _save_indices(self, indices: List[int], filename: str) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            f.writelines(f"{idx}\n" for idx in indices)

    def set_worker_seed(self, worker_id: int) -> None:
        """
        Sets the seed for a DataLoader worker.
        """
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(worker_seed)
            torch.cuda.manual_seed_all(worker_seed)

    @staticmethod
    def custom_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function to handle batching of dictionaries.
        """
        images = torch.stack([item['image'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        xyz = [item['xyz'] for item in batch]  # List of tensors with variable first dimension
        elements = [item['element'] for item in batch]
        return {
            "image": images,
            "xyz": xyz,
            "element": elements,
            "label": labels
        }

    def _create_dataloader(self, dataset: Any, shuffle: bool) -> DataLoader:
        """
        Helper function to create a DataLoader with the custom collate function and worker seed.
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.custom_collate,
            num_workers=self.num_workers,
            worker_init_fn=lambda worker_id: self.set_worker_seed(worker_id)
        )

    @property
    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return self._create_dataloader(self.train_dataset, shuffle=True)

    @property
    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return self._create_dataloader(self.val_dataset, shuffle=False)

    @property
    def id_test_dataloader(self) -> Optional[DataLoader]:
        if self.id_test_dataset and self.id_test_dataset.data:
            return self._create_dataloader(self.id_test_dataset, shuffle=False)
        return None

    @property
    def ood_test_dataloader(self) -> Optional[DataLoader]:
        if self.ood_test_dataset and self.ood_test_dataset.data:
            return self._create_dataloader(self.ood_test_dataset, shuffle=False)
        return None

    @staticmethod
    def check_phase_distribution(dataset: List[Dict[str, Any]], name: str) -> None:
        """
        Utility function to print the phase distribution of a dataset.
        """
        phase_counts = {"anatase": 0, "brookite": 0, "rutile": 0}
        for entry in dataset:
            phase_counts[entry["phase"]] += 1
        print(f"{name} Phase Distribution: {phase_counts}")


# # -----------------------------------------------------------------------------
# # Example usage
# # -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     # The configuration instance is already created at the top and used here.
#     data_module = TDCD25ClassificationDataModule(
#         config=config,
#         batch_size=64,
#         num_workers=0,
#         output_dir=config.split_output_dir
#     )
#     data_module.setup()

#     # Retrieve DataLoaders
#     train_loader = data_module.train_dataloader
#     val_loader = data_module.val_dataloader
#     id_test_loader = data_module.id_test_dataloader
#     ood_test_loader = data_module.ood_test_dataloader

#     print(f"Train Loader: {len(train_loader)} batches")
#     print(f"Validation Loader: {len(val_loader)} batches")
#     if id_test_loader:
#         print(f"ID Test Loader: {len(id_test_loader)} batches")
#     else:
#         print("ID Test loader is not created due to empty dataset.")
#     if ood_test_loader:
#         print(f"OOD Test Loader: {len(ood_test_loader)} batches")
#     else:
#         print("OOD Test loader is not created due to empty dataset.")

#     # Optionally, check phase distributions for train and validation splits
#     def get_data_entries(subset: Any) -> List[Dict[str, Any]]:
#         if isinstance(subset, Subset):
#             return [subset.dataset.data[idx] for idx in subset.indices]
#         elif hasattr(subset, 'data'):
#             return subset.data
#         return []

#     train_entries = get_data_entries(data_module.train_dataset)
#     val_entries = get_data_entries(data_module.val_dataset)

#     TDCD25ClassificationDataModule.check_phase_distribution(train_entries, "Train")
#     TDCD25ClassificationDataModule.check_phase_distribution(val_entries, "Validation")
