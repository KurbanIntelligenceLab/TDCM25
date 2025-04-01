import os
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

# -----------------------------------------------------------------------------
# Global seed setup for reproducibility
# -----------------------------------------------------------------------------
os.environ["PYTHONHASHSEED"] = "32"
GLOBAL_SEED = 32
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(GLOBAL_SEED)
    torch.cuda.manual_seed_all(GLOBAL_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# Helper: Parse labels from file
# -----------------------------------------------------------------------------
def parse_all_labels(label_file_path: str) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Parse the all_labels.txt file and extract labels for each phase and temperature.

    Args:
        label_file_path (str): Path to the all_labels.txt file.

    Returns:
        dict: Nested dictionary with structure labels[phase][temp] = {label_name: value, ...}
    """
    labels = {}
    current_phase = None
    header_regex = re.compile(r"#\s+(\w+)_TiO2")

    with open(label_file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            header_match = header_regex.match(line)
            if header_match:
                current_phase = header_match.group(1).lower()
                labels[current_phase] = {}
                continue

            # Skip column header lines if any
            if line.startswith("Temp") and current_phase:
                continue

            if current_phase and line[0].isdigit():
                parts = line.split()
                if len(parts) < 8:
                    print(f"Warning: Incomplete data line: {line}")
                    continue

                temp = int(parts[0].replace("K", ""))
                try:
                    homa = float(parts[1])
                    lumo = float(parts[2])
                    eg = float(parts[3])
                    ef = float(parts[4])
                    et = float(parts[5])
                    exact_temp = float(parts[6])
                    per_atom = float(parts[7])
                except ValueError as e:
                    print(f"Error parsing line: {line}\n{e}")
                    continue

                labels[current_phase][temp] = {
                    "HOMO": homa,
                    "LUMO": lumo,
                    "Eg": eg,
                    "Ef": ef,
                    "Et": et,
                    "Exact_Temp": exact_temp,
                    "per_atom": per_atom,
                }
    return labels

# -----------------------------------------------------------------------------
# Default Image Transformation (e.g., for ResNet)
# -----------------------------------------------------------------------------
DEFAULT_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

# -----------------------------------------------------------------------------
# Regression Dataset Class
# -----------------------------------------------------------------------------
class TDCD25RegressionDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        label_file_path: str,
        target_label: str,
        temperature_filter: Optional[Callable[[int], bool]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Dataset for TDCD25 regression tasks.

        Args:
            base_dir (str): Root directory of the dataset.
            label_file_path (str): Path to the all_labels.txt file.
            target_label (str): Regression target label (e.g., 'HOMO', 'LUMO', etc.).
            temperature_filter (callable, optional): Function to filter temperatures.
            transform (callable, optional): Transformations to apply on images.
        """
        self.base_dir = base_dir
        self.target_label = target_label
        self.temperature_filter = temperature_filter
        self.transform = transform
        self.labels = parse_all_labels(label_file_path)
        self.data = self._prepare_dataset()
        self.label_mean, self.label_std = self._compute_label_statistics()

    def _prepare_dataset(self) -> List[Dict[str, Any]]:
        data = []
        phases = ["anatase", "brookite", "rutile"]

        for phase in phases:
            if phase not in self.labels:
                print(f"No labels found for phase '{phase}'. Skipping.")
                continue

            for temp in range(0, 1001, 50):
                if self.temperature_filter and not self.temperature_filter(temp):
                    continue

                if temp not in self.labels[phase]:
                    print(f"No labels for phase '{phase}' at {temp}K. Skipping.")
                    continue

                temp_dir = os.path.join(self.base_dir, phase, f"{temp}K")
                if not os.path.exists(temp_dir):
                    print(f"Temperature directory '{temp_dir}' does not exist. Skipping.")
                    continue

                images_dir = os.path.join(temp_dir, "images")
                xyz_dir = os.path.join(temp_dir, "xyz")
                text_dir = os.path.join(temp_dir, "text")
                if not (os.path.isdir(images_dir) and os.path.isdir(xyz_dir) and os.path.isdir(text_dir)):
                    print(f"Missing subdirectories in '{temp_dir}'. Skipping.")
                    continue

                for rotation in range(526):  # original + 525 rotations
                    if rotation == 0:
                        image_file = "original.png"
                        xyz_file = "original_structure.xyz"
                        text_file = "original.txt"
                    else:
                        image_file = f"rot_{rotation}.png"
                        xyz_file = f"rot_{rotation}.xyz"
                        text_file = f"rot_{rotation}.txt"

                    image_path = os.path.join(images_dir, image_file)
                    xyz_path = os.path.join(xyz_dir, xyz_file)
                    text_path = os.path.join(text_dir, text_file)

                    if not (os.path.isfile(image_path) and os.path.isfile(xyz_path) and os.path.isfile(text_path)):
                        print(f"Missing files for rotation {rotation} in '{temp_dir}'. Skipping.")
                        continue

                    regression_label = self.labels[phase][temp].get(self.target_label)
                    if regression_label is None:
                        print(f"Label '{self.target_label}' not found for phase '{phase}' at {temp}K. Skipping.")
                        continue

                    data_entry = {
                        "phase": phase,
                        "temperature": temp,
                        "image_path": image_path,
                        "xyz_path": xyz_path,
                        "text_path": text_path,
                        "regression_label": regression_label,
                    }
                    data.append(data_entry)
        return data

    def _compute_label_statistics(self) -> Tuple[float, float]:
        """
        Compute the mean and standard deviation of the target regression labels.
        """
        all_labels = [entry["regression_label"] for entry in self.data]
        label_tensor = torch.tensor(all_labels, dtype=torch.float)
        return label_tensor.mean().item(), label_tensor.std().item()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.data[idx]

        # Load image
        image = Image.open(entry["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load and process XYZ data (skip first two header lines)
        with open(entry["xyz_path"], "r", encoding="utf-8", errors="replace") as f:
            xyz_lines = f.readlines()[2:]
        element_symbols = []
        xyz_coords = []
        for line in xyz_lines:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            elem, x, y, z = parts
            element_symbols.append(elem)
            try:
                xyz_coords.append([float(x), float(y), float(z)])
            except ValueError as e:
                print(f"Warning: Could not parse coordinates in file {entry['xyz_path']}: {e}")
                continue
        xyz_tensor = torch.tensor(xyz_coords, dtype=torch.float)

        # Load text data
        with open(entry["text_path"], "r", encoding="utf-8", errors="replace") as f:
            text_data = f.read()

        regression_label = torch.tensor(entry["regression_label"], dtype=torch.float)
        normalized_label = (regression_label - self.label_mean) / self.label_std  # computed but not returned

        return {
            "image": image,
            "xyz": xyz_tensor,
            "element": element_symbols,
            "text": text_data,
            "regression_label": regression_label,
        }

# -----------------------------------------------------------------------------
# Data Module for Regression
# -----------------------------------------------------------------------------
class TDCD25RegressionDataModule:
    """
    Data module for TDCD25 regression tasks.
    Manages dataset creation, splitting, index saving, and DataLoader generation.
    """
    def __init__(
        self,
        base_dir: str,
        label_file_path: str,
        target_label: str,
        batch_size: int = 64,
        num_workers: int = 0,
        output_dir: str = "splits",
        seed: int = GLOBAL_SEED,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            base_dir (str): Root directory of the dataset.
            label_file_path (str): Path to the all_labels.txt file.
            target_label (str): Regression target label.
            batch_size (int): Batch size for DataLoaders.
            num_workers (int): Number of worker threads.
            output_dir (str): Directory to save split indices.
            seed (int): Random seed.
            transform (callable, optional): Image transformations.
        """
        self.base_dir = base_dir
        self.label_file_path = label_file_path
        self.target_label = target_label
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.output_dir = output_dir
        self.seed = seed
        self.transform = transform or DEFAULT_TRANSFORMS

        self.train_dataset: Optional[Subset] = None
        self.val_dataset: Optional[Subset] = None
        self.id_test_dataset: Optional[TDCD25RegressionDataset] = None
        self.ood_test_dataset: Optional[TDCD25RegressionDataset] = None

    def setup(self) -> None:
        """
        Set up datasets for train/validation and test splits.
        """
        # Train/Validation split: temperatures in [0,400) or (600,800]
        train_val_dataset = TDCD25RegressionDataset(
            base_dir=self.base_dir,
            label_file_path=self.label_file_path,
            target_label=self.target_label,
            temperature_filter=lambda temp: (0 <= temp < 400) or (600 < temp <= 800),
            transform=self.transform,
        )
        if not train_val_dataset.data:
            raise ValueError("Train/Validation dataset is empty. Check filters and directory structure.")

        train_indices, val_indices = self._split_indices(train_val_dataset.data, train_ratio=0.8)
        self.train_dataset = Subset(train_val_dataset, train_indices)
        self.val_dataset = Subset(train_val_dataset, val_indices)

        # ID Test: temperatures in [800,1000]
        self.id_test_dataset = TDCD25RegressionDataset(
            base_dir=self.base_dir,
            label_file_path=self.label_file_path,
            target_label=self.target_label,
            temperature_filter=lambda temp: 800 <= temp <= 1000,
            transform=self.transform,
        )
        if self.id_test_dataset.data:
            self._save_indices(list(range(len(self.id_test_dataset.data))), "id_test_indices.txt")
            print(f"ID Test set: {len(self.id_test_dataset.data)} samples.")
        else:
            print("Warning: ID Test dataset is empty. Check filters and directory structure.")

        # OOD Test: temperatures in [850,1000]
        self.ood_test_dataset = TDCD25RegressionDataset(
            base_dir=self.base_dir,
            label_file_path=self.label_file_path,
            target_label=self.target_label,
            temperature_filter=lambda temp: 850 <= temp <= 1000,
            transform=self.transform,
        )
        if self.ood_test_dataset.data:
            self._save_indices(list(range(len(self.ood_test_dataset.data))), "ood_test_indices.txt")
            print(f"OOD Test set: {len(self.ood_test_dataset.data)} samples.")
        else:
            print("Warning: OOD Test dataset is empty. Check filters and directory structure.")

    def _split_indices(self, data: List[Any], train_ratio: float) -> Tuple[List[int], List[int]]:
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
        Worker initialization function to set seeds for DataLoader workers.
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
        Custom collate function for regression.
        Pads 'xyz' tensors to the maximum number of atoms in the batch.
        """
        images = torch.stack([item["image"] for item in batch])
        regression_labels = torch.stack([item["regression_label"] for item in batch])
        max_atoms = max(item["xyz"].size(0) for item in batch)
        padded_xyz = torch.zeros((len(batch), max_atoms, 3))
        atom_masks = torch.zeros((len(batch), max_atoms))
        for i, item in enumerate(batch):
            num_atoms = item["xyz"].size(0)
            padded_xyz[i, :num_atoms] = item["xyz"]
            atom_masks[i, :num_atoms] = 1
        elements = [item["element"] for item in batch]
        text_data = [item["text"] for item in batch]
        return {
            "image": images,
            "xyz": padded_xyz,
            "atom_mask": atom_masks,
            "element": elements,
            "text": text_data,
            "regression_label": regression_labels,
        }

    def _create_dataloader(self, dataset: Any, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.custom_collate,
            num_workers=self.num_workers,
            worker_init_fn=lambda worker_id: self.set_worker_seed(worker_id),
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
        phase_counts = {"anatase": 0, "brookite": 0, "rutile": 0}
        for entry in dataset:
            phase_counts[entry["phase"]] += 1
        print(f"{name} Phase Distribution: {phase_counts}")

# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    BASE_DIR = "sample"
    LABEL_FILE_PATH = os.path.join(BASE_DIR, "all_labels.txt")
    TARGET_LABEL = "HOMO"  # Options: 'HOMO', 'LUMO', 'Eg', 'Ef', 'Et'

    data_module = TDCD25RegressionDataModule(
        base_dir=BASE_DIR,
        label_file_path=LABEL_FILE_PATH,
        target_label=TARGET_LABEL,
        batch_size=64,
        num_workers=0,
        output_dir="splits",
    )
    data_module.setup()

    train_loader = data_module.train_dataloader
    val_loader = data_module.val_dataloader
    id_test_loader = data_module.id_test_dataloader
    ood_test_loader = data_module.ood_test_dataloader

    print(f"Train Loader: {len(train_loader)} batches")
    print(f"Validation Loader: {len(val_loader)} batches")
    if id_test_loader:
        print(f"ID Test Loader: {len(id_test_loader)} batches")
    else:
        print("ID Test loader is not created due to empty dataset.")
    if ood_test_loader:
        print(f"OOD Test Loader: {len(ood_test_loader)} batches")
    else:
        print("OOD Test loader is not created due to empty dataset.")

    # Optionally, check phase distribution for train/validation splits.
    def get_data_entries(subset: Any) -> List[Dict[str, Any]]:
        if isinstance(subset, Subset):
            return [subset.dataset.data[idx] for idx in subset.indices]
        elif hasattr(subset, "data"):
            return subset.data
        return []

    train_entries = get_data_entries(data_module.train_dataset)
    val_entries = get_data_entries(data_module.val_dataset)
    TDCD25RegressionDataModule.check_phase_distribution(train_entries, "Train")
    TDCD25RegressionDataModule.check_phase_distribution(val_entries, "Validation")
