from dataclasses import dataclass, field
from typing import List, Optional
from torchvision import transforms
import torch
@dataclass
class BaseTDCD25Config:
    """
    Base configuration for TDCD25 datasets.
    Contains common settings for all tasks.
    """
    seed: int = 42
    base_dir: str = "sample"
    num_rotations: int = 526  # default rotations applied to images
# -------------------------
# Configuration Classes
# -------------------------
@dataclass
class ClassificationConfig(BaseTDCD25Config):
    num_classes: int = 3
    image_size: int = 224
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    output_dir: str = "benchmarks/phase_classification/results"
    num_epochs: int = 10
    learning_rate: float = 0.001
    ELEMENT_TO_Z: dict = field(default_factory=lambda: {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6,
        'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12,
        'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Ti': 22,
    })

    def __post_init__(self):
        self.classification_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def map_elements_to_z(self, elements_batch: List[List[str]]) -> List[torch.Tensor]:
        z_batch = []
        for elements in elements_batch:
            try:
                z = torch.tensor([self.ELEMENT_TO_Z[elem] for elem in elements],
                                 dtype=torch.long).view(-1, 1)
            except KeyError as e:
                raise ValueError(f"Element symbol {e} not recognized in ELEMENT_TO_Z mapping.")
            z_batch.append(z)
        return z_batch

@dataclass
class DimeNetConfig(ClassificationConfig):
    hidden_channels: int = 3
    out_channels: int = 3
    num_blocks: int = 1
    num_spherical: int = 2
    num_radial: int = 2
    num_bilinear: int = 2
    cutoff: float = 2.0
    max_num_neighbors: int = 2
    envelope_exponent: int = 2
    num_before_skip: int = 1
    num_after_skip: int = 1
    num_output_layers: int = 1
    act: str = 'swish'
    output_initializer: str = 'zeros'


@dataclass
class SchNetConfig(ClassificationConfig):
    hidden_channels: int = 8
    num_filters: int = 64
    num_interactions: int = 3
    num_gaussians: int = 25
    cutoff: float = 8.0
    readout: str = 'add'
    dipole: bool = False
    max_num_neighbors: int = 16
    mean_value: Optional[float] = None
    std_value: Optional[float] = None
    atomref: Optional[torch.Tensor] = None

def map_elements_to_z(elements_batch: List[List[str]]) -> List[torch.Tensor]:
    ELEMENT_TO_Z = {
    'Ti': 22,
    'O': 8
    }
    z_batch = []
    for elements in elements_batch:
        try:
            z = torch.tensor([ELEMENT_TO_Z[elem] for elem in elements],
                             dtype=torch.long).view(-1, 1)
        except KeyError as e:
            raise ValueError(f"Element symbol {e} not recognized in ELEMENT_TO_Z mapping.")
        z_batch.append(z)
    return z_batch

@dataclass
class TorchvisionConfig(ClassificationConfig):
    model_name: str = "vgg19"
    pretrained: bool = True

@dataclass
class RegressionConfig(BaseTDCD25Config):
    target_label: str = "HOMO" 
    output_dir: str = "results"
    num_epochs: int = 10
    learning_rate: float = 1e-3
    label_file_path: str = BaseTDCD25Config.base_dir + "/all_labels.txt"
    image_model_name: str = "vit_b_16"
    image_model_pretrained: bool = True