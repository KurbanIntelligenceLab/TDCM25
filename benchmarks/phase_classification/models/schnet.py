import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import SchNet
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from config import SchNetConfig, map_elements_to_z
from benchmarks.phase_classification.classification_dataloader import TDCD25ClassificationDataModule
from benchmarks.phase_classification.classification_base_trainer import BaseTrainer
import torch.optim as optim
from torch_geometric.data import Batch
# -------------------------
# Model Definition
# -------------------------
class SchNetClassifier(nn.Module):
    def __init__(self, config: SchNetConfig):
        super(SchNetClassifier, self).__init__()
        self.schnet = SchNet(
            hidden_channels=config.hidden_channels,
            num_filters=config.num_filters,
            num_interactions=config.num_interactions,
            num_gaussians=config.num_gaussians,
            cutoff=config.cutoff,
            readout=config.readout,
            dipole=config.dipole,
            max_num_neighbors=config.max_num_neighbors,
            mean=config.mean_value,
            std=config.std_value,
            atomref=config.atomref
        )
        self.extender = nn.Sequential(
            nn.Linear(1, config.hidden_channels),
            nn.ReLU(),
            nn.Linear(config.hidden_channels, config.num_classes)
        )

    def forward(self, data: Data) -> torch.Tensor:
        z = data.x.squeeze(-1).long()
        pos = data.pos
        batch = data.batch
        x = self.schnet(z=z, pos=pos, batch=batch)
        logits = self.extender(x)
        return logits
# -------------------------
# SchNet Trainer Subclass
# -------------------------
class SchNetTrainer(BaseTrainer):
    def __init__(self, config, train_loader, val_loader, id_test_loader=None, ood_test_loader=None):
        super().__init__(config, train_loader, val_loader, id_test_loader, ood_test_loader)
        self.logger.info("Initializing SchNetClassifier...")
        self.model = SchNetClassifier(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def process_batch(self, batch) -> tuple[Data, torch.Tensor]:
        labels = batch['label'].to(self.device)
        xyz = batch['xyz']
        elements = batch['element']
        z_batch = map_elements_to_z(elements)
        data_list = []
        for i in range(len(xyz)):
            data_list.append(Data(
                x=z_batch[i],
                pos=xyz[i].to(self.device),
                y=labels[i]
            ))
        graph_batch = Batch.from_data_list(data_list).to(self.device)
        return graph_batch, graph_batch.y

# -------------------------
# Main Entry Point
# -------------------------
def main():
    config = SchNetConfig()
    data_module = TDCD25ClassificationDataModule(
        config=config,
        batch_size=64,
        num_workers=0,
        output_dir=config.split_output_dir
    )
    data_module.setup()
    train_loader = data_module.train_dataloader
    val_loader = data_module.val_dataloader
    id_test_loader = data_module.id_test_dataloader
    ood_test_loader = data_module.ood_test_dataloader

    if not train_loader or not val_loader:
        raise RuntimeError("Train or Validation DataLoader is not available.")

    trainer = SchNetTrainer(config, train_loader, val_loader, id_test_loader, ood_test_loader)
    trainer.run()

if __name__ == "__main__":
    main()