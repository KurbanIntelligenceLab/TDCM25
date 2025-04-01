import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.nn import DimeNet
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from config import DimeNetConfig
from benchmarks.phase_classification.classification_dataloader import TDCD25ClassificationDataModule
from benchmarks.phase_classification.classification_base_trainer import BaseTrainer

# -------------------------
# Model Definition
# -------------------------
class DimeNetClassifier(nn.Module):
    def __init__(self, config: DimeNetConfig):
        super(DimeNetClassifier, self).__init__()
        self.dimenet = DimeNet(
            hidden_channels=config.hidden_channels,
            out_channels=config.out_channels,
            num_blocks=config.num_blocks,
            num_bilinear=config.num_bilinear,
            num_spherical=config.num_spherical,
            num_radial=config.num_radial,
            cutoff=config.cutoff,
            max_num_neighbors=config.max_num_neighbors,
            envelope_exponent=config.envelope_exponent,
            num_before_skip=config.num_before_skip,
            num_after_skip=config.num_after_skip,
            num_output_layers=config.num_output_layers,
            act=config.act,
            output_initializer=config.output_initializer
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.out_channels, 4),
            nn.ReLU(),
            nn.Linear(4, config.num_classes)
        )

    def forward(self, data: Data) -> torch.Tensor:
        z = data.x.squeeze(-1).long()
        pos = data.pos
        batch = data.batch
        out = self.dimenet(z=z, pos=pos, batch=batch)
        logits = self.classifier(out)
        return logits
    
# -------------------------
# DimeNet Trainer Subclass
# -------------------------
class DimeNetTrainer(BaseTrainer):
    def __init__(self, config, train_loader, val_loader, id_test_loader=None, ood_test_loader=None):
        super().__init__(config, train_loader, val_loader, id_test_loader, ood_test_loader)
        self.logger.info("Initializing DimeNetClassifier...")
        self.model = DimeNetClassifier(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def process_batch(self, batch) -> tuple[Data, torch.Tensor]:
        labels = batch['label'].to(self.device)
        xyz = batch['xyz']
        elements = batch['element']
        z_batch = self.config.map_elements_to_z(elements)
        data_list = []
        for i in range(len(xyz)):
            data_list.append(Data(
                x=z_batch[i],           # [num_atoms, 1]
                pos=xyz[i].to(self.device),
                y=labels[i]
            ))
        graph_batch = Batch.from_data_list(data_list).to(self.device)
        return graph_batch, graph_batch.y

# -------------------------
# Main Entry Point
# -------------------------
def main():
    config = DimeNetConfig()
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

    trainer = DimeNetTrainer(config, train_loader, val_loader, id_test_loader, ood_test_loader)
    trainer.run()

if __name__ == "__main__":
    main()