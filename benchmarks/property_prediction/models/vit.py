import os
from datetime import datetime
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from timm import create_model

from benchmarks.property_prediction.regression_dataloader import TDCD25RegressionDataModule
from benchmarks.property_prediction.regression_base_trainer import BaseRegressionTrainer
from config import RegressionConfig

# -------------------------
# ViT Trainer Subclass
# -------------------------
class ViTTrainer(BaseRegressionTrainer):
    def __init__(self, config, train_loader, val_loader, id_test_loader=None, ood_test_loader=None):
        super().__init__(config, train_loader, val_loader, id_test_loader, ood_test_loader)
        self.logger.info(f"Initializing ViT for target: {config.target_label}")
        # Here we use a smaller ViT variant; adjust model_name as needed.
        self.model = create_model("vit_tiny_patch16_224", pretrained=True)
        # Remove classification head and add regression head.
        self.model.head = nn.Identity()
        self.model.regression_head = nn.Linear(self.model.num_features, 1)
        self.model = self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def process_batch(self, batch):
        images = batch['image'].to(self.device)
        labels = batch['regression_label'].to(self.device)
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        return images, labels

    def forward_pass(self, inputs):
        features = self.model(inputs)
        predictions = self.model.regression_head(features)
        return predictions

# -------------------------
# Main Entry Point for ViT
# -------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    target_label = "HOMO"  # Change target as needed.
    config = RegressionConfig(target_label=target_label, output_dir=f"vit_runs_{target_label}", num_epochs=10, learning_rate=1e-3)

    data_module = TDCD25RegressionDataModule(
        base_dir=config.base_dir,
        label_file_path=config.label_file_path,
        target_label=config.target_label,
        batch_size=64,
        num_workers=0,
        output_dir="splits",
    )
    data_module.setup()

    train_loader = data_module.train_dataloader
    val_loader = data_module.val_dataloader
    id_test_loader = data_module.id_test_dataloader
    ood_test_loader = data_module.ood_test_dataloader

    trainer = ViTTrainer(config, train_loader, val_loader, id_test_loader, ood_test_loader)
    trainer.run()

if __name__ == "__main__":
    main()
