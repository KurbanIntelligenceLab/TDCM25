import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from benchmarks.phase_classification.classification_dataloader import TDCD25ClassificationDataModule
from benchmarks.phase_classification.classification_base_trainer import BaseTrainer
from config import TorchvisionConfig

# -------------------------
# Torchvision Trainer Subclass
# -------------------------
class TorchvisionTrainer(BaseTrainer):
    def __init__(self, config, train_loader, val_loader, id_test_loader=None, ood_test_loader=None):
        super().__init__(config, train_loader, val_loader, id_test_loader, ood_test_loader)
        self.logger.info("Building model...")
        self.model = self.build_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.best_val_acc = 0.0

    def build_model(self) -> nn.Module:
        try:
            model_class = getattr(models, self.config.model_name)
        except AttributeError:
            raise ValueError(f"Model '{self.config.model_name}' is not available in torchvision.models")
        model = model_class(pretrained=self.config.pretrained)
        if self.config.model_name.startswith('resnet') or self.config.model_name.startswith('wide_resnet'):
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.config.num_classes)
        elif self.config.model_name.startswith('vgg') or self.config.model_name.startswith('alexnet'):
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, self.config.num_classes)
        elif self.config.model_name.startswith('densenet'):
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, self.config.num_classes)
        else:
            raise NotImplementedError(f"Model architecture modification not implemented for '{self.config.model_name}'")
        return model

    def process_batch(self, batch):
        images = batch['image'].to(self.device)
        labels = batch['label'].to(self.device)
        return images, labels

# -------------------------
# Main Entry Point
# -------------------------
def main():
    config = TorchvisionConfig(model_name="vgg19")
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

    trainer = TorchvisionTrainer(config, train_loader, val_loader, id_test_loader, ood_test_loader)
    trainer.run()

if __name__ == "__main__":
    main()