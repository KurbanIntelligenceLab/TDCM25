import os
import shutil
import logging
from datetime import datetime
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics._classification import classification_report

# -------------------------
# Base Trainer (Abstract)
# -------------------------

class BaseTrainer(ABC):
    def __init__(self, config, train_loader, val_loader, id_test_loader=None, ood_test_loader=None):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.id_test_loader = id_test_loader
        self.ood_test_loader = ood_test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_dir = self.create_run_directory(self.config.output_dir)
        self.checkpoint_dir = os.path.join(self.run_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger = self.setup_logging(self.run_dir)
        self.best_val_acc = 0.0
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        try:
            current_script = os.path.abspath(__file__)
            self.copy_script(self.run_dir, current_script)
        except NameError:
            self.logger.warning("Cannot determine __file__; skipping script copy.")

    @staticmethod
    def create_run_directory(base_dir: str) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(base_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    @staticmethod
    def copy_script(run_dir: str, script_path: str) -> None:
        if os.path.exists(script_path):
            shutil.copy(script_path, run_dir)
        else:
            print(f"Script {script_path} not found. Skipping script copy.")

    @staticmethod
    def setup_logging(run_dir: str) -> logging.Logger:
        log_file = os.path.join(run_dir, 'training.log')
        logging.basicConfig(
            filename=log_file,
            filemode='w',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        logger = logging.getLogger()
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logger.addHandler(console)
        return logger

    @abstractmethod
    def process_batch(self, batch) -> tuple[object, torch.Tensor]:
        """Process a batch and return (inputs, labels)"""
        pass

    def train_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_bar = tqdm(self.train_loader,
                         desc=f"Epoch {epoch+1}/{self.config.num_epochs} - Training",
                         leave=False)
        for batch in train_bar:
            inputs, labels = self.process_batch(batch)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            train_bar.set_postfix(loss=f"{running_loss/total:.4f}",
                                  acc=f"{100.0 * correct/total:.2f}%")
        avg_loss = running_loss / len(self.train_loader)
        acc = 100.0 * correct / total
        self.logger.info(f"Epoch [{epoch+1}/{self.config.num_epochs}] - Train Loss: {avg_loss:.4f}, Train Acc: {acc:.2f}%")
        return avg_loss, acc

    def validate_epoch(self, epoch: int):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        val_bar = tqdm(self.val_loader,
                       desc=f"Epoch {epoch+1}/{self.config.num_epochs} - Validation",
                       leave=False)
        with torch.no_grad():
            for batch in val_bar:
                inputs, labels = self.process_batch(batch)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                val_bar.set_postfix(loss=f"{running_loss/total:.4f}",
                                    acc=f"{100.0 * correct/total:.2f}%")
        avg_loss = running_loss / len(self.val_loader)
        acc = 100.0 * correct / total
        self.logger.info(f"Epoch [{epoch+1}/{self.config.num_epochs}] - Val Loss: {avg_loss:.4f}, Val Acc: {acc:.2f}%")
        return avg_loss, acc

    def evaluate(self, loader, description: str):
        self.model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        all_preds = []
        all_labels = []
        eval_bar = tqdm(loader, desc=f"Evaluating {description}", leave=False)
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch in eval_bar:
                inputs, labels = self.process_batch(batch)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                eval_bar.set_postfix(loss=f"{running_loss/total:.4f}",
                                     acc=f"{100.0 * correct/total:.2f}%")
        avg_loss = running_loss / len(loader)
        acc = 100.0 * correct / total
        self.logger.info(f"{description} - Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")
        report = classification_report(all_labels, all_preds,
                                       target_names=[f"Class {i}" for i in range(self.config.num_classes)],
                                       digits=4)
        self.logger.info(f"Classification Report:\n{report}")
        return acc

    def run(self):
        for epoch in range(self.config.num_epochs):
            self.train_epoch(epoch)
            _, val_acc = self.validate_epoch(epoch)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.best_model_path)
                self.logger.info(f"Best model saved with Val Acc: {self.best_val_acc:.2f}%")
        if self.id_test_loader:
            self.evaluate(self.id_test_loader, "ID Test")
        else:
            self.logger.warning("ID Test loader is not available.")
        if self.ood_test_loader:
            self.evaluate(self.ood_test_loader, "OOD Test")
        else:
            self.logger.warning("OOD Test loader is not available.")
        results_path = os.path.join(self.run_dir, "results.txt")
        with open(results_path, "w") as f:
            f.write(f"Best Validation Accuracy: {self.best_val_acc:.2f}%\n")
        self.logger.info(f"Results saved to {results_path}")