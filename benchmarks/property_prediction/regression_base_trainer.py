import os
from abc import ABC, abstractmethod
from datetime import datetime

import torch
from tqdm import tqdm

# -------------------------
# Base Trainer for Regression
# -------------------------
class BaseRegressionTrainer(ABC):
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

        self.best_val_loss = float("inf")
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
            import shutil
            shutil.copy(script_path, run_dir)
        else:
            print(f"Script {script_path} not found. Skipping script copy.")

    @staticmethod
    def setup_logging(run_dir: str):
        import logging
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
    def process_batch(self, batch):
        pass

    @abstractmethod
    def forward_pass(self, inputs):
        pass

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        train_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} - Training", leave=False)
        for batch in train_bar:
            inputs, labels = self.process_batch(batch)
            self.optimizer.zero_grad()
            outputs = self.forward_pass(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            avg_loss = total_loss / total_samples
            train_bar.set_postfix(loss=f"{avg_loss:.4f}")
        return avg_loss

    def validate_epoch(self, epoch: int):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        val_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} - Validation", leave=False)
        with torch.no_grad():
            for batch in val_bar:
                inputs, labels = self.process_batch(batch)
                outputs = self.forward_pass(inputs)
                loss = self.criterion(outputs, labels)
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                avg_loss = total_loss / total_samples
                val_bar.set_postfix(loss=f"{avg_loss:.4f}")
        self.logger.info(f"Epoch {epoch+1}: Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate(self, loader, description: str, epoch: int):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        eval_bar = tqdm(loader, desc=f"Evaluating {description}", leave=False)
        with torch.no_grad():
            for batch in eval_bar:
                inputs, labels = self.process_batch(batch)
                outputs = self.forward_pass(inputs)
                loss = self.criterion(outputs, labels)
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                avg_loss = total_loss / total_samples
                eval_bar.set_postfix(loss=f"{avg_loss:.4f}")
        self.logger.info(f"{description} Loss: {avg_loss:.4f}")
        return avg_loss

    def run(self):
        for epoch in range(self.config.num_epochs):
            self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                self.logger.info(f"Best model saved with Val Loss: {self.best_val_loss:.4f}")
        if self.id_test_loader:
            self.evaluate(self.id_test_loader, "ID Test", self.config.num_epochs)
        if self.ood_test_loader:
            self.evaluate(self.ood_test_loader, "OOD Test", self.config.num_epochs)
