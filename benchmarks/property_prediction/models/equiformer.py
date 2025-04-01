import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
from benchmarks.property_prediction.regression_dataloader import TDCD25RegressionDataModule
from benchmarks.property_prediction.models.equiformer_model.graph_attention_transformer import graph_attention_transformer_l2
from benchmarks.property_prediction.regression_base_trainer import BaseRegressionTrainer
from config import RegressionConfig
# -------------------------
# Equiformer Trainer Subclass
# -------------------------
class EquiformerTrainer(BaseRegressionTrainer):
    @staticmethod
    def get_atom_type_mapping():
        # The model expects: node_atom = node_atom.new_tensor([-1, 0, -1, -1, -1, -1, 1, 2, 3, 4])[node_atom]
        atom_types = {'Ti': 6, 'O': 7}  # Mapping: Ti->6, O->7
        return atom_types, 10  # max_atom_type = 10

    @staticmethod
    def convert_elements_to_indices(elements):
        atom_types, _ = EquiformerTrainer.get_atom_type_mapping()
        return [atom_types.get(elem, 0) for elem in elements]

    @staticmethod
    def validate_node_atom(node_atom, max_atom_type):
        invalid_mask = (node_atom < 0) | (node_atom >= max_atom_type)
        if invalid_mask.sum().item() > 0:
            raise ValueError("Invalid atom types found!")
        return node_atom

    def __init__(self, config, train_loader, val_loader, id_test_loader=None, ood_test_loader=None):
        super().__init__(config, train_loader, val_loader, id_test_loader, ood_test_loader)
        self.logger.info(f"Initializing Equiformer for target: {config.target_label}")
        self.model = graph_attention_transformer_l2(
            irreps_in="2x0e",
            radius=2.0,
            num_basis=2,
            task_mean=0.0,
            task_std=1.0
        ).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        # Get max atom type for validation
        _, self.max_atom_type = self.get_atom_type_mapping()

    def process_batch(self, batch):
        positions = batch['xyz']        # Tensor: [batch, max_atoms, 3]
        elements_list = batch['element']  # List of lists of element symbols
        batch_size = len(elements_list)

        all_positions = []
        all_atom_types = []
        batch_indices = []
        for i in range(batch_size):
            num_atoms = len(elements_list[i])
            current_positions = positions[i, :num_atoms].to(self.device)
            atom_indices = torch.tensor(self.convert_elements_to_indices(elements_list[i]),
                                          dtype=torch.long, device=self.device)
            all_positions.append(current_positions)
            all_atom_types.append(atom_indices)
            batch_indices.append(torch.full((num_atoms,), i, dtype=torch.long, device=self.device))
        # Concatenate over the batch
        all_positions = torch.cat(all_positions, dim=0)
        all_atom_types = torch.cat(all_atom_types, dim=0)
        batch_indices = torch.cat(batch_indices, dim=0)
        # Validate atom types
        all_atom_types = self.validate_node_atom(all_atom_types, self.max_atom_type)
        labels = batch['regression_label'].to(self.device)
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        inputs = {"pos": all_positions, "batch": batch_indices, "node_atom": all_atom_types}
        return inputs, labels

    def forward_pass(self, inputs):
        # Call the model with the expected keyword arguments.
        return self.model(f_in=None, pos=inputs["pos"], batch=inputs["batch"], node_atom=inputs["node_atom"])

# -------------------------
# Main Entry Point for Equiformer
# -------------------------
def main():
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    target_label = "Et"
    config = RegressionConfig(target_label=target_label, output_dir=f"equiformer_runs_{target_label}", num_epochs=10, learning_rate=1e-3)

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

    trainer = EquiformerTrainer(config, train_loader, val_loader, id_test_loader, ood_test_loader)
    trainer.run()

if __name__ == "__main__":
    main()
