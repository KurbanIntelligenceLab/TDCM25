import torch
import torch.nn as nn
import torch.optim as optim

from benchmarks.property_prediction.regression_dataloader import TDCD25RegressionDataModule
from benchmarks.property_prediction.models.faenet_model.model import FAENet
from benchmarks.property_prediction.regression_base_trainer import BaseRegressionTrainer
from config import RegressionConfig



# -------------------------
# FAENet Trainer Subclass
# -------------------------
class FAENetTrainer(BaseRegressionTrainer):
    def __init__(self, config, train_loader, val_loader, id_test_loader=None, ood_test_loader=None):
        super().__init__(config, train_loader, val_loader, id_test_loader, ood_test_loader)
        self.logger.info(f"Initializing FAENet for target: {config.target_label}")
        self.model = FAENet(
            cutoff=6.0,
            hidden_channels=64,
            tag_hidden_channels=0,
            pg_hidden_channels=0,
            phys_embeds=True,
            phys_hidden_channels=16,
            num_interactions=4,
            num_gaussians=10,
            num_filters=16,
            second_layer_MLP=True,
            skip_co="concat",
            mp_type="updownscale_base",
            graph_norm=True,
            complex_mp=False,
            out_dim=1,
        ).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def process_batch(self, batch):
        # Build a list of Data objects for each sample.
        from torch_geometric.data import Data
        data_list = []
        elements_list = batch['element']  # List of element lists
        positions = batch['xyz']          # Tensor: [batch, max_atoms, 3]
        batch_size = len(elements_list)
        for i in range(batch_size):
            num_atoms = len(elements_list[i])
            current_positions = positions[i, :num_atoms].to(self.device)
            # Convert elements to indices using FAENet mapping
            atom_types = {'Ti': 22, 'O': 8}
            atom_indices = torch.tensor([atom_types.get(elem, 0) for elem in elements_list[i]],
                                        dtype=torch.long, device=self.device)
            label = batch['regression_label'][i].unsqueeze(0).to(self.device)
            data = Data(
                atomic_numbers=atom_indices,
                pos=current_positions,
                y=label
            )
            data_list.append(data)
        from torch_geometric.data import Batch
        batched_data = Batch.from_data_list(data_list)
        return batched_data, batched_data.y.unsqueeze(1)

    def forward_pass(self, inputs):
        # For FAENet, call the energy_forward function.
        outputs = self.model.energy_forward(inputs, preproc=True)
        return outputs['energy']

# -------------------------
# Main Entry Point for FAENet
# -------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    target_label = "Eg"  # Example target; adjust as needed.
    config = RegressionConfig(target_label=target_label, output_dir=f"faenet_runs_{target_label}", num_epochs=10, learning_rate=1e-3)

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

    trainer = FAENetTrainer(config, train_loader, val_loader, id_test_loader, ood_test_loader)
    trainer.run()

if __name__ == "__main__":
    main()
