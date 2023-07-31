import lightning as L
import torch
from lightning.pytorch.loggers import CSVLogger
from shared_utilities import CustomDataModule, LightningModel
from lightning.pytorch.callbacks import ModelCheckpoint



class PyTorchMLP(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.all_layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_features, 100),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(50, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        logits = self.all_layers(x)
        return logits


def main():
    torch.manual_seed(123)

    dm = CustomDataModule()

    pytorch_model = PyTorchMLP(num_features=100, num_classes=2)

    callbacks = [
        ModelCheckpoint(save_top_k=1, mode="max", monitor="val_acc", save_last=True)
    ]

    lightning_model = LightningModel(model=pytorch_model, learning_rate=0.05)

    trainer = L.Trainer(
        callbacks=callbacks,  # New!!!
        max_epochs=10,
        accelerator="cpu",
        devices="auto",
        logger=CSVLogger(save_dir="logs/", name="my-model"),
        deterministic=True,
    )

    trainer.fit(model=lightning_model, datamodule=dm)


if __name__ == "__main__":
    main()
