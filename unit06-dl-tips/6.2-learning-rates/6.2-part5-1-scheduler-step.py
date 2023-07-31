import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.loggers import CSVLogger
from shared_utilities import CustomDataModule, PyTorchMLP
import matplotlib.pyplot as plt
import pandas as pd

class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model

        self.save_hyperparameters(ignore=["model"])

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)

        loss = F.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("train_loss", loss)
        self.train_acc(predicted_labels, true_labels)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

        return [opt], [sch]


def main():

    L.seed_everything(123)
    dm = CustomDataModule()

    pytorch_model = PyTorchMLP(num_features=100, num_classes=2)
    lightning_model = LightningModel(model=pytorch_model, learning_rate=0.1)

    trainer = L.Trainer(
        max_epochs=100,
        accelerator="cpu",
        devices="auto",
        logger=CSVLogger(save_dir="logs/", name="my-model"),
        deterministic=True,
    )

    # Create a Tuner
    # tuner = Tuner(trainer)

    # finds learning rate automatically
    # sets hparams.lr or hparams.learning_rate to that learning rate
    # lr_finder = tuner.lr_find(lightning_model, datamodule=dm)

    trainer.fit(model=lightning_model, datamodule=dm)

    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)

    df_metrics = pd.DataFrame(aggreg_metrics)
    df_metrics[["train_loss", "val_loss"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="Loss"
    )

    plt.ylim([0.0, 0.9])
    plt.savefig("step_loss.pdf")

    df_metrics[["train_acc", "val_acc"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="ACC"
    )

    plt.ylim([0.7, 1.0])
    plt.savefig("step_acc.pdf")

    plt.show()

    trainer.test(model=lightning_model, datamodule=dm)

    opt = torch.optim.SGD(pytorch_model.parameters(), lr=0.1)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

    lrs = []
    max_epochs = 100

    for epoch in range(max_epochs):
        opt.step()
        lrs.append(opt.param_groups[0]["lr"])
        sch.step()

    plt.plot(range(max_epochs), lrs)
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate')

    # plt.savefig('steps.pdf')
    plt.show()

if __name__ == "__main__":
    main()
