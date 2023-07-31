import lightning as L
import torch
from lightning.pytorch.loggers import CSVLogger
from shared_utilities import CustomDataModule, LightningModel, PyTorchMLP
from lightning.pytorch.tuner import Tuner
import matplotlib.pyplot as plt
import pandas as pd

def main():

    torch.manual_seed(123)
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
    tuner = Tuner(trainer)

    # finds learning rate automatically
    # sets hparams.lr or hparams.learning_rate to that learning rate
    lr_finder = tuner.lr_find(lightning_model, datamodule=dm)

    fig = lr_finder.plot(suggest=True)
    fig.show()

    fig.savefig("lr_suggest.pdf")

    # get suggestion
    new_lr = lr_finder.suggestion()
    print(new_lr)


    # update hparams of the model
    lightning_model.learning_rate = new_lr
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

    plt.savefig("suggest_loss.pdf")

    df_metrics[["train_acc", "val_acc"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="ACC"
    )

    plt.savefig("suggest_acc.pdf")

    plt.show()

if __name__ == "__main__":
    main()
