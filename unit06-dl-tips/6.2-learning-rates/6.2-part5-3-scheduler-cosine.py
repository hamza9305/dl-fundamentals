import matplotlib.pyplot as plt
import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.loggers import CSVLogger

from shared_utilities import CustomDataModule, PyTorchMLP



def main():

    num_epochs = 100
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs/10)
    lrs = []


    for i in range(num_epochs):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    plt.ylabel("Learning rate")
    plt.xlabel("Epoch")
    plt.plot(lrs)
    #plt.savefig("cosine-restart.pdf")
    plt.show()

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    lrs = []

    for i in range(num_epochs):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    plt.ylabel("Learning rate")
    plt.xlabel("Epoch")
    plt.plot(lrs)
    # plt.savefig("cosine-1cycle-epoch.pdf")
    plt.show()


if __name__ == "__main__":
    main()
