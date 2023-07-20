from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from torch.utils.data.dataset import random_split
import torch.nn.functional as F

class PyTorchMLP(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.all_layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_features, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        logits = self.all_layers(x)
        return logits


def compute_accuracy(model, dataloader):

    model = model.eval()

    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):

        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)

        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return correct / total_examples



def main():


    train_dataset = datasets.MNIST(
        root="./mnist", train=True, transform=transforms.ToTensor(), download=True
    )

    test_dataset = datasets.MNIST(
        root="./mnist", train=False, transform=transforms.ToTensor()
    )

    print(f"The total training examples = {len(train_dataset)}")
    print(f"The total testing examples = {len(test_dataset)}")

    torch.manual_seed(1)
    train_dataset, val_dataset = random_split(train_dataset, lengths=[55000, 5000])


    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=64,
        shuffle=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
    )

    torch.manual_seed(1)
    model = PyTorchMLP(num_features=784, num_classes=10)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    num_epochs = 30

    loss_list = []
    train_acc_list, val_acc_list = [], []
    for epoch in range(num_epochs):

        model = model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):

            logits = model(features)

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not batch_idx % 250:
                ### LOGGING
                print(
                    f"Epoch: {epoch + 1:03d}/{num_epochs:03d}"
                    f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                    f" | Train Loss: {loss:.2f}"
                )
            loss_list.append(loss.item())

        train_acc = compute_accuracy(model, train_loader)
        val_acc = compute_accuracy(model, val_loader)
        print(f"Train Acc {train_acc * 100:.2f}% | Val Acc {val_acc * 100:.2f}%")
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)


if __name__ == "__main__":
    main()