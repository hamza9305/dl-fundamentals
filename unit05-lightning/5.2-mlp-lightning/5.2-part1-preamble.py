from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main():

    train_dataset = datasets.MNIST(
        root="./mnist", train=True, transform=transforms.ToTensor(), download=True
    )

    test_dataset = datasets.MNIST(
        root="./mnist", train=False, transform=transforms.ToTensor()
    )

    print(f"The total training examples = {len(train_dataset)}")
    print(f"The total testing examples = {len(test_dataset)}")



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

    from collections import Counter

    train_counter = Counter()
    for images, labels in train_loader:
        train_counter.update(labels.tolist())

    print("\nTraining label distribution:")
    print(sorted(train_counter.items()))

    val_counter = Counter()
    for images, labels in val_loader:
        val_counter.update(labels.tolist())

    print("\nValidation label distribution:")
    print(sorted(val_counter.items()))

    test_counter = Counter()
    for images, labels in test_loader:
        test_counter.update(labels.tolist())

    print("\nTest label distribution:")
    print(sorted(test_counter.items()))

    majority_class = test_counter.most_common(1)[0]
    print("Majority class:", majority_class[0])

    baseline_acc = majority_class[1] / sum(test_counter.values())
    print("Accuracy when always predicting the majority class:")
    print(f"{baseline_acc:.2f} ({baseline_acc * 100:.2f}%)")




    for images, labels in train_loader:
        break

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training images")
    plt.imshow(np.transpose(torchvision.utils.make_grid(
        images[:64],
        padding=2,
        normalize=True),
        (1, 2, 0)))
    plt.show()

if __name__ == "__main__":
    main()