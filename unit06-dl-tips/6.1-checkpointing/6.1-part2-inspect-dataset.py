from shared_utilities import CustomDataModule
from collections import Counter




def main():

    dm = CustomDataModule()
    dm.setup("train")
    print("Training set size:", len(dm.train_dataset))
    print("Validation set size:", len(dm.val_dataset))
    print("Test set size:", len(dm.test_dataset))

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    train_counter = Counter()
    for features, labels in train_loader:
        train_counter.update(labels.tolist())

    print("\nTraining label distribution:")
    print(sorted(train_counter.items()))

    val_counter = Counter()
    for features, labels in val_loader:
        val_counter.update(labels.tolist())

    print("\nValidation label distribution:")
    print(sorted(val_counter.items()))

    test_counter = Counter()
    for features, labels in test_loader:
        test_counter.update(labels.tolist())

    print("\nTest label distribution:")
    print(sorted(test_counter.items()))

    majority_class = test_counter.most_common(1)[0]
    print("Majority class:", majority_class[0])

    baseline_acc = majority_class[1] / sum(test_counter.values())
    print("Accuracy when always predicting the majority class:")
    print(f"{baseline_acc:.2f} ({baseline_acc * 100:.2f}%)")

if __name__ == "__main__":
    main()
