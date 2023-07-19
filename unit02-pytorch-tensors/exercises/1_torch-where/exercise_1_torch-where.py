import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Perceptron:
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = torch.zeros(num_features)
        self.bias = torch.tensor(0.)

    def forward(self, x):
        weighted_sum_z = torch.dot(x, self.weights) + self.bias
        prediction = torch.where(weighted_sum_z > 0., 1., 0.)

        return prediction

    def update(self, x, true_y):
        prediction = self.forward(x)
        error = true_y - prediction

        # update
        self.bias += error
        self.weights += error * x

        return error

def train(model, all_x, all_y, epochs):

    for epoch in range(epochs):
        error_count = 0

        for x, y in zip(all_x, all_y):
            error = model.update(x, y)
            error_count += abs(error)

        print(f"Epoch {epoch + 1} errors {error_count}")

def compute_accuracy(model, all_x, all_y):

    correct = 0.0

    for x, y in zip(all_x, all_y):
        prediction = model.forward(x)
        correct += int(prediction == y)

    return correct / len(all_y)

def plot_boundary(model):

    w1, w2 = model.weights[0], model.weights[1]
    b = model.bias

    x1_min = -20
    x2_min = (-(w1 * x1_min) - b) / w2

    x1_max = 20
    x2_max = (-(w1 * x1_max) - b) / w2

    return x1_min, x1_max, x2_min, x2_max

def main():


    df = pd.read_csv("perceptron_toydata-truncated.txt", sep="\t")

    X_train = df[["x1", "x2"]].values
    y_train = df["label"].values

    print(f"The shape of X_train {X_train.shape}")
    print(f"The shape of y_train {y_train.shape}")
    print(f"The count of each class {np.bincount(y_train)}")

    X_train = torch.from_numpy(X_train)
    X_train = X_train.to(torch.float32)
    y_train = torch.from_numpy(y_train)


    plt.plot(
        X_train[y_train == 0, 0],
        X_train[y_train == 0, 1],
        marker="D",
        markersize=10,
        linestyle="",
        label="Class 0",
    )

    plt.plot(
        X_train[y_train == 1, 0],
        X_train[y_train == 1, 1],
        marker="^",
        markersize=13,
        linestyle="",
        label="Class 1",
    )

    plt.legend(loc=2)

    plt.xlim([-5, 5])
    plt.ylim([-5, 5])

    plt.xlabel("Feature $x_1$", fontsize=12)
    plt.ylabel("Feature $x_2$", fontsize=12)

    plt.grid()
    plt.show()




    ppn = Perceptron(num_features=2)

    print(f"The weights of the perceptron {ppn.weights}")
    print(f"The bias of the perceptron {ppn.bias}")


    # for Verification
    # x = [1.1, 2.1]
    # first_forward = ppn.forward(x)
    # print(f"The output of first forward = {first_forward}")
    #
    # first_update_error = ppn.update(x, true_y=1)
    # print(f"The error of the first update = {first_update_error}")
    #
    # print("Model parameters:")
    # print("  Weights:", ppn.weights)
    # print("  Bias:", ppn.bias)

    train(model=ppn, all_x=X_train, all_y=y_train, epochs=5)
    train_acc = compute_accuracy(ppn, X_train, y_train)
    print(f"The accuracy of training {train_acc}")

    x1_min, x1_max, x2_min, x2_max = plot_boundary(ppn)

    plt.plot(
        X_train[y_train == 0, 0],
        X_train[y_train == 0, 1],
        marker="D",
        markersize=10,
        linestyle="",
        label="Class 0",
    )

    plt.plot(
        X_train[y_train == 1, 0],
        X_train[y_train == 1, 1],
        marker="^",
        markersize=13,
        linestyle="",
        label="Class 1",
    )

    plt.plot([x1_min, x1_max], [x2_min, x2_max], color="k")

    plt.legend(loc=2)

    plt.xlim([-5, 5])
    plt.ylim([-5, 5])

    plt.xlabel("Feature $x_1$", fontsize=12)
    plt.ylabel("Feature $x_2$", fontsize=12)

    plt.grid()
    plt.show()




if __name__ == "__main__":
    main()