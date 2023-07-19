import torch

class Perceptron:
    def __init__(self):
        self.weights = torch.tensor([2.86, 1.98])
        self.bias = torch.tensor(-3.0)

    def forward(self, x):
        weighted_sum_z = torch.matmul(x, self.weights) + self.bias
        prediction = torch.where(weighted_sum_z > 0, 1.0, 0.0)

        return prediction


def main():
    X_data = torch.tensor([
        [-1.0, -2.0],
        [-3.0, 4.5],
        [5.0, 6.0]
    ])


    ppn = Perceptron()

    for x in X_data:
        print(ppn.forward(x))




if __name__ == "__main__":
    main()