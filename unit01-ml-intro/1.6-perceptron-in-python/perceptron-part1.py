import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("perceptron_toydata-truncated.txt", sep="\t")
X_train = df[["x1", "x2"]].values
y_train = df["label"].values

print(f"The shape of X_train {X_train.shape}")
print(f"The shape of y_train {y_train.shape}")
print(f"The count of each class {np.bincount(y_train)}")


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