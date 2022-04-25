import numpy as np
import matplotlib.pyplot as plt

data = np.load("knapsack_sample_number_data.npz")

n_train_set = data["n_train_set"]
n_train_accuracy = data["n_train_accuracy"]
n_train_subopt = data["n_train_subopt"]    

fig, ax = plt.subplots(1,1)

ax.plot(n_train_set, n_train_accuracy, "-o")
ylim = ax.get_ylim()


ax.set_xlabel("Number of samples")
ax.set_ylabel("Test accuracy")

fig.savefig("knapsack_sample_number_accuracy.png", dpi=600)

fig, ax = plt.subplots(1,1)

ax.plot(n_train_set, n_train_subopt, "-o")
ylim = ax.get_ylim()


ax.set_xlabel("Number of samples")
ax.set_ylabel("Test Average Suboptimality")

fig.savefig("knapsack_sample_number_subopt.png", dpi=600)

plt.show()