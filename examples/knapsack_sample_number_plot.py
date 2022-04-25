import numpy as np
import matplotlib.pyplot as plt

data = np.load("knapsack_sample_number_data.npz")

n_train_set = data["n_train_set"]
n_train_accuracy = data["n_train_accuracy"]
    

fig, ax = plt.subplots(1,1)

ax.plot(n_train_set, n_train_accuracy)
ylim = ax.ylim()


ax.set_xlabel("Number of samples")
ax.set_ylabel("Test accuracy")