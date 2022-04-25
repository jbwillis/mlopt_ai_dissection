import numpy as np
import matplotlib.pyplot as plt

data = np.load("knapsack_sample_radius_data.npz")
rset = data["rset"]
r_accuracy = data["r_accuracy"]
    

fig, ax = plt.subplots(1,1)

ax.plot(rset, r_accuracy)
ylim = ax.ylim()

radius = 1.0 # training radius
ax.plot([radius, radius], ylim)

ax.set_xlabel("Sampling radius")
ax.set_ylabel("Test accuracy")