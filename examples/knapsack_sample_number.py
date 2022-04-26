#!/usr/bin/env python
# coding: utf-8

# # MLOPT Knapsack Example

# In[1]:


import numpy as np
import cvxpy as cp
import pandas as pd
import logging

import mlopt
from mlopt.sampling import uniform_sphere_sample
from mlopt.learners import XGBoost
from mlopt.utils import n_features, pandas2array


# ## Generate problem data

# In[2]:


np.random.seed(1)  # Reset random seed for reproducibility

# Variable
n = 10
x = cp.Variable(n, integer=True)

# Cost
c = np.random.rand(n)

# Weights
a = cp.Parameter(n, nonneg=True, name='a')
x_u = cp.Parameter(n, nonneg=True, name='x_u')
b = 0.5 * n


# ## Create optimizer object

# In[3]:


# Problem
cost = - c @ x
constraints = [a @ x <= b,
               0 <= x, x <= x_u]


# Define optimizer
# If you just want to remove too many messages
# change INFO to WARNING
problem = cp.Problem(cp.Minimize(cost), constraints)


# ## Define training and testing parameters

# In[4]:


# Average request
theta_bar = 2 * np.ones(2 * n)
radius = 1.0


def sample(theta_bar, radius, n=100):

    # Sample points from multivariate ball
    ndim = int(len(theta_bar)/2)
    X_a = uniform_sphere_sample(theta_bar[:ndim], radius, n=n)
    X_u = uniform_sphere_sample(theta_bar[ndim:], radius, n=n)

    df = pd.DataFrame({
        'a': list(X_a),
        'x_u': list(X_u)
        })

    return df


# Training and testing data
n_test = 100


theta_test = sample(theta_bar, radius, n=n_test)

# ## Train predictor

# In[5]:

n_train_set = np.array([25, 50, 100, 200, 400, 600,  1000, 1500, 2000])

n_train_accuracy = []
n_train_subopt = []

for n_train in n_train_set:
    theta_train = sample(theta_bar, radius, n=n_train)

    m = mlopt.Optimizer(problem,
                        log_level=logging.INFO)
    m.train(theta_train, learner=mlopt.XGBOOST)
    
    results = m.performance(theta_test)
    print("N_train: ", n_train, "Accuracy: %.2f " % results[0]['accuracy'],  "Subopt: %.2f " % results[0]['avg_subopt'])
    n_train_accuracy.append( results[0]['accuracy'])
    n_train_subopt.append( results[0]['avg_subopt'])
    
    # update data on every iteration in case of failure
    np.savez("knapsack_sample_number_data.npz", n_train_set = n_train_set, n_train_accuracy = n_train_accuracy, n_train_subopt = n_train_subopt)
    
