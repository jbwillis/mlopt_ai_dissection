from .problem import OptimizationProblem
from .utils import cvxpy2data
from .sampling import uniform_sphere_sample
import cvxpy as cvx
import pandas as pd


class Inventory(OptimizationProblem):
    def __init__(self, T, M, K, radius, bin_vars=False):
        self.T = T  # Horizon
        self.M = M  # Maximum ordering capacity
        self.K = K  # Fixed ordering cost
        self.radius = radius  # Radius for sampling

        #
        # Define model in cvxpy
        #
        x = cvx.Variable(T+1)
        u = cvx.Variable(T)
        y = cvx.Variable(T+1)  # Auxiliary y = max(h * x, - p * x)
        if bin_vars:
            v = cvx.Variable(T, integer=True)

        # Define parameters
        x0 = cvx.Parameter(nonneg=True)
        h = cvx.Parameter(nonneg=True)
        p = cvx.Parameter(nonneg=True)
        c = cvx.Parameter(nonneg=True)
        d = cvx.Parameter(T, nonneg=True)
        self.params = {'x0': x0, 'h': h, 'p': p,
                       'c': c, 'd': d}

        # Constaints
        constraints = []
        constraints += [x == x0]
        constraints += [y >= h * x]
        constraints += [y >= -p * x]
        for t in range(T):
            constraints += [x[t+1] == x[t] + u[t] - d[t]]
        constraints += [u >= 0]
        if bin_vars:
            constraints += [u <= M * v]
            constraints += [0 <= v, v <= 1]  # Binary variables
        else:
            constraints += [u <= M]

        # Objective
        cost = cvx.sum_entries(y) + cvx.sum_entries(c * u)
        if bin_vars:
            cost += cvx.sum_entries(K * v)

        # Define problem
        self.problem = cvx.Problem(cvx.Minimize(cost), constraints)

        # Get problem data
        self.data = cvxpy2data(self.problem)

    def populate(self, theta):
        """
        Populate problem using parameter theta.
        """

        # Get parameters from dataframe
        self.params['h'].value = theta["h"]
        self.params['p'].value = theta["p"]
        self.params['c'].value = theta["c"]
        self.params['x0'].value = theta["x0"]
        self.params['d'].value = theta.iloc[:, 4:].values.T.flatten()

        # Get new problem data
        self.data = cvxpy2data(self.problem)

    def sample(self, theta_bar, N=100):

        # Sample points from multivariate ball
        X = uniform_sphere_sample(theta_bar, self.radius, N=N)

        df = pd.DataFrame({'h': X[0, :],
                           'p': X[1, :],
                           'c': X[2, :],
                           'x0': X[3, :]})
        for i in range(self.T):
            df['d%d' % i] = X[3 + i, :]
