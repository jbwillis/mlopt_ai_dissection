import numpy as np
import scipy.io as spio
from mlo.problem import ProblemData


def read_mat(filepath):
    # Load file
    m = spio.loadmat(filepath)

    # Convert matrices
    c = m['c'].T.flatten().astype(float)
    A = m['A'].astype(float).tocsc()
    l = m['l'].T.flatten().astype(float)
    u = m['u'].T.flatten().astype(float)
    if len(m['int_idx']) > 0:
        int_idx = m['int_idx'].T.flatten().astype(int)
    else:
        int_idx = np.array([])

    return ProblemData(c, l, A, u, int_idx)
