import numpy as np
import torch
import torch.functional as f

from collections import Counter
from scipy.stats import ortho_group, special_ortho_group


def band_pass(L, a, b):
    """
    :param L: matrix
    :param a: center of the band
    :param b: "width" of the band
    :return: matrix after performing the band-pass (I + ((L - aI) / b)^2)^(-1)
    """
    n = L.size(0)
    X = (L - a * torch.eye(n)) / b
    return torch.linalg.inv(torch.eye(n) + torch.pow(X, 2))


def orthogonal_block_diagonal_from_eigenvalues(eigenvalues, round_precision=3):
    eigenvalues_round = torch.round(eigenvalues, decimals=round_precision)
    cntr = Counter(eigenvalues_round)

    n = len(eigenvalues)
    R = torch.zeros((n, n))
    multiplicities = []
    next_start_from = 0

    for val in torch.unique(eigenvalues_round, sorted=False):
        multiplicity = cntr[val]
        multiplicities.append(multiplicity)
        if multiplicity == 1:
            R[next_start_from, next_start_from] = -1. if np.random.randint(0, 2) == 0 else 1.
        else:
            R[next_start_from: next_start_from + multiplicity,
                next_start_from: next_start_from + multiplicity] = ortho_group.rvs(dim=multiplicity)

        next_start_from += multiplicity

    return R, torch.tensor(multiplicities)


def many_obd_from_eigenvalues(eigvals, m, round_precision=3):
    """
    sample m eigenspace invariant orthogonal matrices
    :param eigvals: (n,) Tensor
    :param m: int - the number of matrices to sample
    :param round_precision: int
    :return: (m, n, n) Tensor
    """
    n = eigvals.size(0)
    R = torch.zeros(size=(m, n, n))
    for i in range(m):
        R[i] = orthogonal_block_diagonal_from_eigenvalues(eigvals, round_precision)
    return R


def edge_gradient(A, x):
    """
    Edge gradient of a function on nodes
    :param A: Adjacency matrix
    :param x: function on nodes
    :return: Gradient of x along edges, edge field in the shape of A
    """""
    I = np.eye(A.shape[0])
    ones = np.ones((A.shape[0], A.shape[0]))
    # negate diagonal of ones
    np.fill_diagonal(ones, -1)
    return ((A - I) @ np.diag(x) - (ones * x).T) * np.abs(A - I)
