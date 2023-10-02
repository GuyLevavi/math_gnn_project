import numpy as np
import torch
import torch.nn as nn
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt

from numpy.polynomial.polynomial import Polynomial
from utils import edge_gradient, orthogonal_block_diagonal_from_eigenvalues


class EdgeGradConv(nn.Module):
    def __init__(self, n, deg=3, directional=True, rotate_eigenspaces=False):
        super(EdgeGradConv, self).__init__()

        self.coefficients = nn.Parameter(torch.Tensor(deg+1), requires_grad=True)
        self.directional = directional
        self.rotate_eigenspaces = rotate_eigenspaces
        self.powers = torch.arange(deg+1)
        # if rotate_eigenspaces:
        #     self.R = torch.nn.Parameter(torch.Tensor(n, n), requires_grad=True)

    def forward(self, G, x):
        L = nx.normalized_laplacian_matrix(G)
        L = torch.Tensor(L.todense())
        A = nx.adjacency_matrix(G).todense()
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        grad = None

        if self.directional:
            fiedler_eigenvector = eigenvectors[:, 1]
            grad = edge_gradient(A, np.array(fiedler_eigenvector))
            abs_sum_rows = np.sum(np.abs(grad), axis=1)
            grad = (grad / abs_sum_rows)
            grad = torch.Tensor(grad)

        c = (eigenvalues.unsqueeze(-1).pow(self.powers) * self.coefficients.unsqueeze(0)).sum(dim=1)
        C = torch.diag(c)

        if self.rotate_eigenspaces:
            R = orthogonal_block_diagonal_from_eigenvalues(eigenvalues)
            spectral_filter = eigenvectors @ R @ C @ R.T @ eigenvectors.T
        else:
            spectral_filter = eigenvectors @ C @ eigenvectors.T

        if self.directional:
            return grad @ spectral_filter @ x
        else:
            return spectral_filter @ x





