import numpy as np
import torch
import torch.nn as nn
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt

from numpy.polynomial.polynomial import Polynomial
from utils import edge_gradient, orthogonal_block_diagonal_from_eigenvalues


class EdgeGradConv(nn.Module):
    def __init__(self, in_channels, out_channels, deg=3, directional=True, rotate_eigenspaces=False):
        """
        :param in_channels: Input channels of the node signal
        :param out_channels: Desired output channels of the node signal, corresponds to number of filters to be applied
        :param deg: Highest degree of the polynomial (not counting degree zero)
        :param directional: If true, filtering is anisotropic via edge gradient of low eigenvectors of the Laplacian
        :param rotate_eigenspaces: If true, random rotation of eigenspaces is applied via random block diagonal matrix,
        in which each block is a random orthogonal matrix, of shape corresponding to the multiplicity of the eigenvalue
        """
        super(EdgeGradConv, self).__init__()

        self.coefficients = nn.Parameter(torch.Tensor(out_channels, in_channels, deg+1), requires_grad=True)
        self.directional = directional
        self.rotate_eigenspaces = rotate_eigenspaces
        self.powers = torch.arange(deg+1)

    def forward(self, G, x):
        """
        :param G: Graph upon which to perform convolution
        :param x: Node signal of the shape (n, in_channels)
        :return: Output signal of the shape (n, out_channels)
        """
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
        c = (eigenvalues.unsqueeze(-1).pow(self.powers).unsqueeze(1).unsqueeze(2) *
             self.coefficients).sum(dim=-1)

        # note - diag(c) @ y = c * y
        # broadcasting of the input signal duplicates the signal in the out_channels dimension
        # summation is done over the in_channels dimension
        if self.rotate_eigenspaces:
            R = orthogonal_block_diagonal_from_eigenvalues(eigenvalues)
            out = eigenvectors @ (R @ (c * (R.T @ (eigenvectors.T @ x).unsqueeze(1)))).sum(dim=-1)
        else:
            out = eigenvectors @ (c * (eigenvectors.T @ x).unsqueeze(1)).sum(dim=-1)

        if self.directional:
            return grad @ out
        else:
            return out




