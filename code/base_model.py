import torch
import torch.nn as nn
import numpy as np
import torch_geometric
import utils

from anisotropic_filter import AnisoConv


class BaseModel(nn.Module):
    """
    This model receives as input the graph Laplacian and the signal on the graph and outputs a single number (to be used
    with BCE loss. Make sure the signal and Laplacian are padded with 0 to max_nodes
    """

    def __init__(self, in_channels, out_channels, max_nodes=128, m_sets=8, kernel_size=3, loc_threshold=0.1):
        super(BaseModel, self).__init__()

        self.max_nodes = max_nodes
        self.m_sets = m_sets
        self.fc1 = nn.Linear(max_nodes, max_nodes)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.loc_threshold = loc_threshold

        self.conv1 = AnisoConv(max_nodes, kernel_size, loc_threshold)
        self.fc1 = nn.Sequential(
            nn.Linear(self.max_nodes, 1)
        )

    def forward(self, L, f):
        # decomposition
        lam, V = torch.linalg.eigh(L)
        # sample eigenspace invariant orthonormal matrices
        R = utils.many_obd_from_eigenvalues(lam, self.m_sets)
        # convolve
        f = self.conv1(L, V, lam, f, R)
        # classify
        out = self.fc1(f)
        # we want to output the output with highest confidence
        idx = torch.argmax(torch.abs(out))
        return out[idx]
