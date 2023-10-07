import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn.conv import ChebConv
import torch.nn.functional as F

from utils import edge_gradient
from band_pass import BandPass


class PRGCLayer(nn.Module):
    def __init__(self, in_channels, out_channels, b, k):
        """
        Multichannel directional convolution filter of the form
            sum_{k = 1}^K p_k(L) R p_k(L) F q(L)
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param b: scale parameter for band pass
        :param k: degree of convolution
        """
        super(PRGCLayer, self).__init__()

        self.in_features = in_channels
        self.out_features = out_channels
        self.b = b
        self.k = k

        self.band_pass = BandPass(b)
        self.conv_params = nn.ParameterList([nn.Parameter(torch.randn(k)) for i in range(2 * b + 1)])

    def forward(self, L, x, R):
        """
        forward pass of layer
        :param L: (n, n) Tensor. Laplacian of the graph
        :param x: (m, n, in_channels) Tensor. Multichannel signal
        :param R: (m, n, n) Tensor. Rotation matrix
        :return: (m, n, out_channels) convolved signal
        """
        # start by applying band passes to the Laplacian
        P = self.band_pass(L)

        # compute powers of the Laplacian
        L_pows = [torch.eye(L.shape[0])]
        for i in range(self.k):
            L_pows.append(L @ L_pows[-1])

        # perform convolution

