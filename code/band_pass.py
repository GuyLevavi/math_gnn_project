import torch
import torch.nn as nn

from utils import band_pass


class BandPass(nn.Module):
    def __init__(self, scale):
        """
        Layer which gets a symmetric normalized Laplacian as input
        :param scale: scale parameter for band pass
        """
        super(BandPass, self).__init__()

        self.band_passes = [lambda L: band_pass(L, i * scale, scale) for i in range(2 * scale + 1)]

    def forward(self, L):
        """
        :param L: (n, n) Tensor - symmetric normalized Laplacian
        :return: (2 * scale + 1, n, n) Tensor
        """
        return torch.stack([bp(L) for bp in self.band_passes])
