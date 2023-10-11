import math
import torch
import torch.nn as nn

from utils import batch_band_pass
from einops import rearrange


class BandPass(nn.Module):
    def __init__(self, scale):
        """
        Layer which gets a symmetric normalized Laplacian as input
        :param scale: scale parameter for band pass
        """
        super(BandPass, self).__init__()

        k_bands = math.ceil(2 / scale) + 1

        self.band_passes = [lambda L: batch_band_pass(L, i * scale, scale) for i in range(k_bands)]

    def forward(self, L, w):
        """
        :param L: Tensor (Batch, Nodes, Nodes) - symmetric normalized Laplacians
        :param w: Tensor (Batch, Frame, Nodes, m_samples) - random vectors
        :return: Tensor (Batch, Frame, Nodes, k_bands = 2 * scale + 1, m_samples)
        """
        # perform band pass on Laplacians
        bpL = rearrange([bp(L) for bp in self.band_passes],
                        'k b n1 n2 -> b () k n1 n2')

        # apply to random vectors
        d = torch.einsum('bfknr, bfrkm -> bfknm', bpL, w)

        # rearrange and return
        d = rearrange(d, 'b f k n m -> b f n k m')

        return d
