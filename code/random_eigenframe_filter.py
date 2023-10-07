import torch
import torch.nn as nn

from einops import rearrange


class RandomEigenframeFilter(nn.Module):
    def __init__(self, in_channels, out_channels, k_bands, m_samples):
        """

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param k_bands: number of bands of the Laplacian's spectrum used
        :param m_samples: number of random vectors sampled
        """
        super(RandomEigenframeFilter, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.coefficients = nn.Parameter(torch.randn((1, out_channels, in_channels, k_bands * m_samples)))

    def forward(self, D, x):
        """

        :param D: Tensor (Batch, Nodes, k_bands * m_samples)
        :param x: Tensor (Batch, Nodes, in_channels)
        :return: Tensor (Batch, Nodes, out_channels)
        """
        m = D.shape[2]

        Dt = rearrange(D, 'b n m -> b m n')  # (b, n, m) -> (b, m, n)
        Dt_x = torch.matmul(Dt, x)  # (b, m, n) x (b, n, c_in) -> (b, m, c_in)
        Dt_x = rearrange(Dt_x, 'b m cin -> b () m cin')  # (b, m, c_in) -> (b, 1, m, c_in)
        D = rearrange(D, 'b n m -> b () n m')  # (b, n, m) -> (b, 1, n, m)
        C = torch.diag_embed(self.coefficients)  # (1, c_out, c_in, m) -> (1, c_out, c_in, m, m)
        C_Dt_x = torch.einsum('boimj, bomi -> boij', C, Dt_x)  # (1, c_out, c_in, m, m) x (b, 1, c_in, m) -> (b, c_out, c_in, m)
        D_C_Dt_x = (1 / m ** 0.5) * torch.einsum('bonm, boim -> boni', D, C_Dt_x)  # (b, 1, n, m) x (b, c_out, m, c_in) -> (b, c_out, n, c_in)
        out = torch.sum(D_C_Dt_x, dim=-1)  # (b, c_out, n, c_in) -> (b, c_out, n)
        return rearrange(out, 'b cout n -> b n cout')  # (b, c_out, n) -> (b, n, c_out)


if __name__ == '__main__':
    bs = 16
    c_in = 3
    c_out = 32
    kb = 20
    ms = 12
    n = 128

    ref_layer = RandomEigenframeFilter(c_in, c_out, kb, ms)
    x = torch.randn((bs, n, c_in))
    D = torch.randn((bs, n, kb * ms))
    out = ref_layer(D, x)
    print(out)
