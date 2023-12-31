import torch
import torch.nn as nn

from einops import rearrange, reduce


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
        self.k_bands = k_bands
        self.m_samples = m_samples

        self.coefficients = nn.Parameter(torch.randn((1, 1, out_channels, in_channels, k_bands * m_samples)))
        self.bias = nn.Parameter(torch.randn((1, 1, 1, out_channels)))  # (b, f, n, cout)

    def forward(self, x, D): #, M):
        """

        :param x: Tensor (Batch, Frame, Nodes, in_channels)
        :param D: Tensor (Batch, Frame, Nodes, k_bands * m_samples)
        :param M: Tensor (Batch, Nodes, Nodes)
        :return: Tensor (Batch, Frame, Nodes, out_channels)
        """
        # perform multichannel convolution
        Dt_x = torch.einsum('bfnm, bfni -> bfmi', D, x)  # (b, f, n, m) x (b, f, n, c_in) -> (b, f, m, c_in)
        Dt_x = rearrange(Dt_x, 'b f m cin -> b f () m cin')  # (b, f, m, c_in) -> (b, f, c_out=1, m, c_in)
        D = rearrange(D, 'b f n m -> b f () n m')  # (b, f, n, m) -> (b, f, c_out=1, n, m)
        # C = torch.diag_embed(self.coefficients)  # (b=1, f=1, c_out, c_in, m) -> (b=1, f=1, c_out, c_in, m, m)
        # C_Dt_x = torch.einsum('bfoimj, bfoji -> bfoim', C, Dt_x)  # (b=1, f=1, c_out, c_in, m, m) x (b, f, c_out=1, m, c_in) -> (b, f, c_out, c_in, m)
        C_Dt_x = self.coefficients * Dt_x.swapaxes(-2, -1)
        D_C_Dt_x = (1 / self.m_samples ** 0.5) * torch.einsum('bfonm, bfoim -> bfoni', D, C_Dt_x)  # (b, f, c_out=1, n, m) x (b, f, c_out, m, c_in) -> (b, f, c_out, n, c_in)
        y = reduce(D_C_Dt_x, 'b f cout n cin -> b f cout n', 'sum')  # (b, f, c_out, n, c_in) -> (b, f, c_out, n)
        out = rearrange(y, 'b f cout n -> b f n cout')  # (b, f, c_out, n) -> (b, f, n, c_out)
        out_b = out + self.bias

        # With mask implementation
        # C = torch.diag_embed(self.coefficients)  # (b=1, f=1, c_out, c_in, m) -> (b=1, f=1, c_out, c_in, m, m)
        # D = rearrange(D, 'b f n m -> b f () () n m')  # (b, f, n, m) -> (b, f, c_out=1, c_in=1, n, m)
        # D_C = torch.einsum('bfoinj, bfoimj -> bfoinm', D, C)  # (b, f, c_out=1, c_in=1, n, m) x (b=1, f=1, c_out, c_in, m, m) -> (b, f, c_out, c_in, n, m)
        # D_C_Dt = torch.einsum('bfoinm, bfoijm -> bfoinj', D_C, D)  # (b, f, o, i, n, m) x (b, f, o, i, n, m) -> (b, f, o, i, n, n)
        # M = rearrange(M, 'b n1 n2 -> b () () () n1 n2')  # (b, n, n) -> (b, f=1, o=1, i=1, n, n)
        # M_D_C_Dt = M * D_C_Dt  # (b, f=1, o=1, i=1, n, n) * (b, f, o, i, n, n) -> (b, f, o, i, n, n)
        # M_D_C_Dt_x = torch.einsum('bfoinj, bfji -> bfoni', M_D_C_Dt, x)  # (b, f, o, i, n, n) x (b, f, n, i) -> (b, f, o, n, i)
        # y = (1 / self.m_samples ** 0.5) * reduce(M_D_C_Dt_x, 'b f cout n cin -> b f cout n', 'mean')  # (b, f, c_out, n, c_in) -> (b, f, c_out, n)
        # out = rearrange(y, 'b f cout n -> b f n cout')  # (b, f, c_out, n) -> (b, f, n, c_out)

        return out_b

    def regularization_term(self):
        diff_norm = torch.linalg.norm(self.coefficients.diff(dim=-1), dim=-1)

        return (1 / (self.k_bands * self.m_samples) ** 0.5) * diff_norm.mean()


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
