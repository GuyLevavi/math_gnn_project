import torch
import torch.nn as nn

from localizing_mask import LocalizingMask


class AnisoConv(nn.Module):
    def __init__(self, n, kernel_size, threshold):
        super(AnisoConv, self).__init__()

        self.n = n
        self.kernel_size = kernel_size
        self.strength = threshold

        self.coefficients = nn.Parameter(torch.Tensor(n), requires_grad=True)
        self.localizer = LocalizingMask(kernel_size, threshold)

    def forward(self, L, V,  f, R):
        """
        perform a single anisotropic filter convolution with many rotations R
        :param L: (n, n) Tensor - the Laplacian of the graph
        :param V: (n, n) Tensor - matrix with eigenvectors of the Laplacian as columns
        :param f: (n,) Tenor - signal to convolve
        :param R: (m, n, n) Tensor - 3D tensor of eigenspace rotation matrices
        :return: (m, n) - convolved signal of shape
        """
        RV = torch.matmul(R, V)
        Ff = torch.matmul(torch.matmul(torch.matmul(RV, torch.diag(self.coefficients)), RV.transpose(1, 2)), f)
        mask = self.localizer(L, f)

