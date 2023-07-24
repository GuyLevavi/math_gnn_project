import torch
import torch.nn as nn

from localizing_mask import LocalizingMask


class AnisoConv(nn.Module):
    def __init__(self, n, kernel_size, threshold):
        super(AnisoConv, self).__init__()

        self.n = n
        self.kernel_size = kernel_size
        self.threshold = threshold

        # init parameters
        self.coefficients = nn.Parameter(torch.Tensor(n), requires_grad=True)
        nn.init.normal_(self.coefficients)

        self.localizer = LocalizingMask(kernel_size, threshold)

    def forward(self, L, V, lam, f, R):
        """
        perform a single anisotropic filter convolution with many rotations R
        :param L: (n, n) Tensor - the Laplacian of the graph
        :param V: (n, n) Tensor - matrix with eigenvectors of the Laplacian as columns
        :param lam: (n,) Tensor - vector of L's eigenvalues
        :param f: (n,) Tenor - signal to convolve
        :param R: (m, n, n) Tensor - 3D tensor of eigenspace rotation matrices
        :return: (m, n) - convolved signal of shape
        """
        RV = torch.matmul(R, V)
        # weight the coefficients with the eigenvalues
        F = torch.matmul(torch.matmul(RV, torch.diag(lam * self.coefficients)), RV.transpose(1, 2))
        # localize the filter
        mask = self.localizer(L)
        F_loc = F * mask
        return torch.matmul(F_loc, f)


