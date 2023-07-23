import torch
import torch.functional as f


def band_pass(L, a, b):
    """
    :param L: matrix
    :param a: center of the band
    :param b: "width" of the band
    :return: matrix after performing the band-pass (I + ((L - aI) / b)^2)^(-1)
    """
    n = L.size(0)
    X = (L - a * torch.eye(n)) / b
    return torch.linalg.inv(torch.eye(n) + torch.pow(X, 2))

