import torch
import torch.nn as nn


class LocalizingMask(nn.Module):
    def __init__(self, degree, threshold):
        super(LocalizingMask, self).__init__()

        self.degree = degree
        self.threshold = threshold

    def forward(self, L):
        """
        make a mask that localizes anisotropic filters. for each node v, find a local isotropic filter using a
        polynomial filter. Then in each entry, fill 1 if it is above the threshold in absolute value, else 0. The mask
        should then be multiplied elementwise with the filter we wish to localize before multiplication with the signal.
        :param L: (Batch, Node, Node)
        :return: mask tensor
        """
        gL = torch.linalg.matrix_power(L, self.degree)
        # take absolute values and threshold to create mask
        mask = torch.abs(gL) > self.threshold
        return mask
