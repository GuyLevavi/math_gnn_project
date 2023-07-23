import torch
import torch.nn as nn


class LocalizingMask(nn.Module):
    def __init__(self, degree, threshold):
        super(LocalizingMask, self).__init__()

        self.degree = degree
        self.threshold = threshold

    def forward(self, L, f):
        gL = torch.pow(L, self.degree) * self.strength
        # separate the entries of f
        df = torch.diag(f)
        # multiply gL with df to get the convolution with each nodes delta function
        gL_df = torch.matmul(gL, df)
        # take absolute values and threshold to create mask
        mask = torch.abs(gL_df) > self.threshold
        return mask
