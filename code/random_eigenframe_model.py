import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Reduce

from band_pass import BandPass
from random_eigenframe_filter import RandomEigenframeFilter
from utils import entropy_from_logits, batched_index_select


class RandomEigenframeModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_classes, k_bands, m_samples,
                 l_frames):  # , mask_degree, mask_threshold):
        super(RandomEigenframeModel, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.n_classes = n_classes
        self.k_bands = k_bands
        self.m_samples = m_samples
        self.l_frames = l_frames

        scale = 2 / (k_bands - 1)
        self.bp = BandPass(scale=scale)
        # self.localizing_mask = LocalizingMask(degree=mask_degree, threshold=mask_threshold)

        self.hidden_channels.insert(0, in_channels)
        self.conv_layers = nn.ParameterList(
            [RandomEigenframeFilter(in_channels=cin,
                                    out_channels=cout,
                                    k_bands=k_bands,
                                    m_samples=m_samples)
             for cin, cout in zip(self.hidden_channels[:-1], self.hidden_channels[1:])]
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_channels[-1], self.hidden_channels[-1] // 2),
            Reduce('b f n c -> b f c', 'mean'),
            nn.ReLU(),
            nn.Linear(self.hidden_channels[-1] // 2, n_classes),
            nn.LogSoftmax(dim=2)  # should get (Batch, Frame, Class)
        )

    def forward(self, x, L, return_intermediate=False, return_idx=False):
        """

        :param x: Tensor (Batch, Nodes, in_channels), input signals
        :param L: Tensor (Batch, Nodes, Nodes), input Laplacians
        :param return_intermediate: bool. If True return outputs of convolution layers as list
        :param return_idx: bool. If True return idx of best frame.
        :return: Tensor (Batch, Classes)
        """
        # sample random frames
        x, D = self.make_filter_input(x, L)
        intermediate = []

        # perform convolution
        for conv_layer in self.conv_layers:
            x = conv_layer(x, D)
            x = F.tanh(x)
            intermediate.append(x)

        # apply fully connected then reduce Nodes dimension then apply fully connected again
        logits = self.classifier(x)  # (Batch, Frame, Class)

        # entropy
        ent = entropy_from_logits(logits, dim=2)  # (Batch, Frame)

        # keep the frame with minimal entropy for each example in batch
        idx = torch.argmin(ent, dim=1)  # (Batch, Frame)
        out = batched_index_select(input=logits, dim=1, index=idx)  # (Batch, Class)

        if not return_intermediate and not return_idx:
            return out
        else:
            res = {'out': out}
            if return_intermediate:
                res['intermediate'] = intermediate
            if return_idx:
                res['idx'] = idx
            return res

    def make_filter_input(self, x, L):
        """
        return the eigenframes for each input in the batch
        :param x: Tensor (Batch, Nodes, in_channels)
        :param L: Tensor (Batch, Nodes, Nodes)
        :return: (x, D) Tensors (Batch, Frame=1, Nodes, in_channels), (Batch, Frame, Nodes, k_bands * m_samples)
        """
        # add frame dimension to input signal
        x = rearrange(x, 'b n cin -> b () n cin')

        # make frames
        bs, n, _ = L.shape

        # sample random vectors
        w = torch.randn(size=(bs, self.l_frames, n, self.k_bands, self.m_samples),
                        device=next(iter(self.parameters())).device,
                        requires_grad=False)

        # apply band pass
        d = self.bp(L, w)  # Tensor (b, f, n, k, m)

        # combine samples and bands dimensions
        d = rearrange(d, 'b f n k m -> b f n (k m)')

        # normalize along Nodes dimension
        D = F.normalize(d, dim=2)

        # localizing mask
        # M = self.localizing_mask(L)

        return x, D  # , M

    def regularization_term(self):
        return torch.cat([conv_layer.regularization_term().unsqueeze(0) for conv_layer in self.conv_layers]).mean()


if __name__ == '__main__':
    bs = 16
    c_in = 3
    c_hidden = [32, 64]
    kb = 21
    ms = 12
    lf = 4
    n_nodes = 128
    nc = 10

    model = RandomEigenframeModel(c_in, c_hidden, nc, kb, ms, lf)
    x = torch.randn((bs, n_nodes, c_in))
    L = torch.randn((bs, n_nodes, n_nodes))
    L = torch.einsum('bij, bkj -> bik', L, L)
    out = model(x, L)
    print(out)
