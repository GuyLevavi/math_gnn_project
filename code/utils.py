import numpy as np
import torch
import torch.functional as f
import matplotlib.pyplot as plt
import networkx as nx

from collections import Counter
from scipy.stats import ortho_group, special_ortho_group


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


def batch_band_pass(L, a, b):
    """

    :param L: Tensor (Batch, Nodes, Nodes)
    :param a: loc
    :param b: scale
    :return: Tensor (Batch, Nodes, Nodes)
    """
    n = L.size(1)
    eye = torch.eye(n).unsqueeze(0)

    loc = a * eye
    scale = b
    X = (L - loc) / scale

    return torch.inverse(eye + torch.linalg.matrix_power(X, 2))


def entropy(t, dim):
    """
    compute entropy of tensor along axis. Assumes tensor is a probability
    distribution along the appropriate dimension.
    :param t: Tensor
    :param dim: int
    :return: Tensor with the dimension along which the entropy was computed reduced
    """
    surprise = t * torch.log(t)
    ent = -1. * surprise.sum(dim=dim)
    return ent


def batched_index_select(input, dim, index):
    """
    https://discuss.pytorch.org/t/batched-index-select/9115/8
    :param input: B x * x ... x *
    :param dim: 0 < scalar
    :param index: B x M
    :return:
    """
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index).squeeze(dim)


def orthogonal_block_diagonal_from_eigenvalues(eigenvalues, round_precision=3):
    eigenvalues_round = torch.round(eigenvalues, decimals=round_precision)
    cntr = Counter(eigenvalues_round)

    n = len(eigenvalues)
    R = torch.zeros((n, n))
    multiplicities = []
    next_start_from = 0

    for val in torch.unique(eigenvalues_round, sorted=False):
        multiplicity = cntr[val]
        multiplicities.append(multiplicity)
        if multiplicity == 1:
            R[next_start_from, next_start_from] = -1. if np.random.randint(0, 2) == 0 else 1.
        else:
            R[next_start_from: next_start_from + multiplicity,
                next_start_from: next_start_from + multiplicity] = ortho_group.rvs(dim=multiplicity)

        next_start_from += multiplicity

    return R, torch.tensor(multiplicities)


def many_obd_from_eigenvalues(eigvals, m, round_precision=3):
    """
    sample m eigenspace invariant orthogonal matrices
    :param eigvals: (n,) Tensor
    :param m: int - the number of matrices to sample
    :param round_precision: int
    :return: (m, n, n) Tensor
    """
    n = eigvals.size(0)
    R = torch.zeros(size=(m, n, n))
    for i in range(m):
        R[i] = orthogonal_block_diagonal_from_eigenvalues(eigvals, round_precision)
    return R


def edge_gradient(A, x):
    """
    Edge gradient of a function on nodes
    :param A: Adjacency matrix
    :param x: function on nodes
    :return: Gradient of x along edges, edge field in the shape of A
    """""
    I = np.eye(A.shape[0])
    ones = np.ones_like(A)
    # negate diagonal of ones
    np.fill_diagonal(ones, -1)
    return ((A - I) @ np.diag(x) - (ones * x).T) * np.abs(A - I)


def plot(G, pos, node_signal, title='', edge_signal=None, cmap=plt.cm.coolwarm, with_labels=False):
    """
    Plot a graph with node colors and edge colors
    :param G: nx Graph
    :param pos: Position of nodes for plotting
    :param node_signal: signal on nodes
    :param title: Title for plot
    :param edge_signal: signal on edges, given as a matrix in the same shape as A
    :param cmap: Color map
    :param with_labels: Node labels will be shown if True
    :return: 
    """
    vmin = np.min(node_signal)
    vmax = np.max(node_signal)
    vfinal = np.max(np.abs([vmin, vmax]))

    options = {'node_color': node_signal, 'node_size': 150, 'cmap': cmap,
               'vmin': -vfinal, 'vmax': vfinal}

    if edge_signal is not None:
        options['edge_color'] = edge_signal
        options['edge_cmap'] = cmap
        options['edge_vmin'] = np.min(edge_signal)
        options['edge_vmax'] = np.max(edge_signal)
        options['width'] = 8

    nx.draw(G, pos=pos, with_labels=with_labels, **options)
    plt.title(title)
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-vfinal, vmax=vfinal), cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca())
    plt.show()
