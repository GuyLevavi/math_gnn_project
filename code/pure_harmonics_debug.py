import torch
import torch_geometric
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn

from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from torch_geometric.data import Data
from scipy.stats import ortho_group
from collections import Counter
from tqdm import tqdm


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


def orthogonal_block_diagonal_from_eigenvalues(eigenvalues, round_precision=2):
    eigenvalues_round = torch.round(eigenvalues, decimals=round_precision).numpy()
    cntr = Counter(eigenvalues_round)

    n = len(eigenvalues)
    R = torch.zeros((n, n))
    multiplicities = []
    next_start_from = 0

    _, idx = np.unique(eigenvalues_round, return_index=True)
    unique_with_order = eigenvalues_round[np.sort(idx)]

    for val in unique_with_order:
        multiplicity = cntr[val]
        multiplicities.append(multiplicity)
        if multiplicity == 1:
            R[next_start_from, next_start_from] = -1. if np.random.randint(0, 2) == 0 else 1.
        else:
            R[next_start_from: next_start_from + multiplicity,
            next_start_from: next_start_from + multiplicity] = torch.tensor(ortho_group.rvs(dim=multiplicity))

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
        R[i] = orthogonal_block_diagonal_from_eigenvalues(eigvals, round_precision)[0]
    return R


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
        :param L: Laplacian of the graph
        :return: mask tensor
        """
        gL = torch.pow(L, self.degree)
        # take absolute values and threshold to create mask
        mask = torch.abs(gL) > self.threshold
        return mask


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
        mask = self.localizer(L).to(next(self.parameters()).device)
        F_loc = F * mask
        return torch.matmul(F_loc, f)


class BaseModel(nn.Module):
    """
    This model receives as input the graph Laplacian and the signal on the graph and outputs a single number (to be used
    with BCE loss. Make sure the signal and Laplacian are padded with 0 to max_nodes
    """

    def __init__(self, in_channels, out_channels, max_nodes=128, m_sets=8, kernel_size=3, loc_threshold=0.1):
        super(BaseModel, self).__init__()

        self.max_nodes = max_nodes
        self.m_sets = m_sets
        self.fc1 = nn.Linear(max_nodes, max_nodes)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.loc_threshold = loc_threshold

        self.conv1 = AnisoConv(max_nodes, kernel_size, loc_threshold)
        self.fc1 = nn.Sequential(
            nn.Linear(self.max_nodes, 1)
        )

    def forward(self, data):
        # code from torch_geometric.laplacian_lambda_max
        edge_weight = data.edge_attr
        if edge_weight is not None and edge_weight.numel() != data.num_edges:
            edge_weight = None

        edge_index, edge_weight = get_laplacian(data.edge_index, edge_weight,
                                                num_nodes=data.num_nodes)

        sparse_L = to_scipy_sparse_matrix(edge_index, edge_weight, data.num_nodes)
        L = torch.tensor(sparse_L.todense())
        # optional - keep as sparse and use scipy sparse operations
        # data is a torch_geometric object of type Data
        # decomposition
        lam, V = torch.linalg.eigh(L)
        # sample eigenspace invariant orthonormal matrices
        R = many_obd_from_eigenvalues(lam, self.m_sets)
        R = R.to(next(self.parameters()).device)
        # convolve
        f = data['x']
        lam = lam.to(next(self.parameters()).device)
        V = V.to(next(self.parameters()).device)
        f = self.conv1(L, V, lam, f, R).squeeze()
        # classify
        out = self.fc1(f)
        # we want to output the output with highest confidence
        idx = torch.argmax(torch.abs(out))
        return out[idx]


# create a dataset of 2d pure harmonic functions on a 2d grid
# first we try a fixed grid size and vary the frequency
# a data element is two grid graphs, each could be a horizontal harmonic or a vertical harmonic
# the label of the data element is whether the two graphs are both horizontal or both vertical (1) or not (0)
def pure_harmonics_2d(grid_size=8, num_data=1000, num_train=800, num_test=200):
    n = grid_size * grid_size

    grid = nx.grid_2d_graph(grid_size, grid_size)
    grid = nx.convert_node_labels_to_integers(grid)
    grid = grid.to_undirected()

    grid2 = grid.copy()
    two_grids = nx.disjoint_union(grid, grid2)

    pos = dict()
    for i in range(2):
        for j in range(grid_size):
            for k in range(grid_size):
                pos[i * n + j * grid_size + k] = (k, -j - i * grid_size)

    # create the dataset
    data_list = []
    for i in range(num_data):
        freq = np.random.randint(1, grid_size // 2)
        sin = np.random.randint(0, 2)

        if sin == 1:
            x = np.sin(np.arange(grid_size) * freq * 2 * np.pi / grid_size)
        else:
            x = np.cos(np.arange(grid_size) * freq * 2 * np.pi / grid_size)

        label = np.random.randint(0, 2)
        if label == 0:
            flip = np.random.randint(0, 2)
            x1 = np.repeat(x.reshape(1, -1), grid_size, axis=0)
            x2 = x1.T
            if flip == 1:
                x1, x2 = x2, x1
            x = np.stack((x1, x2), axis=2)
        else:
            flip = np.random.randint(0, 2)
            x1 = np.repeat(x.reshape(1, -1), grid_size, axis=0)
            x2 = x1
            if flip == 1:
                x1 = x1.T
                x2 = x2.T
            x = np.stack((x1, x2), axis=2)

        edges = list(two_grids.edges)
        edges.extend([(v, u) for (u, v) in edges])
        edge_index = torch.tensor(edges).t().contiguous()

        # x is two channels, one for each graph, fix it
        x = x.transpose(2, 0, 1).reshape(-1, 1)
        x = torch.tensor(x, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=torch.tensor(label, dtype=torch.long))
        data.validate(raise_on_error=True)
        data_list.append(data)

    # split the dataset into train and test
    train_dataset = data_list[:num_train]
    test_dataset = data_list[num_train:]

    return train_dataset, test_dataset


def train(epoch, model, criterion, train_dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    loss_all = 0
    correct = 0
    for data in train_dataset:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output > 0
        correct += pred.eq(data.y.unsqueeze(0)).sum().item()
        loss = criterion(output, data.y.unsqueeze(0).float())
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return loss_all / len(train_dataset), correct / len(train_dataset)


def test(model, test_dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    correct = 0
    for data in test_dataset:
        data = data.to(device)
        output = model(data)
        pred = output > 0
        correct += pred.eq(data.y.unsqueeze(0)).sum().item()
    return correct / len(test_dataset)


if __name__ == '__main__':

    grid_size = 8
    train_dataset, test_dataset = pure_harmonics_2d(grid_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BaseModel(0, 0, max_nodes=2 * grid_size ** 2, m_sets=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(1, 201)):
        train_loss, train_acc = train(epoch, model, criterion, train_dataset)
        test_acc = test(model, test_dataset)
        print('Epoch: {:03d}, Train Loss: {:.7f}, Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                                                              train_acc,
                                                                                              test_acc))
