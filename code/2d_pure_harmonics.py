import torch
import torch_geometric
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from iso_net import Net
from torch_geometric.nn import ChebConv
from scipy.stats import ortho_group, special_ortho_group
from collections import Counter

# create a dataset of 2d pure harmonic functions on a 2d grid
# first we try a fixed grid size and vary the frequency
# a data element is two grid graphs, each could be a horizontal harmonic or a vertical harmonic
# the label of the data element is whether the two graphs are both horizontal or both vertical (1) or not (0)

grid_size = 8
n = grid_size * grid_size
num_data = 1000
num_train = 800
num_test = 200

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

    edges = list(grid.edges)
    edges.extend([(v, u) for (u, v) in edges])
    edge_index = torch.tensor(edges).t().contiguous()

    # x is two channels, one for each graph
    x = x.transpose(2, 0, 1).reshape(-1, 1)
    x = torch.tensor(x, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, y=torch.tensor(label, dtype=torch.long))
    data.validate(raise_on_error=True)
    data_list.append(data)

# split the dataset into train and test
train_dataset = data_list[:num_train]
test_dataset = data_list[num_train:]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.BCEWithLogitsLoss()


def train(epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    loss_all = 0
    correct = 0
    for data in train_dataset:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y.unsqueeze(0)).sum().item()
        loss = criterion(output.squeeze(0), data.y.unsqueeze(0).float())
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return loss_all / len(train_dataset), correct / len(train_dataset)


def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    correct = 0
    for data in test_dataset:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y.unsqueeze(0)).sum().item()
    return correct / len(test_dataset)


for epoch in range(1, 201):
    train_loss, train_acc = train(epoch)
    test_acc = test()
    print('Epoch: {:03d}, Train Loss: {:.7f}, Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss, train_acc,
                                                                                          test_acc))
