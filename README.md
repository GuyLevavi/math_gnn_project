# Anisotropic GNNs

## Abstract
Graph Neural Networks (GNNs) are a popular class of models used for data which has a non-Euclidean structure.
A popular type of GNNs are Graph Convolutional Networks (GCNs) which use the convolution theorem to define convolutions on graphs in the spectral domain.
Early works used polynomials of the graph Laplacian to define convolutions on graphs.
These models create isotropic filters which have limited expressivity.
In this work, we suggest a new approach to anisotropic convolutions on graphs based on random frames as approximation to eigenbases of the Laplacian.

## Required Packages
Use requirements.txt to install packages with pip (create a virtual env first), as conda won't work.

## Code

* Experiments were done in "math_gnn_project_mnist.ipynb".
* Trained model is in directory "best_results".
