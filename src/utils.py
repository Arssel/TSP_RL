from operator import mul
from functools import reduce
from typing import Iterable, Tuple, List


import torch
import numpy as np
from sklearn.metrics import pairwise_distances


        

def generate_graph(n, batch_size):
    """
    Generate dots and .
    :param n: - matrix size.
    """
    dots = np.random.rand(batch_size, n, 2)
    distances = np.array([pairwise_distances(dots[i, :, :]) for i in range(batch_size)])
    return dots, distances


def path_distance(matrix, path):
    """
    :param matrix: - an NxN matrix representing the TSP graph distances.
    :param path: - a list of N nodes, representing TSP solution.
    """
    batch_size = matrix.shape[0]
    N = matrix.shape[1]
    assert N == path.shape[1], \
        "Number of visited nodes must be equal to matrix shape:" \
        f"Expected {N}, but got {len(path)}."
    assert (path < N).all()
    distance = np.array([0.] * batch_size)
    matrix = matrix.reshape(-1, N**2)
    batch_range = np.arange(batch_size)
    for i in range(1, N):
        distance += matrix[(batch_range, path[:, i-1]*N + path[:,i])]
    distance += matrix[(batch_range, path[:, i] * N + path[:, 0])]
    return distance

def path_distance_new(matrix, path):
    """
    :param matrix: - an NxN matrix representing the TSP graph distances.
    :param path: - a list of N nodes, representing TSP solution.
    """
    batch_size = matrix.shape[0]
    N = path.shape[1]
    num_nodes = matrix.shape[1]
    distance = np.array([0.] * batch_size)
    matrix = matrix.reshape(-1, num_nodes**2)
    batch_range = np.arange(batch_size)
    for i in range(1, N):
        distance += matrix[(batch_range, path[:, i-1]*num_nodes + path[:,i])]
    distance += matrix[(batch_range, path[:, i] * num_nodes + path[:, 0])]
    return distance


def sample_nodes(matr):
    """
    Sample the positions from matrix of probabilities for each element of batch
    """
    # matr - [batch, n, n]
    batch, n, n = matr.shape
    matr = matr.view(batch, -1)  # [batch_size, n ** 2]
    positions = matr.multinomial(1)  # [batch_size, 1]
    i_s = positions // n
    j_s = positions % n
    i_s = i_s.view(-1).tolist()
    j_s = j_s.view(-1).tolist()
    return positions, list(zip(i_s, j_s))

def get_nodes(matr: torch.FloatTensor) -> List[Tuple[int, int]]:
    """
    Given a matrix of weights of shape [batch_size, n, m]
    return a list of length batch_size containing indices (i, j)
    of maximum valued positions.
    :param matr: - float tensor of shape [batch_size, n, m].
    """
    # matr - [batch, n, n]
    max_3_dim = torch.max(matr, dim=2)
    # max_3_dim [batch, n]
    max_2_dim = torch.max(max_3_dim.values, dim=1, keepdim=True)
    # max_2_dim - [batch, 1]
    i_s = max_2_dim.indices.view(-1)
    j_s = torch.gather(max_3_dim.indices, 1, max_2_dim.indices)
    j_s = j_s.view(-1)
    return list(zip(i_s.tolist(), j_s.tolist()))