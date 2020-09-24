from collections.abc import Iterable

from src.environment import TSPEnv
from src.heuristics import nearest_neighbour, insert_heuristic, compute_distance
from src.utils import sample_nodes, get_nodes

from celluloid import Camera
from matplotlib import pyplot as plt
import matplotlib.animation as animation

import torch
import numpy as np
from tqdm import tqdm

def test_metric_by_graph_size(actor, n_space, number_of_graphs=256, window=5000, device='cpu'):
    if not isinstance(n_space, Iterable):
        n_space = [n_space]
    results = []
    for n in tqdm(n_space):
        env = TSPEnv(batch_size=number_of_graphs, n=n)
        distances = env.get_distances()
        nn_len, _ = nearest_neighbour(distances)
        nic_len, _ = insert_heuristic(distances, insert_type='close')
        nir_len, _ = insert_heuristic(distances, insert_type='remote')
        or_len = []
        for i in range(number_of_graphs):
            or_len.append(compute_distance(distances[i, :, :].squeeze(), eps=1e-5))
        or_len = np.array(or_len)
        sample_act_len = compute_actor_distance(actor, env, window, device)
        results.append([nn_len.mean(), \
                        nic_len.mean(), \
                        nir_len.mean(), \
                        or_len.mean(), \
                        sample_act_len.mean()])
    return results
        
def compute_actor_distance(actor, env, window, device='cpu'):
    dots = torch.Tensor(env.get_dots()).to(device)
    permuted_dots = env.permute_dots(dots=dots)
    for t in np.arange(window):
        _, actions = sample_nodes(actor(permuted_dots))
        observations, _, _ = env.step(actions)
        permuted_dots = env.permute_dots(observations, dots)
    return env.get_best_length()
                 
def permuted_dots_sequence(env, actor, T, device='cpu'):
    dots = torch.Tensor(env.get_dots()).to(device).squeeze()
    permuted_dots = env.permute_dots(dots=dots)
    sequence = [torch.Tensor(env.get_permuted_dots())[0]]
    for t in range(T):
        prob_matrix = actor(permuted_dots)
        _, actions = sample_nodes(prob_matrix)
        observations, r, _ = env.step(actions)
        permuted_dots = env.permute_dots(observations, dots).squeeze()
        if r[0] > 0:
            sequence.append(permuted_dots[0])
    return sequence

def get_gif_animation(file_name, seq, interval):
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(xlim=(0, 1), ylim=(0, 1))

    p = seq[-1]
    camera = Camera(fig)

    for i in range(len(seq)):
        if seq[i].shape == (2,):
            lines = seq[i].reshape(-1, 2)
        else:
            lines = seq[i]
        ax.scatter(p[:,0], p[:,1], c='b')
        plt.plot(lines[:,0], lines[:,1], c='b')
        plt.plot([lines[-1,0], lines[0,0]], [lines[-1,1], lines[0,1]],c='b')
        camera.snap()

    ani = camera.animate(interval=interval)
    ani.save(file_name, writer = 'imagemagick')
    
def get_gif_animation_heuristics(file_name, seq, point_seq, intervals):
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(xlim=(0, 1), ylim=(0, 1))

    p = seq[-1]
    camera = Camera(fig)

    for i in range(len(seq)):
        if seq[i].shape == (2,):
            lines = seq[i].reshape(-1, 2)
        else:
            lines = seq[i]
        ax.scatter(p[:,0], p[:,1], c='b')
        plt.plot(lines[:,0], lines[:,1], c='b')
        plt.plot([lines[-1,0], lines[0,0]], [lines[-1,1], lines[0,1]],c='b')
        if i < len(seq) - 1:
            plt.scatter(point_seq[i][0], point_seq[i][1],c='r')
        camera.snap()

    ani = camera.animate(intervals)
    ani.save(file_name, writer = 'imagemagick')