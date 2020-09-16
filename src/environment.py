import io
import pickle
from random import shuffle
from typing import Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import gym
from gym import spaces
from PIL import Image

from src.utils import generate_graph, path_distance

class TSPEnv_one_batch(gym.Env):

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, n = 20, t = 200, action = "2-opt", batch_size = 16):
        """
        :param n: number of nodes in TSP graph.
        :param t: number of steps until finish.
        :param action: - path change action: either "swap" or "2-opt".
        """
        super().__init__()
        assert n > 0
        assert action in ["swap", "2-opt"]
        self._bsz = batch_size
        self._n = n
        self._t = t
        self._cur_step = 0
        self._action = action
        self._observation = [None] * batch_size
        self._best_length = [0.] * batch_size
        self._dots, self._distances = generate_graph(self._n, self._bsz)
        
    def _swap(self, action):
        action = np.sort(action, axis=1)
        for i in range(action.shape[0]):
            tmp = self._observation[i, action[i, 0]] 
            self._observation[i, action[i, 0]] = self._observation[i, action[i, 1]]
            self._observation[i, action[i, 1]] = tmp

    def _reverse(self, action):
        action = np.sort(action, axis=1)
        for i in range(action.shape[0]):
            for j in range((action[i, 1] - action[i, 0])//2+1):
                tmp = self._observation[i, action[i, 0]+j] 
                self._observation[i, action[i, 0]+j] = self._observation[i, action[i, 1]-j] 
                self._observation[i, action[i, 1]-j] = tmp
        
        
    def step(self, actions):
        if self._action == "swap":
            self._swap(actions)
        elif self._action == "2-opt":
            self._reverse(actions)
        else:
            raise NotImplemented(f"Trying to call an unimplemented action {self._action}")
        cur_length = path_distance(self._distances, self._observation)
        reward = self._best_length - np.min([self._best_length, cur_length], axis=0)
        self._best_length = np.min([self._best_length, cur_length], axis=0)
        self._cur_step += 1
        return self._observation, reward, self._cur_step >= self._t

    def get_distances(self):
        return self._distances
    
    def get_best_length(self):
        return self._best_length
    
    def reset(self):
        self._cur_step = 0
        observation = [list(range(self._n))]*self._bsz
        list(map(np.random.shuffle, observation))
        self._observation = np.array(observation, dtype=np.int32)
        self._best_length = path_distance(self._distances, self._observation)
        return self._observation, self._dots

    # TODO: rewrite for multiple graphs in case of batch
    def render(self, mode="rgb_array"):
        allowed_modes = self.metadata["render.modes"]
        assert mode in allowed_modes, f"Mode must be one of the following: {allowed_modes}."
        if mode == "rgb_array":
            nodes = self._observation.tolist()
            nodes.append(nodes[0])
            plt.title(f"Distance = {path_distance(self._distances, self._observation)}")
            plt.scatter(self._dots[:, 0], self._dots[:, 1], marker='o')
            plt.plot(self._dots[nodes, 0], self._dots[nodes, 1])
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            return Image.open(buf)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename) -> "TSPEnv":
        with open(filename, "rb") as f:
            return pickle.load(f)
        
class TSPEnv(gym.Env):

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, n = 20, t = 200, action = "2-opt", batch_size = 16):
        """
        :param n: number of nodes in TSP graph.
        :param t: number of steps until finish.
        :param action: - path change action: either "swap" or "2-opt".
        """
        super().__init__()
        assert n > 0
        assert action in ["swap", "2-opt"]
        self._bsz = batch_size
        self._n = n
        self._batch_n_index = np.repeat(np.arange(batch_size), n)
        self._t = t
        self._cur_step = 0
        self._action = action
        self._observations = [None] * batch_size
        self._best_length = [0.] * batch_size
        self.new_graph()
        self.permute_dots()

    def reset(self, ret=True):
        self._cur_step = 0
        observations = [list(range(self._n))]*self._bsz
        list(map(np.random.shuffle, observations))
        self._observations = np.array(observations, dtype=np.int32)
        self._best_length = path_distance(self._distances, self._observations)
        if ret == True:
            return self._observations, self._dots
    
    def new_graph(self):
        self._dots, self._distances = generate_graph(self._n, self._bsz)
        self.reset(ret=False)
        return self._observations, self._dots
    
    def permute_dots(self, observations=None, dots=None):
        if observations is None and dots is None:
            self._permuted_dots = self._dots[self._batch_n_index, self._observations.reshape(-1), :]\
                                                                    .reshape(self._bsz, self._n, 2)
        elif (not observations is None) and dots is None:
            return self._dots[self._batch_n_index, observations.reshape(-1), :]\
                                                                    .reshape(self._bsz, self._n, 2)
        elif observations is None and (not dots is None):
            return dots[self._batch_n_index, self._observations.reshape(-1), :]\
                                                                    .reshape(self._bsz, self._n, 2)
        else:
            return dots[self._batch_n_index, observations.reshape(-1), :]\
                                                                    .reshape(self._bsz, self._n, 2)
        
        
    def _swap(self, action):
        action = np.sort(action, axis=1)
        for i in range(action.shape[0]):
            tmp = self._observations[i, action[i, 0]] 
            self._observations[i, action[i, 0]] = self._observations[i, action[i, 1]]
            self._observations[i, action[i, 1]] = tmp

    def _reverse(self, action):
        action = np.sort(action, axis=1)
        for i in range(action.shape[0]):
            for j in range((action[i, 1] - action[i, 0])//2+1):
                tmp = self._observations[i, action[i, 0]+j] 
                self._observations[i, action[i, 0]+j] = self._observations[i, action[i, 1]-j] 
                self._observations[i, action[i, 1]-j] = tmp
     
        
    def step(self, actions):
        if self._action == "swap":
            self._swap(actions)
        elif self._action == "2-opt":
            self._reverse(actions)
        else:
            raise NotImplemented(f"Trying to call an unimplemented action {self._action}")
        cur_length = path_distance(self._distances, self._observations)
        reward = self._best_length - np.min([self._best_length, cur_length], axis=0)
        self._best_length = np.min([self._best_length, cur_length], axis=0)
        self._cur_step += 1
        return self._observations, reward, self._cur_step >= self._t
    
    def get_distances(self):
        return self._distances
    
    def get_best_length(self):
        return self._best_length
    
    def get_dots(self):
        return self._dots
    
    def get_permuted_dots(self):
        return self._permuted_dots
    
    def get_observations(self):
        return self._observations
    
    def place_dots_to(self, device):
        return self._dots.to(device)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
            
    def save_graph(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self._dots, f)

    @staticmethod
    def load(filename) -> "TSPEnv":
        with open(filename, "rb") as f:
            return pickle.load(f)