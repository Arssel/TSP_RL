from src.architectures import Actor, Critic
from src.environment import TSPEnv_old, TSPEnv_one_batch, TSPEnv
from src.utils import sample_nodes, path_distance

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import numpy as np

def adjust_learning_rate(optimizer, epoch, lr, decay):
    lr_new = lr * (decay ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new

def train_one_batch(actor, critic, device="cpu", batch_size=256, batch_times=40, gamma=0.99, lr=1e-4, epochs=200, T=200, TSP_size=20, n=4, decay=0.99):
    ''' 
    actor - Actor class for action choice
    critic - Critic class for state evaluation
    device - name of device to train on
    batch_size - batch size parameter
    gamma - discounting factor for reward
    lr - learning rate
    epochs - number of epochs to train
    T - number of improvement steps
    TSP_size - the number of points in TSP problem
    n - the number of steps to evaluate the Bellman function
    '''
    criteria = torch.nn.MSELoss() 
    
    env = TSPEnv_one_batch(batch_size=batch_size, t=T, n=TSP_size)

    batch_n_index = np.repeat(np.arange(batch_size), TSP_size) #repeated batch indexes for permutations
    params = list(actor.parameters()) + list(critic.parameters())
    optimizer = optim.Adam(params)

    r_buf = torch.zeros((n,batch_size), dtype=torch.float, requires_grad=False) # reward buffer
    obs_buf = torch.zeros((n,batch_size,TSP_size), dtype=torch.long, requires_grad=False) # observation buffer
    act_buf = torch.zeros((n,batch_size, 2), dtype=torch.long, requires_grad=False) # action_buffer
    lin_pos_buf = torch.zeros((n,batch_size), dtype=torch.long, requires_grad=False)
    
    #writer = SummaryWriter()
    
    global_iteration = 0

    for e in tqdm(range(epochs)):
        
        print("Epoch: ", e)

        adjust_learning_rate(optimizer, e, lr, decay)
        
        for batch_time in range(batch_times):
            t = 0
            observations, dots = env.reset()
            # observations.shape == (batch_size, TSP_size)
            # dots.shape == (batch_size, TSP_size, 2)

            dots = torch.Tensor(dots).to(device)
            total_reward = 0.
            while t < T:
                ts = t
                while t - ts < n and t < T:
                    permuted_dots = dots[batch_n_index, observations.reshape(-1), :].reshape(batch_size, TSP_size, 2)
                    prob_matr = actor(permuted_dots)
                    lin_pos, actions = sample_nodes(prob_matr)
                    act_buf[t-ts, :, :] = torch.tensor(actions)
                    lin_pos_buf[t-ts,:] = lin_pos.squeeze()
                    obs_buf[t-ts, :, :] = torch.tensor(observations)
                    observations, r, done = env.step(actions) 
                    total_reward += r
                    r_buf[t-ts, :] = torch.tensor(r) 
                    t += 1

                permuted_dots = dots[batch_n_index, observations.reshape(-1), :].reshape(batch_size, TSP_size, 2)
                with torch.no_grad():
                    R = critic(permuted_dots)
                i = n - 1
                loss_actor = 0
                loss_critic = 0

                while i >= 0:
                    with torch.no_grad():
                        R = gamma*R + r_buf[i, :].to(device)

                    permuted_dots = dots[batch_n_index, obs_buf[i, :, :].reshape(-1), :]\
                                                                  .reshape(batch_size, TSP_size, 2)
                    state_evaluation = critic(permuted_dots) 
                    with torch.no_grad():
                        delta = (R - state_evaluation)
                    prob_lin_matrix = actor(permuted_dots).view(-1, TSP_size**2)
                    probs = prob_lin_matrix.gather(1, lin_pos_buf[i,:].view(batch_size, 1).to(device))
                    #print(probs)
                    #print(lin_pos_buf[i,:])
                    loss_actor -= (delta.squeeze()*torch.log(probs).squeeze()).sum()
                    loss_critic += criteria(R, state_evaluation)
                    i -= 1

                loss_actor = loss_actor/batch_size/n
                loss_critic = loss_critic/n
                loss = loss_critic + loss_actor
                global_iteration += 1
                #writer.add_scalar('Loss/actor', loss_actor, global_iteration)
                #writer.add_scalar('Loss/critic',loss_critic, global_iteration)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                #print(f"Loss actor: {loss_actor.item()}\nLoss critic: {loss_critic.item()}")
                #print(f"Loss: {loss.item()}")
            print(total_reward.mean())
            #print(prob_matr)
    return env



def train(actor, critic, device="cpu", batch_size=256, batch_times=40, gamma=0.99, lr=1e-4, epochs=200, T=200, TSP_size=20, n=4, decay=0.99):
    ''' 
    actor - Actor class for action choice
    critic - Critic class for state evaluation
    device - name of device to train on
    batch_size - batch size parameter
    gamma - discounting factor for reward
    lr - learning rate
    epochs - number of epochs to train
    T - number of improvement steps
    TSP_size - the number of points in TSP problem
    n - the number of steps to evaluate the Bellman function
    '''
    criteria = torch.nn.MSELoss() 
    
    env = TSPEnv(batch_size=batch_size, t=T, n=TSP_size)

    batch_n_index = np.repeat(np.arange(batch_size), TSP_size) #repeated batch indexes for permutations
    params = list(actor.parameters()) + list(critic.parameters())
    optimizer = optim.Adam(params)

    r_buf = torch.zeros((n,batch_size), dtype=torch.float, requires_grad=False) # reward buffer
    obs_buf = torch.zeros((n,batch_size,TSP_size), dtype=torch.long, requires_grad=False) # observation buffer
    act_buf = torch.zeros((n,batch_size, 2), dtype=torch.long, requires_grad=False) # action_buffer
    lin_pos_buf = torch.zeros((n,batch_size), dtype=torch.long, requires_grad=False)
    
    writer = SummaryWriter()
    
    global_iteration = 0

    for e in tqdm(range(epochs), leave=False):
        
        print("Epoch: ", e)

        adjust_learning_rate(optimizer, e, lr, decay)
        
        for batch_time in range(batch_times):
            t = 0
            observations, dots = env.new_graph()
            # observations.shape == (batch_size, TSP_size)
            # dots.shape == (batch_size, TSP_size, 2)

            dots = torch.Tensor(dots).to(device)
            total_reward = 0.
            while t < T:
                ts = t
                while t - ts < n and t < T:
                    permuted_dots = env.permute_dots(observations, dots)
                    prob_matr = actor(permuted_dots)
                    lin_pos, actions = sample_nodes(prob_matr)
                    act_buf[t-ts, :, :] = torch.tensor(actions)
                    lin_pos_buf[t-ts,:] = lin_pos.squeeze()
                    obs_buf[t-ts, :, :] = torch.tensor(observations)
                    observations, r, done = env.step(actions) 
                    total_reward += r
                    r_buf[t-ts, :] = torch.tensor(r) 
                    t += 1

                permuted_dots = env.permute_dots(observations, dots)
                with torch.no_grad():
                    R = critic(permuted_dots)
                i = n - 1
                loss_actor = 0
                loss_critic = 0

                while i >= 0:
                    with torch.no_grad():
                        R = gamma*R + r_buf[i, :].to(device)

                    permuted_dots = env.permute_dots(obs_buf[i, :, :], dots)
                    state_evaluation = critic(permuted_dots) 
                    with torch.no_grad():
                        delta = (R - state_evaluation)
                    prob_lin_matrix = actor(permuted_dots).view(-1, TSP_size**2)
                    probs = prob_lin_matrix.gather(1, lin_pos_buf[i,:].view(batch_size, 1).to(device))
                    loss_actor -= (delta.squeeze()*torch.log(probs).squeeze()).sum()
                    loss_critic += criteria(R, state_evaluation)
                    i -= 1

                loss_actor = loss_actor/batch_size/n
                loss_critic = loss_critic/n
                writer.add_scalar('Loss/actor', loss_actor, global_iteration)
                writer.add_scalar('Loss/critic',loss_critic, global_iteration)
                loss = loss_critic + loss_actor
                global_iteration += 1
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            writer.add_scalar('Mean total reward', total_reward.mean(), e*batch_times + batch_time)
    return env