#!/usr/bin/env python
# -*- coding:utf-8 -*-

import gym
import numpy as np
import torch as th
import torch
import torch.nn as nn
import random
import copy
import matplotlib.pyplot as plt
from carpole_gym import CartPoleEnv
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def render(expert_q, is_random=False):
    '''
    show result, when is_random= True, you don't need to pass expert_q
    '''
    env = gym.make('CartPole-v0')
    x = env.reset()
    if is_random:
        for _ in range(200):
            env.render()
            env.step(env.action_space.sample())  # take a random action
        env.close()
    else:
        for _ in range(200):
            u = expert_q.control(th.from_numpy(x).float().unsqueeze(0),
                                 eps=0)
            env.render()
            xp, _, _, _ = env.step(u)  # take a random action
            x = xp
        env.close()


def rollout(e, q, eps=0.1, T=200):
    traj = []

    x = e.reset()
    x_max = e.x_threshold
    theta_max = e.theta_threshold_radians
    for t in range(T):
        u = q.control(th.from_numpy(x).float().unsqueeze(0),
                      eps=eps)
        # u = u.int().numpy().squeeze()

        '''
        We want to keep the pole in the middle as much as possible, 
        The original reward does not reflect this dependency
        Therefore we redefine the reward function for training process, 
        we want to give the best reward if it stay on (0,_,0,_).
        The further away from the center, the less the reward we give.
        '''
        xp, r, d, info = e.step(u)
        x_, _, theta, _ = xp
        r1 = (x_max - abs(x_)) / x_max - 1
        r2 = (theta_max - abs(theta)) / theta_max - 0.7
        r = r1 + r2

        t = dict(x=x, xp=xp, r=r, u=u, d=d, info=info)
        x = xp
        traj.append(t)
        if d:
            break
    return traj


def process_single_state(state):
    """
    :param state: numpy array of shape (4,)
    :return: tensor of size (1,4)
    """
    a = th.tensor(state)
    a = a.view(1, 4).float()
    return a


class q_t(nn.Module):
    def __init__(s, xdim, udim, hdim=16):
        super().__init__()
        s.xdim, s.udim = xdim, udim
        s.m = nn.Sequential(
            nn.Linear(xdim, hdim),
            nn.ReLU(True),
            nn.Linear(hdim, udim),
        )

    def forward(s, x):
        return s.m(x)

    def control(s, x, eps=0.1):
        # 1. get q values for all controls
        q = s.m(x)
        if np.random.random() < 1 - eps:
            action = q.argmax().item()
        else:
            action = np.random.randint(0, s.udim)
        # eps-greedy strategy to choose control input
        # note that for eps=0 you should return the correct control u
        return action


def sample_mini_batch(ds, batch_size=64):
    '''

    :param ds: dataset
    :param batch_size:
    :return: x numpy array of shape (64,4)
            xp: numpy array of shape (64,4)
            r: numpy array of shape (64,)
            u: numpy array of shape(64,)
    how we sample a mini_batch:
    randomly choose 64(batch_size) (out of size(datatsize)) item in dataset
    in the chosen episode, randomly choose one state, -> (x_t,r,a,x_{t+1})

    '''
    l = len(ds)
    i = 0
    x_list = []
    xp_list = []
    r_list = []
    u_list = []
    while True:
        if i > batch_size - 1:
            break

        idx = random.randint(0, l - 1)
        l_episode = len(ds[idx])
        idx1 = random.randint(0, l_episode - 1)
        dic = ds[idx][idx1]
        if dic['d']:
            continue
        x_list.append(list(dic['x']))
        xp_list.append(list(dic['xp']))
        r_list.append(dic['r'])
        u_list.append(dic['u'])
        i += 1
    x = np.array(x_list)
    xp = np.array(xp_list)
    r = np.array(r_list)
    u = np.array(u_list)
    return x, xp, r, u


def loss(q, ds, q_target):
    # 1. sample mini-batch from datset ds
    # 2. code up dqn with double-q trick
    # 3. return the objective f
    batch_size = 64
    x, xp, r, u = sample_mini_batch(ds, batch_size)
    x = torch.from_numpy(x).float()
    xp = torch.from_numpy(xp).float()
    r = torch.from_numpy(r).float().view(batch_size, 1)
    u = torch.from_numpy(u).view(batch_size, 1)

    prediction = q(x).gather(1, u)
    q_next = torch.max(q_target(xp), 1).values.view(-1, 1)
    target = r + 1 * q_next
    loss = nn.MSELoss()
    f = loss(prediction, target)

    return f


def evaluate(q):
    # 1. create a new environment e
    # 2. run the learnt q network for 100 trajectories on
    # this new environment and report the average undiscounted
    # return of these 100 trajectories
    e = gym.make('CartPole-v1')

    eps = 0
    reward_all = []
    for i in range(2):
        x = e.reset()
        reward = 0
        d = False
        count = 0
        while (d != True) and (count < 200):
            u = q.control(th.from_numpy(x).float().unsqueeze(0),
                          eps=eps)
            xp, r, d, info = e.step(u)
            reward += 1
            x = xp
            count += 1
            if d or count == 200:
                reward_all.append(reward)

    return np.mean(reward_all)


if __name__ == '__main__':
    e = gym.make('CartPole-v1')

    xdim, udim = e.observation_space.shape[0], \
                e.action_space.n

    q = q_t(xdim, udim, 8)
    optim = th.optim.Adam(q.parameters(), lr=1e-2,
                          weight_decay=1e-4)

    ds = []
    q_target = q_t(xdim, udim, 8)

    # collect few random trajectories with
    # eps=1
    loss_list = []
    avg_reward_list = []
    for i in range(1000):
        ds.append(rollout(e, q, eps=1, T=200))
    for i in range(1000):

        q.train()
        q.zero_grad()
        t = rollout(e, q)
        ds.append(t)

        # perform sgd updates on the q network
        # need to call zero grad on q function
        # to clear the gradient buffer

        optim.zero_grad()
        f = loss(q, ds, q_target)
        f.backward()
        optim.step()
        if i % 100 == 99:
            q_target.load_state_dict(q.state_dict())
        loss_list.append(f.item())
        avg_reward = evaluate(q)
        print('average un-discounted reward after {} iteration: {}'.format(i, avg_reward))
        avg_reward_list.append(avg_reward)
    # print('Logging data to plot')
