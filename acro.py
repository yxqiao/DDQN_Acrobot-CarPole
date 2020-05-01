
import numpy as np

import torch
import torch.nn as nn
import random

import matplotlib.pyplot as plt

from acrobot_gym import AcrobotEnv
from utility import *


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


class q_acrobot(nn.Module):
    def __init__(self, xdim=6, udim=3, hdim=64):
        super().__init__()
        self.xdim, self.udim = xdim, udim
        self.model = nn.Sequential(
            nn.Linear(xdim, hdim),
            nn.ReLU(True),
            nn.Linear(hdim, udim),
        )

    def forward(self, x):
        return self.model(x)

    def control(self, x, eps=0.1):
        # 1. get q values for all controls
        # if eps=0, it means we get control based on our network
        # if eps=1, it means we get random control
        q = self.model(x)
        if np.random.random() < 1 - eps:
            action = q.argmax().item()
        else:
            action = np.random.randint(0, self.udim)
        return action


def evaluate(q):
    # 1. create a new environment e
    # 2. run the learnt q network for 100 trajectories on
    # this new environment and report the average undiscounted
    # return of these 100 trajectories
    e = AcrobotEnv()

    eps = 0
    reward_all = []
    for i in range(50):
        x = e.reset()
        reward = 0
        d = False
        count = 0
        while (d != True) and (count < 500):
            u = q.control(th.from_numpy(x).float().unsqueeze(0),
                          eps=eps)
            xp, r, d, info = e.step(u)
            reward += 1
            x = xp
            count += 1
            if d or count == 500:
                reward_all.append(reward)
    print(reward_all)

    return np.mean(reward_all)


def rollout(e, q, eps=0.1, T=300):
    traj = []

    x = e.reset()

    for t in range(T):
        u = q.control(th.from_numpy(x).float().unsqueeze(0),
                      eps=eps)
        xp, r, d, info = e.step(u)

        cos_theta1, sin_theta1, cos_theta2, sin_theta2, theta1dot, theta2dot = xp
        r1 = -cos_theta1
        r2 = cos_theta2
        r = r1 + r2
        t = dict(x=x, xp=xp, r=r, u=u, d=d, info=info)
        x = xp
        traj.append(t)
        if d:
            break
    return traj


if __name__ == '__main__':
    e = AcrobotEnv()

    xdim, udim = e.observation_space.shape[0], \
                 e.action_space.n

    # q = q_acrobot(xdim, udim, 12)
    q = load_model()
    optim = th.optim.Adam(q.parameters(), lr=1e-2,
                          weight_decay=1e-4)

    ds = []
    # q_target = q_acrobot(xdim, udim, 12)
    # q_target.load_state_dict(q.state_dict())
    q_target = load_model()

    # collect few random trajectories with
    # eps=1
    loss_list = []

    for i in range(1):
        ds.append(rollout(e, q, eps=1, T=500))
        print('build original dictionary:', i)
    for i in range(1):

        q.train()
        q.zero_grad()
        t = rollout(e, q, 0, 500)
        ds.append(t)

        optim.zero_grad()
        f = loss(q, ds, q_target)
        f.backward()
        optim.step()
        #print(i, len(t))
        if i % 100 == 99:
            q_target.load_state_dict(q.state_dict())
            print('average reward after ',i,' iterations:',evaluate(q))

        loss_list.append(f.item())

    # print('Logging data to plot')
