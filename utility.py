from torch import save
from torch import load
from os import path
from acro import q_acrobot
import torch as th
from acrobot_gym import AcrobotEnv
def save_model(model):
    if isinstance(model, q_acrobot):
        return save(model.state_dict(), path.join('./model', 'acrobot.th'))



def load_model():
    r = q_acrobot(hdim=12)
    r.load_state_dict(load(path.join('./model', 'acrobot.th')))
    return r

def render():
    """
    show result, when is_random= True, you don't need to pass expert_q
    """
    env = AcrobotEnv()
    x = env.reset()
    for _ in range(200):
        env.render()
        xp, r, d, info = env.step(env.action_space.sample())  # take a random action
        print(env.state[:2])
        # print(np.arctan2(xp[1],xp[0]),np.arctan2(xp[3],xp[2]))
    env.close()


def render_q(q):
    """
    show result, when is_random= True, you don't need to pass expert_q
    """
    env = AcrobotEnv()
    x = env.reset()
    t = 0
    d = False
    key=0
    for i in range(500):

        env.render()
        if d:
            env.render()
            break
        u = q.control(th.from_numpy(x).float().unsqueeze(0),
                      eps=0)
        xp, r, d, info = env.step(u)  # take action based on network
        x = xp
        t += 1
        print(x)

    print(t)
    env.close()

