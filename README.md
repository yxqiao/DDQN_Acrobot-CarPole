# DQN 

This project contains the source code for Q-learning with the DDQN trick. We implement the DDQN in two classific model provided by gyms, CartPole and Acrobot. 

The code is testes in the following enviroment.

- OpenAI Gym’s version of CartPole and Acrobot, which is provided in the source code
- Pytorch


# Test result

For each iteration, the mini-batch size I use is 64. I track loss and 
average cumulative reward evaluated on 50 trajectories for each iteration.
 I update target network every 100 iterations. 
 Each time I update the target network, the loss will increase as shown in figure 1. 
 The average reward will increase and become stable over time. 
 The average cumulative reward will converge to 200 eventually.
![average cumulative reward after each iteration](document/img/avg_reward1.png) 

![average cumulative reward after each iteration](document/img/acrobot_final_position.png)

