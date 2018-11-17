# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 21:24:32 2018

@author: kashishban
"""

import gym

env=gym.make("CartPole-v1")

for i in range(200):
    obs=env.reset()
    for t in range(500):
        env.render()
        action=env.action_space.sample()
        obs,reward,done,info=env.step(action)
        if done:
            print(t+1)
            break