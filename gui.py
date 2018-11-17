# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 19:36:56 2018

@author: kashishban
"""

import gym
import numpy as np
import math
from collections import deque
from matplotlib import pylab
from pylab import *

n_episodes=1000
n_win_ticks=195
max_env_steps=None

gamma=0.95
epsilon=1.0
epsilon_min=0.01
epsilon_decay=0.995
alpha=0.01
alpha_decay=0.01
batch_size=64
monitor=False
quiet=False
memory=deque(maxlen=100000) 

def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))
        
def choose_action(env,model,state,epsilon):
    return env.action_space.sample() if(np.random.random()<=epsilon) else np.argmax(model.predict(state))

def get_epsilon(t):
    return max(epsilon_min,min(epsilon,1.0-math.log10((t+1)*epsilon_decay)))

def preprocess_state(env,state):
    return np.reshape(state,[1,env.observation_space.shape[0]])

def replay(model,batch_size,epsilon):
    x_batch,y_batch=[],[]
    import random
    minibatch=random.sample(memory,min(len(memory),batch_size))
    
    for state,action,reward,next_state,done in minibatch:
        y_target=model.predict(state)
        y_target[0][action]=reward if done else reward+gamma*np.max(model.predict(next_state)[0])
        x_batch.append(state[0])
        y_batch.append(y_target[0])
    
    model.fit(np.array(x_batch),np.array(y_batch),batch_size=len(x_batch),verbose=0)
    
    if epsilon>epsilon_min:
        epsilon*=epsilon_decay


def run(environment):
    env=gym.make(environment)
    if max_env_steps is not None: env.max_epsiode_steps=max_env_steps
    
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import Adam
    
    model = Sequential()
    model.add(Dense(24, input_dim=env.observation_space.shape[0], activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(env.action_space.n, activation='relu'))
    model.compile(loss='mse', optimizer=Adam(lr=alpha,decay=alpha_decay))
    #model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=alpha))
    scores=deque(maxlen=100)
    rewards=deque()
    episodes=deque()
    for e in range(n_episodes):
        state=preprocess_state(env,env.reset())
        done=False
        i=0
        while not done:
            action=choose_action(env,model,state,get_epsilon(e))
            next_state,reward,done,_=env.step(action)
            #env.render()
            next_state=preprocess_state(env,next_state)
            remember(state,action,reward,next_state,done)
            state=next_state
            i+=1
        scores.append(i) 
        rewards.append(i)
        episodes.append(e)       
        #print("scores deque:{}".format(scores))                                   
        mean_score=np.mean(scores)
        if mean_score>=n_win_ticks and e>=100:
            if not quiet: print("Ran {} episodes. Solved after {} trials.".format(e,e-100))
            return e-100
        if e%20==0 and not quiet:
            print('[Episode {}]. Mean survival time after last 20 episodes was {} ticks'.format(e,mean_score))
            printSomething('[Episode {}]. Mean survival time after last 20 episodes was {} ticks'.format(e,mean_score))
            replay(model,batch_size,get_epsilon(e))
    if not quiet: print ("Did not solve after {} epsiodes".format(e))
    pylab.plot(episodes,rewards,'b')
    pylab.title('Rewards graph')
    pylab.xlabel('Episodes')
    pylab.ylabel('Rewards')
    pylab.savefig("cartpole_reinforce.png")
    return e




from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image

root = Tk(className='reinforcement Learning')
root.title("Reinforcement Learning")
root.geometry("300x300")
#LABEL
label = ttk.Label(root,text = 'Machine Learning Project')
label.pack()
label = ttk.Label(root,text = 'UML501')
label.pack()
label.config(justify = CENTER)
label.config(foreground = 'black')
img="./assets/cartpole.jpg"
image2 = ImageTk.PhotoImage(Image.open(img))
panel = ttk.Label(root, image = image2)
panel.pack(side = "bottom", fill = "both", expand = "yes")
#BUTTON
button = ttk.Button(root, text = 'TRAIN')
button.pack() 

def printSomething(text_value):
    # if you want the button to disappear:
    # button.destroy() or button.pack_forget()
    label = Label(root, text= text_value)
    #this creates a new label to the GUI
    label.pack()
    
    
#FUNCTION DEF
def call():
    run('CartPole-v0')
    #printSomething(text_value)

button.config(command = call)   

root.mainloop()