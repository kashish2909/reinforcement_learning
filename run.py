import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
from non_trained import run3
from matplotlib import pylab
from pylab import *
import pandas as pd
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
#from tensorflow.python.keras.callbacks import TensorBoard
import tensorflow as tf
import csv
from time import time
EPISODES = 400

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()


    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        return self.model.get_weights()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        import random
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        import random
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                #tensorboard=TensorBoard(log_dir="logs/{}".format(time()))
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    @staticmethod
    def run1():
        env = gym.make('CartPole-v1')
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = DQNAgent(state_size, action_size)
        agent.load("./save/cartpole-ddqn.h5")
        done = False
        batch_size = 32
        rewards=deque()
        episodes=deque()
        for e in range(EPISODES):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            for time in range(500):
                env.render()
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    weights=agent.update_target_model()
                    df = pd.DataFrame(weights)
                    df.to_csv("weights.csv")
                    print("episode: {}/{}, score: {}, e: {:.2}"
                        .format(e, EPISODES, time, agent.epsilon))
                    rewards.append(time)
                    episodes.append(e)
                    break
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
            # if e % 10 == 0:
            #     agent.save("./save/cartpole-ddqn.h5")
        pylab.plot(episodes,rewards,'b')
        pylab.title('Rewards graph')
        pylab.xlabel('Episodes')
        pylab.ylabel('Rewards')
        pylab.savefig("cartpole_reinforce.png")
        episodes.clear()
        rewards.clear()
    
    @staticmethod
    def run2():
        env = gym.make('CartPole-v1')
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = DQNAgent(state_size, action_size)
        agent.load("./save/cartpole-ddqn.h5")
        done = False
        batch_size = 32
        rewards=deque()
        episodes=deque()
        for e in range(EPISODES):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            for time in range(500):
                #env.render()
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    weights=agent.update_target_model()
                    df = pd.DataFrame(weights)
                    df.to_csv("weights.csv")
                    print("episode: {}/{}, score: {}, e: {:.2}"
                        .format(e, EPISODES, time, agent.epsilon))
                    rewards.append(time)
                    episodes.append(e)
                    break
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
            # if e % 10 == 0:
            #     agent.save("./save/cartpole-ddqn.h5")
        pylab.plot(episodes,rewards,'b')
        pylab.title('Rewards graph')
        pylab.xlabel('Episodes')
        pylab.ylabel('Rewards')
        pylab.savefig("cartpole_reinforce.png")
        episodes.clear()
        rewards.clear()


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
button1 = ttk.Button(root, text = 'Train with render')
button1.pack() 

button2 = ttk.Button(root, text = 'Train without render')
button2.pack() 

button3 = ttk.Button(root, text = 'Non Trained')
button3.pack() 

def printSomething(text_value):
    # if you want the button to disappear:
    # button.destroy() or button.pack_forget()
    label = Label(root, text= text_value)
    #this creates a new label to the GUI
    label.pack()
    
 
#FUNCTION DEF
def call1():
    obj=DQNAgent(4,2)
    obj.run1()
    #printSomething(text_value)
def call2():
    obj=DQNAgent(4,2)
    obj.run2()
    
def call3():
    run3()

button1.config(command = call1)   
button2.config(command = call2)   
button3.config(command = call3)   

root.mainloop()