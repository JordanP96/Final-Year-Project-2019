
# coding: utf-8

# # Cartpole DQN:
# 
# ### Author: Jordan Phillips
# 
# ### ID: B623995
# 
# ### Module: 18COC257

# ## Import Dependancies

# In[1]:


# Block 1
# Import dependancies for the cartpole DQN

import random
import gym
import numpy as np
import ctypes
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os


# ### Set parameters

# In[2]:


# Block 2
#Set parameters for the cartpole game and create the environment of the cartpole game

# [Source 1 and Source 2]

env = gym.make('CartPole-v1')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

#the batch size is the number of memories the agent will learn from at any one time, larger size
# means a longer time to learn and process the memories but gives more accuracy in learning
batch_size = 32
n_episodes = 100

output_dir = 'model_output/CartPoleA'

# End of [Source 1 and 2] for Block 2
# [Source 1] https://gym.openai.com/envs/#classic_control
# [Source 2] https://www.youtube.com/watch?v=OYhFoMySoVs


# In[3]:


# Block 3
# Create a new directory to store memories in if the directory is not already existing

#if no directory exists create a directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# #### Define agent

# In[4]:


# Block 4
# Define the agent properties and hyperparameters to use within the network

# [Source 2]

class DQNAgent:
    
    # initialise the parameters 
    def __init__(self, state_size, action_size):
        
        self.state_size = state_size
        self.action_size = action_size
        
        #only use some of the memory, take a sample, too long to review all of them
        self.memory = deque(maxlen=2000)
        #discount factor (gamma)
        self.gamma = 0.95
        #exploration rate(epsilon), don't just exploit best practices, find new paths through the environment wheer 1.0 is 100% exploration compared to exploitation
        self.epsilon = 1.0
        self.epsilon_decay = 0.9
        #very minimum exploration the agent will do (1% of the time)
        self.epsilon_min = 0.01
        #stochastic learning degrading gradient
        self.learning_rate = 0.0001
        
        #private method only used by this instance
        self.model = self._build_model()
    
    #buld the model of the network using Keras API and Adam optimiser
    def _build_model(self):
        
        model = Sequential()
        
        #first layer will have 24 neurons, activation has rectified linear unit
        model.add(Dense(24, input_dim = self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        #linear activation due to direct estimation
        model.add(Dense(self.action_size, activation='linear'))
        
        #method of loss is 'mean squared error'
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        
        return model
    
    #takes in the state, the action, the reward, and the next state at the current time stamp
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    #figuring out what action to take at this state
    def act(self, state):
        
        #if the randomly selected value less than epsilon, explore a random path
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        #have a guess at what the best possible state is to max future reward
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    #replay method, takes the batch size as input for selction of memories
    def replay(self, batch_size):
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            #predicted future reward
            target_f = self.model.predict(state)
            target_f[0][action] = target
            
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def load(self, name):
        self.model.load_weights(name)
        
    def save(self, name):
        self.model.save_weights(name)

# End of [Source 2] for Block 4
# [Source 2] https://www.youtube.com/watch?v=OYhFoMySoVs


# In[5]:


# Block 5
# create the agent from the class

agent = DQNAgent(state_size, action_size)


# #### interact with the environment

# In[ ]:


# Block 6
# Run the agent through the environment for the number of episodes defined earlier
#start environment at False condition so the game isn't initally 'done'

# [Source 2]

done = False
for e in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    
    for time in range(5000):
        env.render() #Using render worked on windows 10 here as the 
        #environment was not dependant on Linus based packages like Atari
        action = agent.act(state)
        
        next_state, reward, done, _ = env.step(action)
        #if 5000 has been reached or you die, get penalised by -10
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        
        agent.remember(state, action, reward, next_state, done)
        
        state = next_state
        
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, n_episodes, time, agent.epsilon))
            break
            
        #train the theta to learn from experience. maximise Q*
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            
        if e % 50 == 0:
            agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")
            
# End of [Source 2] for Block 6
# [Source 2] https://www.youtube.com/watch?v=OYhFoMySoVs


# In[7]:


# Block 7
#Close the environment down after use since it has been rendered elsewhere
env.close()

