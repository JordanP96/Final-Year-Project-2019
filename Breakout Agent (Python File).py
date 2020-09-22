#!/usr/bin/env python
# coding: utf-8

# # Breakout Agent:
# 
# ### Author: Jordan Phillips
# ### ID: B623995
# ### Module: 18COC257

# In[2]:


# Block 1
# Import libraries and packages to use on the breakout learning agent

import numpy as np 
from sklearn.preprocessing import normalize
import cPickle as pickle 
import gym

import sys

import matplotlib.pyplot as plt
import tensorflow as tf
get_ipython().magic(u'matplotlib inline')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu


# In[3]:


# Block 2
# hyperparameters, as discussed in the report, they will be unchanged from the tested parameters 
# due to them already being optimal for the environment

# [Source 1]

H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4 #for convergence (too low- slow to converge, too high,never converge)
gamma = 0.99 # discount factor for reward (i.e later rewards are exponentially less important)
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?

# End of [Source 1] for Block 2
# [Source 1] https://www.youtube.com/watch?v=PDbXPBwOavc


# In[4]:


# Block 3
# Model initialisation, instead of using an API like Keras in the CartPole code, 
# build the model from a more lower level
# [Source 1]
D = 80 * 80 # input dimensionality: 80x80 grid (the breakout environment)
if resume:
  model = pickle.load(open('save.p', 'rb')) #load from pickled checkpoint
else:
  model = {} #initialize model 
  # "Xavier" initialization for making sure the weights are not too big or small
  model['W1'] = np.random.randn(H,D) / np.sqrt(D)
  model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } 
## rmsprop (gradient descent) memory used to update model
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } 

# End of [Source 1] for Block 3
# [Source 1] https://www.youtube.com/watch?v=PDbXPBwOavc


# In[5]:


# Block 4
# use the sigmoid activation function for creating intervals of 0 -> 1
# [Source 1 and Source 2]

def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x))

# End of [Source 1 and Source 2] for Block 4
# [Source 1] https://www.youtube.com/watch?v=PDbXPBwOavc
# [Source 2] https://www.python-course.eu/neural_networks_with_python_numpy.php


# In[6]:


# Block 5
# prepocessing of the environment, taking a single input frame (I) each time which 
# is after fed into the model

# [Source 1]
def prepro(I):

    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2

    return I.astype(np.float).ravel()  #flattens the input to a vector format

# End of [Source 1] for Block 5
# [Source 1] https://www.youtube.com/watch?v=PDbXPBwOavc


# In[8]:


# Block 6
# Calculate the discounted rewards from the one dimensional array, r

# [Source 1 and Source 3]

def discount_rewards(r):
    # initialise reward matrix as empty
    discounted_r = np.zeros_like(r)
    #to store reward sums
    running_add = 0
    #for each reward in array
    for t in reversed(xrange(0, r.size)):
        #if reward at index t is nonzero, reset the sum, since this was a game boundary
        if r[t] != 0: 
            running_add = 0
        running_add = running_add * gamma + r[t]
        #print(running_add)
        #earlier rewards given more value over time 
        #assign the calculated sum to our discounted reward matrix
        discounted_r[t] = running_add
    return discounted_r

# End of [Source 1] for Block 6
# End of [Source 3] for Block 6
# [Source 1] https://www.youtube.com/watch?v=PDbXPBwOavc
# [Source 3] https://github.com/hunkim/ReinforcementZeroToAll/issues/1


# In[9]:


# Block 7
# get the hidden states by multiplying the input by the first set of weights to 
# detect features like the ball

# [Source 1]

def policy_forward(x):
    h = np.dot(model['W1'], x)
    #apply an activation function to it
    #f(x)=max(0,x) take max value, if less than 0, use 0
    h[h<0] = 0 # ReLU nonlinearity
    #repeat process once more
    #will decide if in each case we should be going UP or DOWN.
    logp = np.dot(model['W2'], h)
    #squash it with an activation (this time sigmoid to output probabilities)
    p = sigmoid(logp)
    return p, h # return probability of taking action 2, and hidden state

# End of [Source 1] for Block 7
# [Source 1] https://www.youtube.com/watch?v=PDbXPBwOavc


# In[10]:


# Block 8
# use the chain rule to compute the error for both layers of the network, 
# taking eph (array of hidden states),
# and epdlogp (gradient) as inputs

# [Source 1]

def policy_backward(eph, epdlogp):
    dW2 = np.dot(eph.T, epdlogp).ravel()
    #Compute derivative hidden. It's the outer product of gradient w/ advatange and weight matrix 2 of 2
    dh = np.outer(epdlogp, model['W2'])
    #apply activation
    
    ####### In early testing trying to address the exploding gradient while debugging, 
    #######ignore was used to see where the error came from
    #with np.errstate(invalid='ignore'):
    #dh = dh[~np.isnan(dh)]
    
    dh[eph <= 0] = 0 # backpro prelu
    #compute derivative with respect to weight 1 using hidden states transpose and input observation
    dW1 = np.dot(dh.T, epx)
    #return both derivatives to update weights
    return {'W1':dW1, 'W2':dW2}

# End of [Source 1] for Block 8
# [Source 1] https://www.youtube.com/watch?v=PDbXPBwOavc


# In[11]:


# Block 9
# Assign the environment to the Pong environment from OpenAIGym ---- https://gym.openai.com/envs/#atari
# create some arrays to hold data as you move through the learning process and keep score, 
# etc., for the analysis with matplotlib

# [Source 4]

env = gym.make("Pong-v0")
observation = env.reset()

# End of [Source 4] for Block 9
# [Source 4] https://gym.openai.com/envs/#atari

prev_x = None # used in computing the difference frame
# observation, hidden state, gradient, reward
xs,hs,dlogps,drs = [],[],[],[]

# matplotlib metrics
running_mean_score = []
score_array = []
episode_array = []


# current reward
running_reward = None
episode_number = 0
reward_sum = 0


# In[ ]:


# Block 10
# Train the agent through a while loop to keep iterating through games.

# [Source 1]

import random #include this import to use the normalise function
summ_array = [] # an array to calculate the average scores
summ = 0


while True:

    # This step is unique to Breakout since the action 1, FIRE, is used to spawn the ball
    # it will need to be called upon at the start of the iteration
    env.render()
    observation, reward, done, info = env.step(1)
    reward_sum += reward

    cur_x = prepro(observation) # Preprocess the current frame to reduce computation
    # get the change in frame from this current one to the previous to detect motion
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x # since the frame is now passed assign it to the previous frame.

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3 # choose an action based on the 
    #returned probability of the action above

    # record various intermediates (needed later for backprop)
    xs.append(x) # observation
    hs.append(h) # hidden state
    if action == 2:
        y = 1
    if action == 3:
        y = 0

    dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken 
    #(see http://cs231n.github.io/neural-networks-2/#losses)
    
    # step the environment and get new measurements
    env.render() # Call the Xming program to display the current frame
    # Take the info from each action performed in the environment, which is returned by step()
    observation, reward, done, info = env.step(action)
    reward_sum += reward # add the reward to the reward sum, this is the score of the current game
    # record reward (has to be done after we call step() to get reward for previous action)
    drs.append(reward) 

    if done: # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        #each episode is a few dozen games
        epx = np.vstack(xs) #obsveration
        eph = np.vstack(hs) #hidden
        epdlogp = np.vstack(dlogps) #gradient
        epr = np.vstack(drs) #reward
        xs,hs,dlogps,drs = [],[],[],[] # reset array memory

        #the strength with which we encourage a sampled action is the weighted sum of all rewards 
        # afterwards, but later rewards are exponentially less important
        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        discounted_epr = np.random.normal(discounted_epr) 
        # Use the normalise instead of np.nmean and np.standarddeviation

        #advatnage - quantity which describes how good the action is compared to the average of all the action.
        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph, epdlogp)
        for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        # [Source 1 and Source 5]
        if episode_number % batch_size == 0:
            for k,v in model.iteritems():
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
        # End of [Source 5] for Block 10
        # [Source 5] http://68.media.tumblr.com/2d50e380d8e943afdfd66554d70a84a1/tumblr_inline_o4gfjnL2xK1toi3ym_500.png


        # book-keeping for matplotlib and keeping rewards correct
        score_array.append(reward_sum)
        episode_array.append(episode_number)
        print('Episode: %d.' % (episode_number))

        #Adaptation on the Pong agent, to save computation, use the 100 most recent games to track average performance
        if episode_number > 100:
            summ = 0
            for f in range(0, 100):
                summ += score_array[-f-1]
            summ_array.append(summ/100)
            print('recent sum', summ_array[-1])

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f.' % (reward_sum))
        if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))

        reward_sum = 0 # since the game has finished, the score must be reset to 0
        observation = env.reset() # reset environment
        prev_x = None # since the game is over there is no previous frame of the environment


        if reward != 0: 
            print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')

# End of [Source 1] for Block 10
# [Source 1] https://www.youtube.com/watch?v=PDbXPBwOavc


# In[ ]:


# Block 11
# call the close on the environment, since Xming is rendered by the render() function, 
# it must be closed by the close() function

env.close()


# In[ ]:


# Block 12
# Debugging of the breakout scores to see if all values are correct

print(score_array)
print(running_mean_score)
print(episode_array)


# In[23]:


# Block 13
# Using matplotlib to display the output of the Breakout Agent on a graph

plt.plot(episode_array, score_array, 'bo')
plt.axis([1, 200383, 0, 16])
plt.xlabel('Episode Number', fontsize=18)
plt.ylabel('Game score', fontsize=16)
plt.show()


# In[66]:


# Block 14
# Of the games the agent has played, get a value, count, of how many games the agent has won.

k = 0
count = 0
for i in score_array : 
    if i > k : 
        count = count + 1
        
print ("Number of games the AI won : " + str(count))


# In[20]:


# Block 15
# Find the index of the max score of the agent

no_ep = score_array.index(int(max(score_array)))
print(no_ep)
print(max(score_array))


# In[17]:


# Block 16
# See how many games the agent played

print(len(score_array))


# In[18]:


# Block 17
# See how many the agent played and the running average of the games

print(episode_number)
print(running_reward)

