import gym
import random
import numpy as np

from keras.layers import Input, Dense, Conv2D
from keras.models import Model
import keras.backend as K

env = gym.make('CarRacing-v0')

# print(observation)
# print(env.action_space.sample())

# action = [wheelturn, throttle, brake]
# wheelturn -1 to +1, left to right
# throttle 0 to 1
# brake 0 to 1


class Agent:
    def __init__(self):
        self.sum = 0
        self.high = 0
        self.action_space = [[1.0, 0.3, 0.0],    # Right turn, with reduced throttle                             
                             [-1.0, 0.3, 0.0],   # Left turn, with reduced throttle
                             [0.0, 1.0, 0.0],    # Full throttle
                             [0.0, 0.0, 0.8]]    # 80% brake

    def action(self, observation, reward):
        # action = random.randint(0, 3)
        self.sum += reward
        if (self.sum > self.high):
            self.high = self.sum

        # This returns a tensor
        inputs = Input(shape=(84,64,1))

        # a layer instance is callable on a tensor, and returns a tensor
        output_1 = Conv2D(32, 8, strides=(4), activation='relu')(inputs)
        output_2 = Conv2D(64, 4, strides=(2), activation='relu')(output_1)
        output_3 = Conv2D(64, 3, strides=(1), activation='relu')(output_2)
        output_4 = Dense(512, activation='relu')(output_3)
        predictions = Dense(4, activation='softmax')(output_4)

        # This creates a model that includes
        # the Input layer and three Dense layers
        model = Model(inputs=inputs, outputs=predictions)

        print(observation[0:84, 16:80, 1])
        print(np.shape(observation[0:84, 16:80, 1]))
        action = model(K.variable(np.reshape(observation[0:84, 16:80, 1], (-1, 84, 64, 1))))

        # print(action, self.sum)
        print(action)
        return self.action_space[action]

    def isLost(self):
        if self.sum + 10 < self.high:
            return True
        else:
            return False 


agent = Agent()
observation = env.reset()
reward = 0

while(True):
    # env.render()
    observation, reward, done, _ = env.step(
        agent.action(observation, reward))  # take a random action
    
    if agent.isLost() or done:
        break

print(np.shape(observation[0:84, 16:80, 1]))
print(agent.sum)
env.close()
