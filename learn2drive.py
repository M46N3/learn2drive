import gym
import random
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
        action = random.randint(0, 3)
        self.sum += reward
        if (self.sum > self.high):
            self.high = self.sum
        print(action, self.sum)
        return self.action_space[action]

    def isLost(self):
        if self.sum + 5 < self.high:
            return True
        else:
            return False 


agent = Agent()
observation = env.reset()
reward = 0

for _ in range(1000):
    env.render()
    observation, reward, done, _ = env.step(
        agent.action(observation, reward))  # take a random action
    
    if agent.isLost():
        break

# print(observation[:, :, 1])
print(agent.sum)
env.close()
