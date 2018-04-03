import numpy as np
import gym

from fuzzy_system import FuzzySystem

fuzzy_system = FuzzySystem.load("result.pkl")
env = gym.make('CartPole-v0')
env.seed(int(np.random.randint(0, 2000, 1)[0]))
observation = env.reset()
done = False
while True:
    env.render()
    action = fuzzy_system.take_action(observation)
    action = int(action > 0)
    observation, reward, done, info = env.step(action)
    theta = abs(observation[0])
    if theta > 0.85:
        break
    # if done:
    #     break
    # time.sleep(0.05)

env.close()
