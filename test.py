import gym

from fuzzy_system import FuzzySystem

fuzzy_system = FuzzySystem.load("result.pkl")
env = gym.make('CartPole-v0')
observation = env.reset()
done = False
while not done:
    env.render()
    action = fuzzy_system.take_action(observation)
    action = action > 0
    observation, reward, done, info = env.step(action)
