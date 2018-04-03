import gym

from fuzzy_system import FuzzySystem

fuzzy_system = FuzzySystem.load("result.pkl")
env = gym.make('CartPole-v0')
env.seed(200)
observation = env.reset()
done = False
while True:
    env.render()
    action = fuzzy_system.take_action(observation)
    action = int(action > 0)
    observation, reward, done, info = env.step(action)
    if done:
        break
    # time.sleep(0.05)

env.close()
