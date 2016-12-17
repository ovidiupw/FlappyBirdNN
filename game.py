import gym
import gym_ple


# The world's simplest agent!
import sys


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == "__main__":

    env = gym.make('FlappyBird-v0')
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    for _ in range(episode_count):
        observation = env.reset()
        while True:
            action = agent.act(observation, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
            env.render()
