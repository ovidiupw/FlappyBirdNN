import random
from collections import deque

import gym_ple  # Do not delete, even if IDE says it's not used
import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, copy
from keras.models import Sequential
from keras.utils.visualize_util import plot

from utils import GrayscaleConverter, ImageResizer


class Bird:
    class Actions:
        DO_NOTHING = 0
        FLAP = 1

        @classmethod
        def from_index(cls, idx):
            if idx == 0:
                return "DO_NOTHING"
            else:
                return "FLAP"


class QLearningHistory:
    def __init__(self, max_size=100):
        self.__max_size = max_size
        self.__history = deque(maxlen=max_size)
        self.size = 0

    def record_event(self, state, action, reward, new_state):
        if self.size < self.__max_size:
            self.size += 1
        self.__history.append((state, action, reward, new_state))

    def get_last_event(self):
        return self.__history[-1]

    def is_full(self):
        return self.size >= self.__max_size

    def get_events(self):
        return self.__history


class FlappyBirdDeepQNetwork:
    def __init__(self, pipe_reward=5, dead_reward=-10, alive_reward=1, discount_factor=0.95, nn_batch_size=10,
                 nn_train_epochs=5, nn_image_resize_ratio=0.5, nn_history_size=100, nn_history_sample_size=50):

        self.LOG = gym.logger
        self.nn_batch_size = nn_batch_size
        self.nn_train_epochs = nn_train_epochs
        self.nn_image_resize_ratio = nn_image_resize_ratio
        self.nn_history_size = nn_history_size
        self.nn_history_sample_size = nn_history_sample_size
        self.pipe_reward = pipe_reward
        self.dead_reward = dead_reward
        self.alive_reward = alive_reward
        self.discount_factor = discount_factor
        self.q_learning_history = QLearningHistory(self.nn_history_size)
        self.exploration_factor = 1

        self.state_prediction_nn = None
        self.nn_initialized = False

        self.env = gym.make('FlappyBird-v0')
        out_dir = '/tmp/random-agent-results'
        self.env.monitor.start(out_dir, force=True, write_upon_reset=True)

    def run(self, episode_count=1000, learning_rate=0.5, training=False):
        for episode_idx in range(0, episode_count):
            self.LOG.info("Episode #{n} started.".format(n=episode_idx))

            observation = self.env.reset()
            observation_object = self.__downscale_image(observation)

            if not self.nn_initialized:
                self.__initialize_nn(observation_object)
                self.nn_initialized = True
                self.LOG.info("Initialized the neural network!")

            observation_width = observation_object.width
            observation_height = observation_object.height
            observation = observation_object.data

            while True:
                q_values = self.state_prediction_nn.predict(
                    x=observation.reshape(1, observation_width * observation_height),
                    batch_size=1)
                old_observation = copy.deepcopy(observation)

                bird_action = (np.argmax(q_values))
                observation_object, reward, done, _ = self.__take_bird_action(bird_action)
                observation = self.__downscale_image(observation_object).data
                self.LOG.info("Current action reward: {r}. Done: {d}".format(r=reward, d=done))

                if training:
                    reward = self.__get_custom_reward(reward)

                    # Uncomment below to show the reduced-size, grayscale flappy bird screenshot
                    # plt.imshow(observation, cmap=matplotlib.cm.Greys_r)  # DEBUG
                    # plt.show()  # DEBUG

                    self.q_learning_history.record_event(
                        state=old_observation, action=bird_action, reward=reward, new_state=observation)

                    last_event = self.q_learning_history.get_last_event()
                    self.LOG.info("Added event #{n} to history. Action: {a}; Reward: {r}"
                                  .format(a=Bird.Actions.from_index(last_event[1]),
                                          r=reward,
                                          n=self.q_learning_history.size))

                    if self.q_learning_history.is_full():
                        history_batch = random.sample(self.q_learning_history.get_events(), self.nn_history_sample_size)
                        self.LOG.info("Sampling {n} events from history with size {s}"
                                      .format(n=self.nn_history_sample_size, s=self.q_learning_history.size))

                        nn_training_batch_data = []
                        nn_training_batch_labels = []

                        for history_event in history_batch:
                            old_state, action, reward, new_state = history_event

                            q_values_before_action = self.state_prediction_nn.predict(
                                x=old_state.reshape(1, observation_width * observation_height), batch_size=1)
                            q_values_after_action = self.state_prediction_nn.predict(
                                x=new_state.reshape(1, observation_width * observation_height), batch_size=1)
                            best_q_value_after_action = np.max(q_values_after_action)
                            y = np.zeros((1, 2))  # only 2 possible actions
                            for value_idx in range(0, len(q_values_before_action)):
                                y[value_idx] = q_values_before_action[value_idx]

                            if reward != self.dead_reward:
                                update = learning_rate * (reward + (self.discount_factor * best_q_value_after_action))
                            else:
                                update = reward

                            y[0][action] = update
                            nn_training_batch_data.append(old_state.reshape(observation_width * observation_height, ))
                            nn_training_batch_labels.append(y.reshape(2, ))

                        nn_training_batch_data = np.array(nn_training_batch_data)
                        nn_training_batch_labels = np.array(nn_training_batch_labels)

                        self.state_prediction_nn.fit(
                            x=nn_training_batch_data,
                            y=nn_training_batch_labels,
                            nb_epoch=self.nn_train_epochs,
                            batch_size=self.nn_batch_size)
                if done:
                    break
            if self.exploration_factor > 0.1:
                self.exploration_factor -= (1.0 / episode_count)
                self.LOG.info("Exploration factor updated! New value: {v}".format(v=self.exploration_factor))

        self.env.monitor.close()

    def __downscale_image(self, image):
        """
        Resized the input image and converts it to grayscale
        :param image: The image do be downsized and grayscaled
        :return: The grayscale, resized image corresponding to the input image
        """
        grayscale_observation_image = GrayscaleConverter.rgb_to_grayscale(image)
        resized_observation_image = ImageResizer.resize_image(
            image=grayscale_observation_image,
            ratio=self.nn_image_resize_ratio)
        return resized_observation_image

    def __initialize_nn(self, nn_input_observation):
        nn_input_layer_size = nn_input_observation.data.shape[0] * nn_input_observation.data.shape[1]
        nn_hidden_layer_size = 100
        nn_output_layer_size = 2  # Two possible actions that the bird can take (flap or do nothing)

        nn_input_layer = Dense(
            init='lecun_uniform',  # Uniform initialization scaled by the square root of the number of inputs
            output_dim=nn_hidden_layer_size,
            input_shape=(nn_input_layer_size,),
            activation='sigmoid')

        self.LOG.info("Adding layer to neural network: input_size: {i}, output_size: {o}"
                      .format(i=nn_input_layer_size, o=nn_hidden_layer_size))

        nn_hidden_layer = Dense(
            init='lecun_uniform',
            output_dim=nn_output_layer_size,
            activation='linear'
        )

        self.LOG.info("Adding layer to neural network: output_size: {o}"
                      .format(o=nn_output_layer_size))

        self.state_prediction_nn = Sequential()  # Initialize a nn with a linear stack of layers
        self.state_prediction_nn.add(nn_input_layer)
        self.state_prediction_nn.add(nn_hidden_layer)

        # use mean squared error regression (aka cost derivative)
        # to compute errors to propagate backwards
        self.state_prediction_nn.compile(
            optimizer='rmsprop',
            loss='mean_squared_error')

        # Uncomment the line below to output an image with the structure of the neural network
        # plot(self.state_prediction_nn, to_file='nn_{n}.png'.format(n=episode_idx), show_shapes=True)

    def __take_bird_action(self, bird_action):
        try:
            random_number = np.random.random_sample()
            if not self.q_learning_history.is_full():
                bird_action = self.env.action_space.sample()
                self.LOG.info("Bird chose to do a random move - building qHistory!")
                return self.env.step(bird_action)

            elif random_number < self.exploration_factor:
                bird_action = self.env.action_space.sample()
                self.LOG.info("Epsilon strikes rand={r} < {ef}! Bird chose random move!"
                              .format(r=random_number, ef=self.exploration_factor))
                return self.env.step(bird_action)

            elif bird_action == Bird.Actions.FLAP:
                self.LOG.info("Bird chose to FLAP!")
                return self.env.step(0)  # lift the bird!

            elif bird_action == Bird.Actions.DO_NOTHING:
                self.LOG.info("Bird chose to DO_NOTHING!")
                return self.env.step(1)  # do not lift the bird
        finally:
            self.env.render()

    def __get_custom_reward(self, reward):
        if reward >= 1:
            self.LOG.info("Passed through pipe! -> Reward is {r}".format(r=reward))
            return self.pipe_reward
        elif reward >= 0:
            self.LOG.info("Stayed alive! -> Reward is {r}".format(r=reward))
            return self.alive_reward
        else:
            self.LOG.info("Crashed! -> Reward is {r}".format(r=reward))
            return self.dead_reward


if __name__ == "__main__":
    nn = FlappyBirdDeepQNetwork(
        pipe_reward=25,
        dead_reward=-100,
        alive_reward=1,
        discount_factor=0.95,  # a future reward is more important than a proximity reward, so closer to 1.
        nn_batch_size=50,
        nn_train_epochs=5,
        nn_image_resize_ratio=0.35,
        nn_history_size=5,
        nn_history_sample_size=5
    )
    nn.run(
        episode_count=100,
        learning_rate=0.8,
        training=True
    )
