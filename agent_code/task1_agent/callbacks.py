import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        #weights = np.random.rand(len(ACTIONS))
        #weights[4] = 0
        #weights[5] = 0
        #self.model = weights / weights.sum()

        # Q-Matrix with possbile 400 states times 6 actions initializied with zeros
        # weights = np.zeros((400, len(ACTIONS)))
        # random initialization
        # weights = np.abs(np.random.random((26, len(ACTIONS)-2)))
        weights = np.zeros((5*15*15,len(ACTIONS)-2))

        # idea of the model (remove the actions bomb and wait for the simpelst model)
        # and represent the location of the agent in a 5 x 5 grid (ignore the outer walls and then build 3x3 blocks)
        # track the distance to the neareast coin to the agent in the last vector spot (1,2,3,4 or higher)
        self.model = weights
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def nearest_coin(game_state):
    targets = game_state["coins"]
    start = game_state["self"][3]

    # calculates the distance between the agent and the nearest coin
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()
    return best_dist


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.25, .25, .25, .25, .0, .0])

    self.logger.debug("Querying model for action.")
    # return np.random.choice(ACTIONS, p=self.model) # needs to be reworked

    """"
    step = game_state['step'] - 1 #array indexing begins with 0 and the steps with 1
    idx = np.where(self.model[step,] == np.amax(self.model[step,]))
    return ACTIONS[idx[0][0]]
    """

    # reduce 17x17 grid into 5x5 grid
    name, score, bomb, coordinates = game_state["self"]
    x,y = coordinates
    x = x-1
    y = y-1
    # x = int((x-1)/3)
    # y = int((y-1)/3)
    # counting the 25 states from the upper left to the right and then the next row (0 to 24)
    field = x + 15*y

    distance = nearest_coin(game_state)

    # reduce states by handling coin-agents distances of more than 4 as the same state
    if distance > 4:
        distance = 4

    # count the 25 states from lowest to highest distance
    state = field + 15*15*(distance-1)

    action = np.argmax(self.model[state])
    return ACTIONS[action]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(game_state["self"][3])
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
