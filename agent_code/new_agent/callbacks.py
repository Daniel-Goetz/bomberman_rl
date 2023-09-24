import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import relu


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


class Network(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden1_size)
        # self.layer2 = nn.Linear(hidden1_size, hidden2_size)
        self.layer3 = nn.Linear(hidden2_size, output_size)

    def forward(self, x):
        x = relu(self.layer1(x))
        # x = relu(self.layer2(x))
        x = self.layer3(x)

        return x


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
        
        self.model = Network(225, 128, 128, 6)
        
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


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
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")

    features = state_to_features(game_state)
    prediction = self.model(features)
    idx = torch.argmax(prediction).item()
    action = ACTIONS[idx]

    return action


def state_to_features(game_state: dict):
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: 
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    agent = game_state["self"]
    field = game_state["field"]
    others = game_state["others"]
    coins = game_state["coins"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]

    agent_name, agent_score, agent_bomb_available, (agent_xpos, agent_ypos) = agent
    field[agent_xpos, agent_ypos] = 4

    for enemy_name, enemy_score, enemy_bomb_available, (enemy_xpos, enemy_ypos) in others:
        field[enemy_xpos, enemy_ypos] = 3

    for coin_xpos, coin_ypos in coins:
        field[coin_xpos, coin_ypos] = 2

    for (bomb_xpos, bomb_ypos), bomb_time in bombs:
        field[bomb_xpos, bomb_ypos] = -2

    field = np.where(explosion_map == 0, field, -3)

    field = field[1:-1,1:-1]

    return torch.from_numpy(field.flatten().astype(np.float32))
