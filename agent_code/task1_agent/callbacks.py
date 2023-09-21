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
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = relu(self.layer1(x))
        x = self.layer2(x)

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

        self.model = Network(9, 64, 6)

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

    while True:
        if ((action == "UP" and features[0] != 0) or
            (action == "DOWN" and features[1] != 0) or
            (action == "LEFT" and features[2] != 0) or
            (action == "RIGHT" and features[3] != 0)):
            prediction[idx] = -torch.inf
            idx = torch.argmax(prediction).item()
            action = ACTIONS[idx]
        else: 
            break

    # if game_state["step"] > 2:
    #     if self.transitions[-1].action == opposite(self.transitions[-2].action) == self.transitions[-3].action:
    #         action = np.random.choice(ACTIONS, p=[.25, .25, .25, .25, 0, 0])

    return action

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

    # # For example, you could construct several channels of equal shape, ...
    # channels = []
    # channels.append(...)
    # # concatenate them as a feature tensor (they must have the same shape), ...
    # stacked_channels = np.stack(channels)
    # # and return them as a vector
    # return stacked_channels.reshape(-1)

    feature_vector = []

    agent_position = game_state["self"][3]
    x_agent, y_agent = agent_position

    field = game_state["field"]

    # neighbouring fields
    field_up = field[x_agent, y_agent-1]
    field_down = field[x_agent, y_agent+1]
    field_left = field[x_agent-1, y_agent]
    field_right = field[x_agent+1, y_agent]

    feature_vector.append(field_up)
    feature_vector.append(field_down)
    feature_vector.append(field_left)
    feature_vector.append(field_right)


    coin_positions = game_state["coins"]

    try:
        # calculate distance to nearest coin
        distance_to_all_coins = np.sum(np.abs(np.subtract(coin_positions, agent_position)), axis=1)
        
        # add two if wall
        if x_agent % 2 == 0:
            distance_to_all_coins = np.where(x_agent == np.array(coin_positions)[:, 0], distance_to_all_coins + 2, distance_to_all_coins)
        elif y_agent % 2 == 0:
            distance_to_all_coins = np.where(y_agent == np.array(coin_positions)[:, 1], distance_to_all_coins + 2, distance_to_all_coins)

        nearest_coin = coin_positions[np.argmin(distance_to_all_coins)]
        distance_to_nearest_coin = np.min(distance_to_all_coins)

        x_nearest_coin, y_nearest_coin = nearest_coin
        coin_up = 1 if (y_nearest_coin < y_agent) and ((np.subtract(y_nearest_coin, y_agent) == 1) and y_nearest_coin % 2 == 1) else 0
        coin_down = 1 if (y_nearest_coin > y_agent) else 0
        coin_left = 1 if (x_nearest_coin < x_agent) else 0
        coin_right = 1 if (x_nearest_coin > x_agent) else 0

    except ValueError:
        distance_to_nearest_coin = 0
        coin_up = 0
        coin_down = 0
        coin_left = 0
        coin_right = 0

    feature_vector.append(distance_to_nearest_coin) #4
    feature_vector.append(coin_up) #5
    feature_vector.append(coin_down) 
    feature_vector.append(coin_left)
    feature_vector.append(coin_right) #8



    # create features to dodge bombs
    bombs = game_state["bombs"]

    # filter for bombs which can kill the agent
    # using the information when the bomb detonates could restrict this list even further
    dangerous_bombs = []
    for bomb in bombs:
        x_bomb, y_bomb = bomb[0]
        if(np.abs(np.subtract(x_bomb - x_agent)) + np.abs(np.subtract(y_bomb, y_agent)) < 8):
            dangerous_bombs.append(bomb)
        
    for bomb in dangerous_bombs:
        # explosion_fields contains the fields which will explode
        explosion_fields = np.empty(2, 12, dtype= int)
        x_bomb, y_bomb = bomb[1]
        for i in range(12):
            k = i % 4
            j = i % 3
            # fields downwards
            if k == 0:
                explosion_fields[0,i] = x_bomb
                explosion_fields[1,i] = y_bomb + 1 + j
            # fields upwards
            if k == 1:
                explosion_fields[0,i] = x_bomb
                explosion_fields[1,i] = y_bomb - 1 - j
            # fields to the right
            if k == 2:
                explosion_fields[0,i] = x_bomb + 1 + j
                explosion_fields[1,i] = y_bomb
            # fields to the left
            if k == 3:
                explosion_fields[0,i] = x_bomb - 1 - j
                explosion_fields[1,i] = y_bomb

            # delete fields which hit stone_walls or crates
            # iterate over explosion_fields and check if field of coordinate == -1 or 1

            # check for self traps
            
            # check downwards
            dodge_down = 0
            downwards_explosion = explosion_fields[explosion_fields[1,:] < y_bomb]
            for i in range(len(downwards_explosion)/2):
                if(field[x_bomb + 1, y_bomb - i - 1] == 0) or (field[x_bomb - 1, y_bomb - i - 1] == 0):
                    dodge_down = 1
            if(len(downwards_explosion) == 6 and field[x_bomb, y_bomb - 4] == 0):
                dodge_down = 1



    # dodge = 1 there is an escape route
    # dodge = 0 there is no escape route
    # feature_vector.append(dodge_up) #9
    # feature_vector.append(dodge_down) 
    # feature_vector.append(dodge_left)
    # feature_vector.append(dodge_right) #12


    feature_vector = torch.tensor(feature_vector, dtype=torch.float)
    return feature_vector

def opposite(direction):
    if direction == "UP": return "DOWN"
    if direction == "DOWN": return "UP"
    if direction == "LEFT": return "RIGHT"
    if direction == "RIGHT": return "LEFT"
