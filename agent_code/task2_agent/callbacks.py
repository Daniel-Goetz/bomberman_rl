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

        self.model = Network(20, 256, 6)

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

    field = game_state["field"]
    bombs = game_state["bombs"]
    bomb_positions = [bomb_pos for (bomb_pos, _) in bombs]
    others = game_state["others"]
    other_postions = [other_pos for (_,_,_, other_pos) in others]
    x_agent, y_agent = game_state["self"][3]


    while True:
        if ((action == "UP" and not tile_is_free(field, bomb_positions, other_postions, x_agent, y_agent - 1)) or
            (action == "DOWN" and not tile_is_free(field, bomb_positions, other_postions, x_agent, y_agent + 1)) or
            (action == "LEFT" and  not tile_is_free(field, bomb_positions, other_postions, x_agent - 1, y_agent)) or
            (action == "RIGHT" and not tile_is_free(field, bomb_positions, other_postions, x_agent + 1, y_agent)) or
            (action == "BOMB" and not game_state["self"][2])):
            prediction[idx] = -torch.inf
            idx = torch.argmax(prediction).item()
            action = ACTIONS[idx]
        else: 
            break

    return action

def tile_is_free(field, bombs, active_agents, x, y):
        is_free = (field[x, y] == 0)
        if is_free:
            for (x_,y_) in bombs + active_agents:
                is_free = is_free and (x_ != x or y_ != y)
        return is_free

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
        coin_up = 1 if y_nearest_coin < y_agent else 0
        coin_down = 1 if y_nearest_coin > y_agent else 0
        coin_left = 1 if x_nearest_coin < x_agent else 0
        coin_right = 1 if x_nearest_coin > x_agent else 0

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

    bombs = game_state["bombs"]
    in_danger = 0
    distance_to_nearest_bomb = 5
    dodge_down = dodge_left = dodge_right = dodge_up = -1
    safe = -1
    if bombs:
        safe = 1
        for (x_bomb, y_bomb), t_bomb in bombs:
            if ((x_bomb == x_agent and np.abs(y_bomb - y_agent) < 4) or 
                (y_bomb == y_agent and np.abs(x_bomb - x_agent) < 4)):
                in_danger = 1

        bomb_positions = [bomb_pos for (bomb_pos, _) in bombs]
        distance_to_all_bombs = np.sum(np.abs(np.subtract(bomb_positions, agent_position)), axis=1)
        distance_to_nearest_bomb = np.min(distance_to_all_bombs)
        # set to 5 if bigger than 5
        distance_to_nearest_bomb = 5 if distance_to_nearest_bomb > 5 else distance_to_nearest_bomb

        nearest_bomb = bomb_positions[np.argmin(distance_to_all_bombs)]

        # check for escape directions
        x_nearest_bomb, y_nearest_bomb = nearest_bomb

        # check for escape in upward direction
        explosion_map_up = []
        dodge_up = 0

        for i in range(3):
            if(field[x_nearest_bomb, y_nearest_bomb - i - 1] == 0):
                explosion_map_up.append((x_nearest_bomb, y_nearest_bomb - i - 1))
            else:
                break

        # check for escape in downward direction
        explosion_map_down = []
        dodge_down = 0

        for i in range(3):
            if(field[x_nearest_bomb, y_nearest_bomb + i + 1] == 0):
                explosion_map_down.append((x_nearest_bomb, y_nearest_bomb + i + 1))
            else:
                break

        # check for escape to the left
        explosion_map_left = []
        dodge_left = 0

        for i in range(3):
            if(field[x_nearest_bomb - i - 1, y_nearest_bomb] == 0):
                explosion_map_left.append((x_nearest_bomb - i - 1, y_nearest_bomb))
            else:
                break

        # check for escape to the right
        explosion_map_right = []
        dodge_right = 0

        for i in range(3):
            if(field[x_nearest_bomb + i + 1, y_nearest_bomb] == 0):
                explosion_map_right.append((x_nearest_bomb + i + 1, y_nearest_bomb))
            else:
                break

        # give escape directions when on bomb
        if(x_agent == x_nearest_bomb) and (y_agent == y_nearest_bomb):
            safe = 0
            for tile in explosion_map_up:
                x, y = tile
                if (field[x + 1, y] == 0) or (field[x - 1, y] == 0):
                    dodge_up = 1
                    break
            if(len(explosion_map_up) == 3) and dodge_up != 1:
                if(field[x_nearest_bomb, y_nearest_bomb - 4] == 0):
                    dodge_up = 1

            for tile in explosion_map_down:
                x, y = tile
                if (field[x + 1, y] == 0) or (field[x - 1, y] == 0):
                    dodge_down = 1
                    break
            if(len(explosion_map_down) == 3) and dodge_up != 1:
                if(field[x_nearest_bomb, y_nearest_bomb + 4] == 0):
                    dodge_down = 1

            for tile in explosion_map_left:
                x, y = tile
                if (field[x, y + 1] == 0) or (field[x, y - 1] == 0):
                    dodge_left = 1
                    break
            if(len(explosion_map_left) == 3) and dodge_up != 1:
                if(field[x_nearest_bomb - 4, y_nearest_bomb] == 0):
                    dodge_left = 1

            for tile in explosion_map_right:
                x, y = tile
                if (field[x, y + 1] == 0) or (field[x, y - 1] == 0):
                    dodge_right = 1
                    break
            if(len(explosion_map_right) == 3) and dodge_up != 1:
                if(field[x_nearest_bomb + 4, y_nearest_bomb] == 0):
                    dodge_right = 1

        # check for escape directions and safe squares
        if(x_nearest_bomb == x_agent) and (y_nearest_bomb != y_agent):
            safe = 0
            if(field[x_agent - 1, y_agent] == 0):
                dodge_left = 1
            if(field[x_agent + 1, y_agent] == 0):
                dodge_right = 1
            if(y_nearest_bomb < y_agent):
                dodge_up = 0 # even though this could be possbile running towards the bomb is never ideal
                if(np.abs(np.subtract(y_nearest_bomb, y_agent)) == 1):
                    if((field[x_agent, y_agent + 1] == 0) and (field[x_agent, y_agent + 2] == 0) and 
                       ((field[x_agent - 1, y_agent + 2] == 0) or (field[x_agent + 1, y_agent + 2] == 0) or 
                       (field[x_agent, y_agent + 3] == 0))):
                        dodge_up = 1
                if(np.abs(np.subtract(y_nearest_bomb, y_agent)) == 2):
                    if((field[x_agent, y_agent + 1] == 0) and ((field[x_agent - 1, y_agent + 1] == 0) or
                        (field[x_agent + 1, y_agent + 1] == 0) or (field[x_agent, y_agent + 2] == 0))):
                        dodge_up = 1
                if(np.abs(np.subtract(y_nearest_bomb, y_agent)) == 3):
                    if((field[x_agent - 1, y_agent] == 0) or (field[x_agent + 1, y_agent] == 0) or 
                        (field[x_agent, y_agent + 1] == 0)):
                        dodge_up = 1
            if(y_nearest_bomb > y_agent):
                dodge_down = 0 # even though this could be possbile running towards the bomb is never ideal
                if(np.abs(np.subtract(y_nearest_bomb, y_agent)) == 1):
                    if((field[x_agent, y_agent - 1] == 0) and (field[x_agent, y_agent - 2] == 0) and 
                       ((field[x_agent - 1, y_agent - 2] == 0) or (field[x_agent + 1, y_agent - 2] == 0) or 
                       (field[x_agent, y_agent - 3] == 0))):
                        dodge_down = 1
                if(np.abs(np.subtract(y_nearest_bomb, y_agent)) == 2):
                    if((field[x_agent, y_agent - 1] == 0) and ((field[x_agent - 1, y_agent - 1] == 0) or
                        (field[x_agent + 1, y_agent - 1] == 0) or (field[x_agent, y_agent - 2] == 0))):
                        dodge_down = 1
                if(np.abs(np.subtract(y_nearest_bomb, y_agent)) == 3):
                    if((field[x_agent - 1, y_agent] == 0) or (field[x_agent + 1, y_agent] == 0) or 
                        (field[x_agent, y_agent - 1] == 0)):
                        dodge_down = 1


        
        if(y_nearest_bomb == y_agent) and (x_nearest_bomb != x_agent):
            safe = 0
            if(field[x_agent, y_agent - 1] == 0):
                dodge_up = 1
            if(field[x_agent, y_agent + 1] == 0):
                dodge_down = 1
            if(x_nearest_bomb < x_agent):
                dodge_left = 0 # even though this could be possbile running towards the bomb is never ideal
                if(np.abs(np.subtract(x_nearest_bomb, x_agent)) == 1):
                    if((field[x_agent + 1, y_agent] == 0) and (field[x_agent + 2, y_agent] == 0) and 
                       ((field[x_agent + 2, y_agent - 1] == 0) or (field[x_agent + 2, y_agent + 1] == 0) or 
                       (field[x_agent + 3, y_agent] == 0))):
                        dodge_right = 1
                if(np.abs(np.subtract(x_nearest_bomb, x_agent)) == 2):
                    if((field[x_agent + 1, y_agent] == 0) and ((field[x_agent + 1, y_agent - 1] == 0) or
                        (field[x_agent + 1, y_agent + 1] == 0) or (field[x_agent + 2, y_agent] == 0))):
                        dodge_right = 1
                if(np.abs(np.subtract(x_nearest_bomb, x_agent)) == 3):
                    if((field[x_agent, y_agent - 1] == 0) or (field[x_agent, y_agent + 1] == 0) or 
                        (field[x_agent + 1, y_agent] == 0)):
                        dodge_right = 1
            if(x_nearest_bomb > x_agent):
                dodge_right = 0 # even though this could be possbile running towards the bomb is never ideal
                if(np.abs(np.subtract(x_nearest_bomb, x_agent)) == 1):
                    if((field[x_agent - 1, y_agent] == 0) and (field[x_agent - 2, y_agent] == 0) and 
                       ((field[x_agent - 2, y_agent - 1] == 0) or (field[x_agent - 2, y_agent + 1] == 0) or 
                       (field[x_agent - 3, y_agent] == 0))):
                        dodge_left = 1
                if(np.abs(np.subtract(x_nearest_bomb, x_agent)) == 2):
                    if((field[x_agent - 1, y_agent] == 0) and ((field[x_agent - 1, y_agent - 1] == 0) or
                        (field[x_agent - 1, y_agent + 1] == 0) or (field[x_agent - 2, y_agent] == 0))):
                        dodge_left = 1
                if(np.abs(np.subtract(x_nearest_bomb, x_agent)) == 3):
                    if((field[x_agent, y_agent - 1] == 0) or (field[x_agent, y_agent + 1] == 0) or 
                        (field[x_agent - 1, y_agent] == 0)):
                        dodge_left = 1


    
    feature_vector.append(in_danger) #9
    feature_vector.append(distance_to_nearest_bomb) #10

    # get escape directions
    # dodge = 1 escape route / dodge = 0 no escape route

    feature_vector.append(dodge_up) #11
    feature_vector.append(dodge_down)
    feature_vector.append(dodge_left)
    feature_vector.append(dodge_right) #14
    feature_vector.append(safe) #15

    # explosion status for each direction
    explosion_map = game_state["explosion_map"]

    explosion_up = explosion_map[x_agent, y_agent-1]
    explosion_down = explosion_map[x_agent, y_agent+1]
    explosion_left = explosion_map[x_agent-1, y_agent]
    explosion_right = explosion_map[x_agent+1, y_agent]

    feature_vector.append(explosion_up) #16
    feature_vector.append(explosion_down)
    feature_vector.append(explosion_left)
    feature_vector.append(explosion_right) #19
     
    feature_vector = torch.tensor(feature_vector, dtype=torch.float)
    return feature_vector
