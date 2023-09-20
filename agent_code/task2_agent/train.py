from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, ACTIONS

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.9

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
GOT_CLOSER_TO_COIN = "CLOSER"
GOT_AWAY_FROM_COIN = "AWAY"
STAYED_PUT = "STAYED_PUT"
NO_COIN_COLLECTED = "NO_COIN"
WIGGLE_WIGGLE_WIGGLE = "DU_DU_DU_DUU_DUU_DUUUU"
IN_DANGER = "IN_DANGER"
OUT_OF_DANGER = "OUT_OF_DANGER"

class Trainer:
    def __init__(self, model, learning_rate, discount_factor) -> None:
        self.model = model
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()

    def train_step(self, transition: "Transition", end_of_round):

        state = transition.state
        action = transition.action
        next_state = transition.next_state
        reward = transition.reward

        # predict Q values with current state
        prediction = self.model(state)

        target = prediction.clone()
        Q_new = reward + self.discount_factor * torch.max(self.model(next_state))

        target[ACTIONS.index(action)] = Q_new

        self.optimizer.zero_grad()
        loss = self.loss_function(target, prediction)
        loss.backward()

        self.optimizer.step()

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.trainer = Trainer(self.model, LEARNING_RATE, DISCOUNT_FACTOR)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    old_game_features = state_to_features(old_game_state)
    new_game_features = state_to_features(new_game_state)

    # Add custom events
    if old_game_features[4] <= new_game_features[4]:
        events.append(GOT_AWAY_FROM_COIN)
    # elif old_game_features[4] > new_game_features[4]:
    #     events.append(GOT_CLOSER_TO_COIN)

    if (new_game_features[5] == 1) and (e.MOVED_UP in events) and (new_game_features[0] == 0):
        events.append(GOT_CLOSER_TO_COIN)
    if (new_game_features[6] == 1) and (e.MOVED_DOWN in events) and (new_game_features[1] == 0):
        events.append(GOT_CLOSER_TO_COIN)
    if (new_game_features[7] == 1) and (e.MOVED_LEFT in events) and (new_game_features[2] == 0):
        events.append(GOT_CLOSER_TO_COIN)
    if (new_game_features[8] == 1) and (e.MOVED_RIGHT in events) and (new_game_features[3] == 0):
        events.append(GOT_CLOSER_TO_COIN)

    if old_game_state["self"][3] == new_game_state["self"][3]:
        events.append(STAYED_PUT)

    if e.COIN_COLLECTED not in events:
        events.append(NO_COIN_COLLECTED)

    if new_game_state["step"] > 2:
        if self.transitions[-2].action == opposite(self.transitions[-1].action) == self_action:
            events.append(WIGGLE_WIGGLE_WIGGLE)

    if new_game_features[9]:
        events.append(IN_DANGER)
    else: 
        events.append(OUT_OF_DANGER)

    # state_to_features is defined in callbacks.py
    transition = Transition(old_game_features, self_action, new_game_features, reward_from_events(self, events))
    
    self.transitions.append(transition)

    self.trainer.train_step(transition, end_of_round=False)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.KILLED_OPPONENT: 5,
        e.BOMB_DROPPED: 4,
        e.KILLED_SELF: -10,
        e.CRATE_DESTROYED: 5,
        GOT_CLOSER_TO_COIN: 1,
        GOT_AWAY_FROM_COIN: -1.5,
        STAYED_PUT: -2,
        WIGGLE_WIGGLE_WIGGLE: -2,
        IN_DANGER: -3,
        OUT_OF_DANGER: 3
        # NO_COIN_COLLECTED: -0.2
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def opposite(direction):
    if direction == "UP": return "DOWN"
    if direction == "DOWN": return "UP"
    if direction == "LEFT": return "RIGHT"
    if direction == "RIGHT": return "LEFT"