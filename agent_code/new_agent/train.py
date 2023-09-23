from collections import namedtuple, deque
import random

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
                        ('state', 'action', 'next_state', 'reward', 'end'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 10_000  # keep only ... last transitions
BATCH_SIZE = 500
# RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.9

class Trainer:
    def __init__(self, model, learning_rate, discount_factor) -> None:
        self.model = model
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()

    def train_step(self, transition: "Transition"):

        state = transition.state
        action = transition.action
        next_state = transition.next_state
        reward = transition.reward
        end_of_round = transition.end

        # predict Q values with current state
        prediction = self.model(state)

        target = prediction.clone()
        if end_of_round:
            Q_new = reward
        else:
            Q_new = reward + self.discount_factor * torch.max(self.model(next_state))

        target[ACTIONS.index(action)] = Q_new

        self.optimizer.zero_grad()
        loss = self.loss_function(target, prediction)
        loss.backward()

        self.optimizer.step()

    def train_from_buffer(self, transitions):

        if len(transitions) > BATCH_SIZE:
            sampled_transitions = random.sample(transitions, BATCH_SIZE)
        else:
            sampled_transitions = transitions

        states, actions, next_states, rewards, ends = zip(*sampled_transitions)
        states = torch.stack(states)

        # predict Q values with current state
        predictions = self.model(states)

        targets = predictions.clone()
        for idx, end in enumerate(ends):
            if end:
                Q_new = rewards[idx]
            else:
                Q_new = rewards[idx] + self.discount_factor * torch.max(self.model(next_states[idx]))

            targets[idx][ACTIONS.index(actions[idx])] = Q_new

        self.optimizer.zero_grad()
        loss = self.loss_function(targets, predictions)
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

    # state_to_features is defined in callbacks.py
    transition = Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events), False)
    self.transitions.append(transition)
    self.trainer.train_step(transition)


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
    transition = Transition(state_to_features(last_game_state), last_action, torch.zeros(225), reward_from_events(self, events), True)
    self.transitions.append(transition)
    self.trainer.train_step(transition)

    self.trainer.train_from_buffer(self.transitions)
    
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
        e.MOVED_LEFT: 0,
        e.MOVED_RIGHT: 0,
        e.MOVED_UP: 0,
        e.MOVED_DOWN: 0,
        e.WAITED: -1,
        e.INVALID_ACTION: -1,

        e.BOMB_DROPPED: 1,
        e.BOMB_EXPLODED: 0,

        e.CRATE_DESTROYED: 1,
        e.COIN_FOUND: 1,
        e.COIN_COLLECTED: 5,

        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -3,

        e.GOT_KILLED: -1,
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: 0
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
