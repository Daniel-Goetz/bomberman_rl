from collections import namedtuple, deque

import pickle
import random
from typing import List

import events as e
from .callbacks import state_to_features, ACTIONS

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','end'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 10_000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
LEARNING_RATE = 0.005
DISCOUNT_FACTOR = 0.9
BATCH_SIZE = 500

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
GOT_CLOSER_TO_COIN = "CLOSER_COIN"
GOT_CLOSER_TO_BOMB = "CLOSER_BOMB"
GOT_AWAY_FROM_COIN = "AWAY_COIN"
GOT_AWAY_FROM_BOMB = "AWAY_BOMB"
STAYED_PUT = "STAYED_PUT"
NO_COIN_COLLECTED = "NO_COIN"
WIGGLE_WIGGLE_WIGGLE = "DU_DU_DU_DUU_DUU_DUUUU"
IN_DANGER = "IN_DANGER"
OUT_OF_DANGER = "OUT_OF_DANGER"
ESCAPE_DIRECTION = "ESCAPE_DIRECTION"
NO_ESCAPE_DIRECTION = "NO_ESCAPE_DIRECTION"
CORNER_BOMB = "CORNER_BOMB"
BAD_BOMB = "OWN_BOMB_THE_AGENT_CANNOT_DODGE"
RUN_INTO_ACTIVE_BOMB = "RUN_INTO_ACTIVE_BOMB"
SAFE_SQUARE = "SAFE_SQUARE"
WAIT_IN_EXPLOSION_AREA = "WAIT_IN_EXPLOSION_AREA"
DIED_TO_BEGIN = "DIED_TO_BEGIN"
REDUCE_BOMB_AWARD_WHEN_COINS_EXIST = "REDUCE_BOMB_AWARD_WHEN_COINS_EXIST"
BOMB_REPETITION = "BOMB_REPETITION"
WAIT_2 = "WAIT_2"


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
    self.bombhistory = []
    self.actionhistory = []
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

    if(e.BOMB_DROPPED in events):
        bombs = new_game_state["bombs"]
        bomb_positions = [bomb_pos for (bomb_pos, _) in bombs]
        self.bombhistory.append(bomb_positions)

    try:
        if(self.bombhistory[-1] == self.bombhistory[-3]) and (self.bombhistory[-2] == self.bombhistory[-4]):
            events.append(BOMB_REPETITION)
    except IndexError: len(self.bombhistory) < 4

    try:
        if(self.actionhistory[-1] == self.actionhistory[-2] == 'WAIT') and(self.actionhistory[-6] == 'BOMB'):
            events.append(WAIT_2)
    except IndexError: len(self.actionhistory) < 2

    if (new_game_features[5] == 1) and (e.MOVED_UP in events) and (new_game_features[0] == 0) and (new_game_features[4] < 8):
        events.append(GOT_CLOSER_TO_COIN)
    if (new_game_features[6] == 1) and (e.MOVED_DOWN in events) and (new_game_features[1] == 0) and (new_game_features[4] < 8):
        events.append(GOT_CLOSER_TO_COIN)
    if (new_game_features[7] == 1) and (e.MOVED_LEFT in events) and (new_game_features[2] == 0) and (new_game_features[4] < 8):
        events.append(GOT_CLOSER_TO_COIN)
    if (new_game_features[8] == 1) and (e.MOVED_RIGHT in events) and (new_game_features[3] == 0) and (new_game_features[4] < 8):
        events.append(GOT_CLOSER_TO_COIN)

    if old_game_state["self"][3] == new_game_state["self"][3]:
        events.append(STAYED_PUT)

    if e.COIN_COLLECTED not in events:
        events.append(NO_COIN_COLLECTED)

    if new_game_state["step"] > 2:
        if self.transitions[-2].action == opposite(self.transitions[-1].action) == self_action:
            events.append(WIGGLE_WIGGLE_WIGGLE)

    if new_game_features[10] < 5:
        if new_game_features[9]:
            events.append(IN_DANGER)
        else: 
            events.append(OUT_OF_DANGER)

    if new_game_features[10] > old_game_features[10]:
        events.append(GOT_AWAY_FROM_BOMB)
    elif new_game_features[10] < old_game_features[10]:
        events.append(GOT_CLOSER_TO_BOMB)

    # 11 - 14 escape route up, down, left ,right (1 possbile 0 not possible)
    if (old_game_features[11] == 1) and (e.MOVED_UP in events) and (old_game_features[15] == 0):
        events.append(ESCAPE_DIRECTION)
    if (old_game_features[12] == 1) and (e.MOVED_DOWN in events) and (old_game_features[15] == 0):
        events.append(ESCAPE_DIRECTION)
    if (old_game_features[13] == 1) and (e.MOVED_LEFT in events) and (old_game_features[15] == 0):
        events.append(ESCAPE_DIRECTION)
    if (old_game_features[14] == 1) and (e.MOVED_RIGHT in events) and (old_game_features[15] == 0):
        events.append(ESCAPE_DIRECTION)

    if (old_game_features[11] == 0) and (e.MOVED_UP in events) and (old_game_features[15] == 0):
        events.append(NO_ESCAPE_DIRECTION)
    if (old_game_features[12] == 0) and (e.MOVED_DOWN in events) and (old_game_features[15] == 0):
        events.append(NO_ESCAPE_DIRECTION)
    if (old_game_features[13] == 0) and (e.MOVED_LEFT in events) and (old_game_features[15] == 0):
        events.append(NO_ESCAPE_DIRECTION)
    if (old_game_features[14] == 0) and (e.MOVED_RIGHT in events) and (old_game_features[15] == 0):
        events.append(NO_ESCAPE_DIRECTION)

    corner = [(1,1), (1,15), (15,1), (15,15)]
    if(e.BOMB_DROPPED in events) and (new_game_state["self"][3] in corner):
        events.append(CORNER_BOMB)

    if(e.MOVED_UP in events) and (old_game_features[16] != 0):
        events.append(RUN_INTO_ACTIVE_BOMB)
    if(e.MOVED_DOWN in events) and (old_game_features[17] != 0):
        events.append(RUN_INTO_ACTIVE_BOMB)
    if(e.MOVED_LEFT in events) and (old_game_features[18] != 0):
        events.append(RUN_INTO_ACTIVE_BOMB)
    if(e.MOVED_RIGHT in events) and (old_game_features[19] != 0):
        events.append(RUN_INTO_ACTIVE_BOMB)

    # bomb dropped without any chances to escape this bomb
    if(e.BOMB_DROPPED in events) and (new_game_features[11] == new_game_features[12]
                                       == new_game_features[13] == new_game_features[14] == 0):
        events.append(BAD_BOMB)
    
    if(new_game_features[15] == 1):
        events.append(SAFE_SQUARE)
    
    if(new_game_features[15] == 0) and (e.WAITED in events):
        events.append(WAIT_IN_EXPLOSION_AREA)

    if(e.KILLED_SELF in events) and (old_game_state["step"] < 10):
        events.append(DIED_TO_BEGIN)

    if(e.BOMB_DROPPED in events) and (old_game_state["coins"]):
        events.append(REDUCE_BOMB_AWARD_WHEN_COINS_EXIST)

    # state_to_features is defined in callbacks.py
    # if(old_game_state["coins"]):
    #     transition = Transition(old_game_features, self_action, new_game_features, reward_from_events_coin(self, events), False)
    # else:

    transition = Transition(old_game_features, self_action, new_game_features, reward_from_events(self, events), False)


    self.transitions.append(transition)

    self.trainer.train_step(transition)

    self.actionhistory.append(self_action)


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
    # if(last_game_state["coins"]):
    #     transition = Transition(state_to_features(last_game_state), last_action, None, reward_from_events_coin(self, events), True)
    # else:
         
    transition = Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events), True)


    self.transitions.append(transition)

    self.trainer.train_step(transition)

    self.trainer.train_from_buffer(self.transitions)

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

def reward_from_events_coin(self, events: List[str]) -> int:
    game_rewards_coin = {
        e.COIN_COLLECTED: 5,
        # e.KILLED_OPPONENT: 0,
        e.BOMB_DROPPED: -5,
        # e.KILLED_SELF: -20,
        # e.GOT_KILLED: 0,
        # e.CRATE_DESTROYED: 20,
        WAIT_2: 15,
        GOT_CLOSER_TO_COIN: 2,
        GOT_AWAY_FROM_COIN: -1,
        # STAYED_PUT: -0.05,
        # WIGGLE_WIGGLE_WIGGLE: -2,
        # IN_DANGER: 0,
        # OUT_OF_DANGER: 0,
        # GOT_AWAY_FROM_BOMB: 30,
        # GOT_CLOSER_TO_BOMB: -50,
        # SAFE_SQUARE: 5,
        # ESCAPE_DIRECTION: 2.5,
        # NO_ESCAPE_DIRECTION: -5,
        # WAIT_IN_EXPLOSION_AREA: -5,
        # CORNER_BOMB: -10,
        RUN_INTO_ACTIVE_BOMB: -25,
        # BAD_BOMB: -100
        # NO_COIN_COLLECTED: -0.2,
        # DIED_TO_BEGIN: -15, 1
        # REDUCE_BOMB_AWARD_WHEN_COINS_EXIST: -8
        # BOMB_REPETITION: -20
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards_coin:
            reward_sum += game_rewards_coin[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 5,
        # e.KILLED_OPPONENT: 0,
        e.BOMB_DROPPED: 5,
        # e.KILLED_SELF: -5,
        # e.GOT_KILLED: 0,
        # e.CRATE_DESTROYED: 1,
        GOT_CLOSER_TO_COIN: 0.5,
        # GOT_AWAY_FROM_COIN: -0.75,
        # STAYED_PUT: -0.05,
        # WIGGLE_WIGGLE_WIGGLE: -2,
        # IN_DANGER: 0,
        # OUT_OF_DANGER: 0,
        # GOT_AWAY_FROM_BOMB: 1,
        # GOT_CLOSER_TO_BOMB: -1,
        # SAFE_SQUARE: 5,
        ESCAPE_DIRECTION: 2.5,
        # NO_ESCAPE_DIRECTION: -5,
        WAIT_IN_EXPLOSION_AREA: -5,
        CORNER_BOMB: -20,
        # RUN_INTO_ACTIVE_BOMB: -5,
        # BAD_BOMB: -100
        # NO_COIN_COLLECTED: -0.2,
        # DIED_TO_BEGIN: -15, 1
        # REDUCE_BOMB_AWARD_WHEN_COINS_EXIST: -8
        # BOMB_REPETITION: -20
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