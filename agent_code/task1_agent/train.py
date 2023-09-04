from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

import numpy as np

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
SAME_POS_EVENT = "SAME_POSITION"
WALL_BUMP_EVENT = "WALL_BUMP"
COIN_DIST_DECREASE_EVENT = "COIN_DIST_DECREASE"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, s', r)
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

def coin_dist(game_state):
    targets = game_state["coins"]
    start = game_state["self"][3]

    # calculates the distance between the agent and the nearest coin
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()
    return best_dist

def same_pos(transitions):
    if(len(transitions) != 3):
        return False
    state0, action, new_state, reward = transitions.pop()
    state1, action, new_state, reward = transitions.pop()
    state2, action, new_state, reward = transitions.pop()

    x,y = state0
    a,b = state1
    c,d = state2

    if((x == a and y == b) or (a == c and b == d)):
        return True
    else:
        return False
    
def wall_bump(old_game_state, new_game_state):
    old_pos = old_game_state["self"][3]
    new_pos = new_game_state["self"][3]

    if(new_pos == old_pos):
        return True
    else:
        return False



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

    # Idea: Add your own events to hand out rewards
    if same_pos(self.transitions):
        events.append(SAME_POS_EVENT)

    if wall_bump(old_game_state, new_game_state):
        events.append(WALL_BUMP_EVENT)   

    if (coin_dist(new_game_state) < coin_dist(old_game_state)) and (not same_pos(self.transitions)):
        events.append(COIN_DIST_DECREASE_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

    # Q-learning update
    alpha = 0.9     # learning rate
    gamma = 0.95    # discount factor
    weights = self.model
    round = new_game_state["round"]

    ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    for idx, a in ACTIONS:
        if a == self_action:
            action = idx
    
    a,b,c,reward_old = self.transitions.pop()
    a,b,c,reward_new = self.transitions.pop()
    reward = reward_new - reward_old

    # Q-update rule
    weights[round, action] = (1- alpha) * weights[round, action] + alpha *(reward + gamma * weights[round +1,].max())


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
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        WALL_BUMP_EVENT: -.8,
        SAME_POS_EVENT: -.5,
        COIN_DIST_DECREASE_EVENT: .1
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
