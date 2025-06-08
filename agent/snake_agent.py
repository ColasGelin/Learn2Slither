import numpy as np
import random
import torch

from src.constants import (UP, DOWN, LEFT, RIGHT,
                           AGENT_STATE_SIZE, AGENT_ACTION_SIZE)
from .replay_buffer import ReplayBuffer
from .deep_q_network import DQNAgent
from .state import State


class SnakeAgent:

    def __init__(self,
                 learning_rate=0.002,
                 gamma=0.99,
                 epsilon_start=1.0,
                 batch_size=64,
                 update_target_frequency=100,
                 step_counter=0,
                 use_smart_exploration=False):
        self.state_processor = State()
        self.replay_buffer = ReplayBuffer()
        self.agent = DQNAgent(
            state_size=AGENT_STATE_SIZE,
            action_size=AGENT_ACTION_SIZE,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
        )
        self.use_smart_exploration = use_smart_exploration

        self.actions = [UP, DOWN, LEFT, RIGHT]

        self.batch_size = batch_size
        self.update_target_frequency = update_target_frequency
        self.step_counter = step_counter

    def get_action(self, state_or_game_manager):
        if isinstance(state_or_game_manager, np.ndarray):
            current_state = state_or_game_manager
        else:
            current_state = self.state_processor.get_state(
                state_or_game_manager)

        # Exploration
        if random.random() < self.agent.epsilon:
            if self.use_smart_exploration:
                valid_actions = self._get_valid_actions(current_state)
                if valid_actions:
                    action_idx = self.actions.index(
                        random.choice(valid_actions))
                else:
                    action_idx = self.agent.get_action(current_state)
            else:
                action_idx = self.agent.get_action(current_state)
        else:
            # Exploitation
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(
                self.agent.device)
            q_values = self.agent.policy_net(state_tensor)
            action_idx = torch.argmax(q_values).item()

        return self.actions[action_idx]

    def train(self, state, action_idx, reward, next_state, done):
        self.replay_buffer.add(state, action_idx, reward, next_state, done)

        self.step_counter += 1
        if (len(self.replay_buffer) > self.batch_size
                and self.step_counter % self.update_target_frequency == 0):
            experiences = self.replay_buffer.sample(self.batch_size)
            loss = self.agent.learn(experiences)
            return loss
        return None

    def action_to_idx(self, action):
        return self.actions.index(action)

    def save_model(self, path):
        self.agent.save(path)

    def load_model(self, path):
        self.agent.load(path)

    # Add method to predict valid moves
    def _get_valid_actions(self, state):
        """Get actions that don't lead to immediate death based on state"""
        valid_actions = []

        current_direction_idx = -1
        for i in range(4):
            if state[i] == 1:
                current_direction_idx = i
                break

        direction_offsets = [4, 9, 14, 19]
        directions = [LEFT, RIGHT, UP, DOWN]

        # Skip if trying to go backwards (into snake own body)
        for i, direction in enumerate(directions):
            if current_direction_idx != -1:
                if (i == 0 and current_direction_idx == 1) or \
                   (i == 1 and current_direction_idx == 0) or \
                   (i == 2 and current_direction_idx == 3) or \
                   (i == 3 and current_direction_idx == 2):
                    continue

            # Check if there's an immediate wall or snake in this direction
            wall_indicator = state[direction_offsets[i] + 3]
            snake_indicator = state[direction_offsets[i]]
            dist_to_obstacle = state[direction_offsets[i] + 4]

            if (wall_indicator == 0 or dist_to_obstacle < 0.99) and \
               (snake_indicator == 0 or dist_to_obstacle < 0.99):
                valid_actions.append(direction)

        return valid_actions
