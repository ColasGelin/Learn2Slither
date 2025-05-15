import numpy as np
from src.constants import UP, DOWN, LEFT, RIGHT, AGENT_STATE_SIZE, AGENT_ACTION_SIZE
from .replay_buffer import ReplayBuffer
from .deep_q_network import DQNAgent
from .state import State

class SnakeAgent:
    def __init__(self, learning_rate=0.001, gamma=0.99, epsilon_start=1.0, 
                 batch_size=64, update_target_frequency=100, step_counter=0):
        self.state_processor = State()
        self.replay_buffer = ReplayBuffer()
        self.agent = DQNAgent(
            state_size=AGENT_STATE_SIZE,
            action_size=AGENT_ACTION_SIZE,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start
        )
        
        self.actions = [UP, DOWN, LEFT, RIGHT]
        
        self.batch_size = batch_size
        self.update_target_frequency = update_target_frequency
        self.step_counter = step_counter
        
    def get_action(self, state_or_game_manager):
        # Check if the input is already a state array or if it's a game_manager
        if isinstance(state_or_game_manager, np.ndarray):
            current_state = state_or_game_manager
        else:
            # It's a game_manager object
            current_state = self.state_processor.get_state(state_or_game_manager)
        
        action_idx = self.agent.get_action(current_state)
        return self.actions[action_idx]
        
    def train(self, state, action_idx, reward, next_state, done):
        self.replay_buffer.add(state, action_idx, reward, next_state, done)
        
        self.step_counter += 1
        if (len(self.replay_buffer) > self.batch_size and 
            self.step_counter % self.update_target_frequency == 0):
            experiences = self.replay_buffer.sample(self.batch_size)
            loss = self.agent.learn(experiences, self.batch_size)
            return loss
        return None
    
    def action_to_idx(self, action):
        return self.actions.index(action)
    
    def save_model(self, path):
        self.agent.save(path)

    def load_model(self, path):
        self.agent.load(path)