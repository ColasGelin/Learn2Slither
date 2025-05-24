import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.constants import AGENT_STATE_SIZE, AGENT_ACTION_SIZE


class DQN(nn.Module):
    def __init__(self, input_size=AGENT_STATE_SIZE, output_size=AGENT_ACTION_SIZE, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """ Returns the Q-values for the given state. """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, state_size=AGENT_STATE_SIZE, 
                 action_size=AGENT_ACTION_SIZE, 
                 learning_rate=0.001, gamma=0.99, 
                 epsilon_start=0.9, epsilon_min=0.2, epsilon_decay=0.998):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(input_size=state_size, output_size=action_size).to(self.device)
        self.target_net = DQN(input_size=state_size, output_size=action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Epsilon (ε): The probability of choosing a random action instead of the best action
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Gamma (γ) determines how much the agent values future rewards compared to immediate rewards:
        self.gamma = gamma
        self.action_size = action_size
        
        # Used to know when to update the target network
        self.learn_step_counter = 0
        
    def get_action(self, state):
        """ Returns the action to take based on the current state. """
        
        # Exploration ?
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            q_values = self.policy_net(state)
            action = torch.argmax(q_values).item()
        return action

    def learn(self, experiences, batch_size=None):
        states, actions, rewards, next_states, dones = experiences
        
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states_tensor).gather(1, actions_tensor)

        next_actions = self.policy_net(next_states_tensor).max(1)[1].unsqueeze(1)
        next_q_values = self.target_net(next_states_tensor).gather(1, next_actions)

        # Calculate target Q-values using Bellman equation
        target_q_values = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)
        
        loss = F.mse_loss(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.learn_step_counter += 1
        if self.learn_step_counter % 1000 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()
    
    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        
    def load(self, path):
        if torch.cuda.is_available():
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']