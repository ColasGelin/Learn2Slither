import numpy as np

class RewardSystem:
    def __init__(self, base_move_penalty=-0.05, 
                 collision_penalty=-100, 
                 apple_reward=10,
                 bad_apple_penalty=-25,
                 approach_reward_factor=0.5):
        self.base_move_penalty = base_move_penalty
        self.collision_penalty = collision_penalty
        self.apple_reward = apple_reward
        self.bad_apple_penalty = bad_apple_penalty
        self.approach_reward_factor = approach_reward_factor
        
        # Initialize tracking variables
        self.prev_distance_to_apple = None
        self.last_apple_positions = []
        
    def calculate_reward(self, game_manager, game_over, prev_score, new_score):
        """
        Calculate the reward based on the game state and outcome.
        
        Args:
            game_manager: The current game state
            game_over: Whether the game has ended
            prev_score: Score before the action
            new_score: Score after the action
            
        Returns:
            float: The calculated reward
        """
        reward = 0
        
        # Game over penalty
        if game_over:
            return self.collision_penalty
            
        # Apple rewards/penalties
        if new_score > prev_score:
            # Big reward for eating green apple
            reward += self.apple_reward * (new_score - prev_score)
            # Reset distance tracking after eating an apple
            self.prev_distance_to_apple = None
            return reward
            
        elif new_score < prev_score:
            # Penalty for eating red apple
            reward += self.bad_apple_penalty
            return reward
            
        # Distance-based rewards (approaching/avoiding)
        head_pos = game_manager.snake.head
        
        # Find distance to nearest green apple
        min_distance_to_apple = float('inf')
        for apple in game_manager.apples:
            if apple.color == "green":
                apple_pos = apple.position
                distance = self._calculate_distance(head_pos, apple_pos)
                min_distance_to_apple = min(min_distance_to_apple, distance)
        
        # If we have a previous distance to compare with
        if self.prev_distance_to_apple is not None and min_distance_to_apple < float('inf'):
            # Reward for getting closer to green apple, penalize for moving away
            distance_change = self.prev_distance_to_apple - min_distance_to_apple
            distance_reward = distance_change * self.approach_reward_factor
            reward += distance_reward
        
        # Update distance for next step
        if min_distance_to_apple < float('inf'):
            self.prev_distance_to_apple = min_distance_to_apple
            
        # Default small penalty to encourage efficiency
        reward += self.base_move_penalty
        
        return reward
    
    def _calculate_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
    def reset(self):
        """Reset the reward system for a new episode"""
        self.prev_distance_to_apple = None
        self.last_apple_positions = []