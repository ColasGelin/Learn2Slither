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
            distance_reward = distance_change * self.approach_reward_factor * (1 / (min_distance_to_apple + 1))
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
    
    def calculate_dual_snake_reward(self, game_manager, game_over, prev_score, current_score, is_snake_1=True):
        """
        Calculate reward for a snake in dual-snake environment with enhanced incentives
        while keeping the score system simple (just based on length)
        """
        if hasattr(game_manager, 'multiple_snake') and game_manager.multiple_snake:
            # Get both snakes
            snake = game_manager.snake_1 if is_snake_1 else game_manager.snake_2
            opponent = game_manager.snake_2 if is_snake_1 else game_manager.snake_1
            
            # Initialize reward
            reward = 0
            
            # Base reward from score difference (apple eating/length changes)
            score_diff = current_score - prev_score
            if score_diff > 0:
                reward += 10  # Reward for growing
            elif score_diff < 0:
                reward -= 5   # Penalty for shrinking
                
            # Survival reward
            if not game_over or (game_over and ((is_snake_1 and game_manager.snake_1_alive) or 
                                              (not is_snake_1 and game_manager.snake_2_alive))):
                reward += 0.1  # Small reward for staying alive
            
            # Collision-based rewards/penalties (only affects rewards, not actual score)
            if game_over:
                # Check if this snake died
                if (is_snake_1 and not game_manager.snake_1_alive) or (not is_snake_1 and not game_manager.snake_2_alive):
                    reward -= 15  # Penalty for dying
                    
                    # Additional penalty for causing your own death by collision
                    if (is_snake_1 and hasattr(game_manager, 'snake_1_caused_collision') and game_manager.snake_1_caused_collision) or \
                       (not is_snake_1 and hasattr(game_manager, 'snake_2_caused_collision') and game_manager.snake_2_caused_collision):
                        reward -= 10  # Extra penalty for running into opponent
                
                # Reward for outliving opponent
                if (is_snake_1 and game_manager.snake_1_alive and not game_manager.snake_2_alive) or \
                   (not is_snake_1 and game_manager.snake_2_alive and not game_manager.snake_1_alive):
                    reward += 20  # Major bonus for outliving opponent
                
                # Reward for winning head-to-head collision
                if (is_snake_1 and hasattr(game_manager, 'snake_1_won_head_collision') and game_manager.snake_1_won_head_collision) or \
                   (not is_snake_1 and hasattr(game_manager, 'snake_2_won_head_collision') and game_manager.snake_2_won_head_collision):
                    reward += 15  # Bonus for winning head-to-head
                
                # Penalty for losing head-to-head collision
                if (is_snake_1 and hasattr(game_manager, 'snake_2_won_head_collision') and game_manager.snake_2_won_head_collision) or \
                   (not is_snake_1 and hasattr(game_manager, 'snake_1_won_head_collision') and game_manager.snake_1_won_head_collision):
                    reward -= 10  # Penalty for losing head-to-head
                    
                # Score comparison at game end
                if game_manager.score_1 > game_manager.score_2:
                    if is_snake_1:
                        reward += 5  # Bonus for having higher score at end
                    else:
                        reward -= 3  # Penalty for having lower score at end
                elif game_manager.score_2 > game_manager.score_1:
                    if not is_snake_1:
                        reward += 5  # Bonus for having higher score at end
                    else:
                        reward -= 3  # Penalty for having lower score at end
            
            # Proximity rewards (strategic positioning)
            head_pos = snake.head
            opponent_head_pos = opponent.head
            distance_to_opponent = self._calculate_distance(head_pos, opponent_head_pos)
            
            # Reward for being at strategic distance from opponent
            if snake.length > opponent.length:
                # If longer, reward for being closer (aggressive stance)
                if distance_to_opponent <= 3:
                    reward += 0.2
            else:
                # If shorter, reward for maintaining safe distance
                if distance_to_opponent > 3:
                    reward += 0.1
            
            # Apple proximity rewards
            min_distance_to_green_apple = float('inf')
            for apple in game_manager.apples:
                if apple.color == "green":
                    apple_pos = apple.position
                    distance = self._calculate_distance(head_pos, apple_pos)
                    min_distance_to_green_apple = min(min_distance_to_green_apple, distance)
            
            # Small reward for moving toward green apples
            if min_distance_to_green_apple < float('inf'):
                reward += 0.5 / (min_distance_to_green_apple + 1)
                
            return reward
        else:
            # Original reward calculation for single snake
            return self.calculate_reward(game_manager, game_over, prev_score, current_score)