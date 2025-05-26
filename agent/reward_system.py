

class RewardSystem:

    def __init__(self,
                 base_move_penalty=-0.05,
                 collision_penalty=-100,
                 apple_reward=10,
                 bad_apple_penalty=-25,
                 approach_reward_factor=0.5):
        self.base_move_penalty = base_move_penalty
        self.collision_penalty = collision_penalty
        self.apple_reward = apple_reward
        self.bad_apple_penalty = bad_apple_penalty
        self.approach_reward_factor = approach_reward_factor

        # Initialize tracking variables for multiple players
        self.prev_distance_to_apple = {}  # Dictionary to track per player

    def calculate_reward(self,
                         game_manager,
                         game_over,
                         prev_score,
                         new_score,
                         num_players=1,
                         player_index=0):
        """
        Calculate the reward based on the game state and outcome.
        Args:
            game_manager: The current game state
            game_over: Whether the game has ended for this player
            prev_score: Score before the action
            new_score: Score after the action
            num_players: Number of players in the game (default: 1)
            player_index: Index of the player to calculate reward for
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
            if player_index in self.prev_distance_to_apple:
                self.prev_distance_to_apple[player_index] = None
            return reward

        elif new_score < prev_score:
            # Penalty for eating red apple
            reward += self.bad_apple_penalty
            return reward

        # Distance-based rewards (approaching/avoiding)
        if num_players == 1:
            head_pos = game_manager.snake.head
        else:
            if player_index >= len(
                    game_manager.snakes
            ) or not game_manager.snake_alive[player_index]:
                return 0  # No reward for inactive snake
            head_pos = game_manager.snakes[player_index].head

        # Find distance to nearest green apple
        min_distance_to_apple = float('inf')
        for apple in game_manager.apples:
            if apple.color == "green":
                apple_pos = apple.position
                distance = self._calculate_distance(head_pos, apple_pos)
                min_distance_to_apple = min(min_distance_to_apple, distance)

        # If we have a previous distance to compare with
        if (player_index in self.prev_distance_to_apple
                and self.prev_distance_to_apple[player_index] is not None
                and min_distance_to_apple < float('inf')):
            distance_change = self.prev_distance_to_apple[
                player_index] - min_distance_to_apple
            distance_reward = distance_change * self.approach_reward_factor * (
                1 / (min_distance_to_apple + 1))
            reward += distance_reward

        # Update distance for next step
        if min_distance_to_apple < float('inf'):
            self.prev_distance_to_apple[player_index] = min_distance_to_apple

        # Default small penalty to encourage efficiency
        reward += self.base_move_penalty

        # Additional rewards specific to multiplayer (if applicable)
        if num_players > 1:
            # Add bonus for being longer than other snakes
            current_length = len(game_manager.snakes[player_index].body)
            other_snakes_avg_length = 0
            active_other_snakes = 0

            for i, snake in enumerate(game_manager.snakes):
                if i != player_index and game_manager.snake_alive[i]:
                    other_snakes_avg_length += len(snake.body)
                    active_other_snakes += 1

            if active_other_snakes > 0:
                other_snakes_avg_length /= active_other_snakes
                length_advantage = (current_length -
                                    other_snakes_avg_length) * 0.01
                reward += length_advantage  # Small bonus for being longer

        return reward

    def _calculate_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def reset(self):
        """Reset the reward system for a new episode"""
        self.prev_distance_to_apple = {}
