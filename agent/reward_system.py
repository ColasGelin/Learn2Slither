

class RewardSystem:

    def __init__(self,
                 base_move_reward=-0.01,
                 collision_penalty=-100,
                 apple_reward=10,
                 bad_apple_penalty=-25,
                 approach_reward_factor=0.5):
        self.base_move_reward = base_move_reward
        self.collision_penalty = collision_penalty
        self.apple_reward = apple_reward
        self.bad_apple_penalty = bad_apple_penalty
        self.approach_reward_factor = approach_reward_factor

        # Initialize tracking variables for multiple players
        self.prev_distance_to_apple = {}  # Dictionary to track per player

    def calculate_reward(self,
                         game_manager,
                         just_died,
                         prev_score,
                         new_score,
                         num_players=1,
                         player_index=0):
        reward = 0

        # Death penalty
        if just_died:
            return self.collision_penalty

        # Apple reward
        if new_score > prev_score:
            score_multiplier = 1 + (new_score * 0.1)
            apple_gain = new_score - prev_score
            reward += self.apple_reward * (apple_gain) * score_multiplier
            if player_index in self.prev_distance_to_apple:
                self.prev_distance_to_apple[player_index] = None
            return reward
        elif new_score < prev_score:
            reward += self.bad_apple_penalty
            return reward

        # ignore dead snakes
        if player_index >= len(
                game_manager.snakes
        ) or not game_manager.snake_alive[player_index]:
            return 0

        head_pos = game_manager.snakes[player_index].head
        min_distance_to_apple = float('inf')

        # Check for green apples in horizontal or vertical line from head
        for apple in game_manager.apples:
            if apple.color == "green":
                apple_pos = apple.position

                if apple_pos[0] == head_pos[0] or apple_pos[1] == head_pos[1]:
                    distance = self._calculate_distance(head_pos, apple_pos)
                    min_distance_to_apple = min(min_distance_to_apple,
                                                distance)

        # apple approach reward
        if (player_index in self.prev_distance_to_apple
                and self.prev_distance_to_apple[player_index] is not None
                and min_distance_to_apple < float('inf')):
            prev_distance = self.prev_distance_to_apple[player_index]
            distance_change = prev_distance - min_distance_to_apple

            # scaling approach reward
            if distance_change > 0:
                approach_reward = distance_change * self.approach_reward_factor
                approach_reward *= (2.0 / (min_distance_to_apple + 1))
                reward += approach_reward
            elif distance_change < 0:
                penalty = abs(distance_change) * 0.2
                reward -= penalty

        if min_distance_to_apple < float('inf'):
            self.prev_distance_to_apple[player_index] = min_distance_to_apple

        # Base move reward
        reward += self.base_move_reward

        # Multiplayer length advantage
        if num_players > 1:
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
                reward += length_advantage

        return reward

    def _calculate_distance(self, pos1, pos2):
        if pos1[0] == pos2[0]:
            return abs(pos1[1] - pos2[1])
        elif pos1[1] == pos2[1]:
            return abs(pos1[0] - pos2[0])

    def reset(self):
        self.prev_distance_to_apple = {}
