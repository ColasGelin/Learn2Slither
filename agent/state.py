import numpy as np
from src.constants import BOARD_WIDTH, BOARD_HEIGHT, AGENT_STATE_SIZE


class State:

    def __init__(self):
        self.max_possible_distance = max(BOARD_WIDTH, BOARD_HEIGHT)

    def _normalize_distance(self, distance):
        # Inverse distance scaling (closer objects have higher values)
        return 1.0 / distance

    def get_state(self, game_manager, snake_index=0):
        multi_player = hasattr(game_manager, 'snakes') and len(
            game_manager.snakes) > 1

        if multi_player:
            if snake_index >= len(game_manager.snakes):
                raise ValueError(
                    f"Invalid snake index {snake_index}. "
                    f"Only {len(game_manager.snakes)} snakes available."
                )

            # Get the current snake and opponent snakes
            current_snake = game_manager.snakes[snake_index]
            opponent_snakes = [
                snake for i, snake in enumerate(game_manager.snakes)
                if i != snake_index and game_manager.snake_alive[i]
            ]

            # Process state with awareness of all snakes
            return self.process_multi_snake_state(game_manager, current_snake,
                                                  opponent_snakes)
        else:
            # Single snake mode
            return self.process_single_snake_state(game_manager)

    def process_single_snake_state(self, game_manager):
        snake = game_manager.snakes[0]
        state = np.zeros(AGENT_STATE_SIZE)

        state[0] = 1 if snake.direction == (-1, 0) else 0  # LEFT
        state[1] = 1 if snake.direction == (1, 0) else 0  # RIGHT
        state[2] = 1 if snake.direction == (0, -1) else 0  # UP
        state[3] = 1 if snake.direction == (0, 1) else 0  # DOWN

        # Define the four directions to scan
        directions = [
            # (dx, dy, state_offset)
            (-1, 0, 4),  # LEFT
            (1, 0, 9),  # RIGHT
            (0, -1, 14),  # UP
            (0, 1, 19)  # DOWN
        ]

        for dx, dy, offset in directions:
            self._scan_direction(state, offset, snake.head, dx, dy,
                                 [snake.body[1:]], [], game_manager.apples)

        return state

    def process_multi_snake_state(self, game_manager, current_snake,
                                  opponent_snakes):
        state = np.zeros(AGENT_STATE_SIZE)

        # Current direction one-hot encoding (4 values)
        state[0] = 1 if current_snake.direction == (-1, 0) else 0  # LEFT
        state[1] = 1 if current_snake.direction == (1, 0) else 0  # RIGHT
        state[2] = 1 if current_snake.direction == (0, -1) else 0  # UP
        state[3] = 1 if current_snake.direction == (0, 1) else 0  # DOWN

        # Define the four directions to scan
        directions = [
            # (dx, dy, state_offset)
            (-1, 0, 4),  # LEFT
            (1, 0, 9),  # RIGHT
            (0, -1, 14),  # UP
            (0, 1, 19)  # DOWN
        ]

        # Get obstacles: own body (except head) and opponent snakes' bodies
        own_body = [current_snake.body[1:]]
        opponent_bodies = [snake.body for snake in opponent_snakes]

        for dx, dy, offset in directions:
            self._scan_direction(state, offset, current_snake.head, dx, dy,
                                 own_body, opponent_bodies,
                                 game_manager.apples)

        return state

    def _scan_direction(self, state, offset, head_pos, dx, dy, own_bodies,
                        opponent_bodies, apples):
        head_x, head_y = head_pos
        object_detected = "none"
        distance = 0

        # Start looking from the adjacent cell in the given direction
        x, y = head_x + dx, head_y + dy
        distance = 1

        # Continue looking in this direction until we hit something
        while 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT:
            position = (x, y)
            for body in own_bodies:
                if position in body:
                    object_detected = "snake"
                    break

            if object_detected != "none":
                break

            # Check for opponent snakes
            for body in opponent_bodies:
                if position in body:
                    object_detected = "opponent_snake"
                    break

            if object_detected != "none":
                break

            # Check for apples
            for apple in apples:
                if position == apple.position:
                    object_detected = f"{apple.color}_apple"
                    break

            if object_detected != "none":
                break

            # Move to next cell in this direction
            x += dx
            y += dy
            distance += 1

        # If exited the loop without finding
        # anything inside the board, it's a wall
        if object_detected == "none" and not (0 <= x < BOARD_WIDTH
                                              and 0 <= y < BOARD_HEIGHT):
            object_detected = "wall"

        # Update state based on what was detected
        state[offset] = 1 if object_detected in ["snake", "opponent_snake"
                                                 ] else 0
        state[offset + 1] = 1 if object_detected == "green_apple" else 0
        state[offset + 2] = 1 if object_detected == "red_apple" else 0
        state[offset + 3] = 1 if object_detected == "wall" else 0

        # Add normalized distance (0 if nothing found)
        if object_detected == "none":
            state[offset + 4] = 0
        else:
            state[offset + 4] = self._normalize_distance(distance)
