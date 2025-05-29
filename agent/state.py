import numpy as np
from src.constants import BOARD_WIDTH, BOARD_HEIGHT, AGENT_STATE_SIZE


class State:

    def __init__(self):
        pass

    def _normalize_distance(self, distance):
        # Inverse distance scaling: closer objects have higher values
        return 1.0 / distance

    def get_state(self, game_manager, snake_index=0):
        """
        Returns a numpy array representing the state for the specified snake.

        State representation:
        - state[0:4]: One-hot encoding of the snake's current direction:
            [LEFT, RIGHT, UP, DOWN]
        - For each direction (LEFT, RIGHT, UP, DOWN), the following 5 features are included:
            - [i]: 1 if the first object in this direction is a snake (own or opponent), else 0
            - [i+1]: 1 if the first object is a green apple, else 0
            - [i+2]: 1 if the first object is a red apple, else 0
            - [i+3]: 1 if the first object is a wall, else 0
            - [i+4]: Normalized inverse distance to the first object in this direction (0 if nothing found)
        """
        multiplayer = hasattr(game_manager, 'snakes') and len(
            game_manager.snakes) > 1

        if not multiplayer:
            return self.process_single_snake_state(game_manager)
        else:
            if snake_index >= len(game_manager.snakes):
                raise ValueError(
                    f"Invalid snake index {snake_index}. "
                    f"Only {len(game_manager.snakes)} snakes available."
                )

            current_snake = game_manager.snakes[snake_index]
            opponent_snakes = [
                snake for i, snake in enumerate(game_manager.snakes)
                if i != snake_index and game_manager.snake_alive[i]
            ]

            return self.process_multi_snake_state(game_manager, current_snake,
                                                  opponent_snakes)
            

    def process_single_snake_state(self, game_manager):
        snake = game_manager.snake
        state = np.zeros(AGENT_STATE_SIZE)
        head_x, head_y = snake.head

        state[0] = 1 if snake.direction == (-1, 0) else 0  # LEFT
        state[1] = 1 if snake.direction == (1, 0) else 0  # RIGHT
        state[2] = 1 if snake.direction == (0, -1) else 0  # UP
        state[3] = 1 if snake.direction == (0, 1) else 0  # DOWN

        directions = [
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
        """Process state for multi-snake environment"""
        state = np.zeros(AGENT_STATE_SIZE)
        head_x, head_y = current_snake.head

        state[0] = 1 if current_snake.direction == (-1, 0) else 0  # LEFT
        state[1] = 1 if current_snake.direction == (1, 0) else 0  # RIGHT
        state[2] = 1 if current_snake.direction == (0, -1) else 0  # UP
        state[3] = 1 if current_snake.direction == (0, 1) else 0  # DOWN

        directions = [
            (-1, 0, 4),  # LEFT
            (1, 0, 9),  # RIGHT
            (0, -1, 14),  # UP
            (0, 1, 19)  # DOWN
        ]

        # Get own body and opponent snakes' bodies
        own_body = [current_snake.body[1:]] 
        opponent_bodies = [snake.body for snake in opponent_snakes]

        for dx, dy, offset in directions:
            self._scan_direction(state, offset, current_snake.head, dx, dy,
                                 own_body, opponent_bodies,
                                 game_manager.apples)

        return state

    def _scan_direction(self, state, offset, head_pos, dx, dy, own_bodies,
                        opponent_bodies, apples):
        """
        Scan in a direction from the head position and update state accordingly
        Args:
            state: numpy array to update with the scan results
            offset: index offset for state array
            head_pos: (x,y) position of the snake's head
            dx, dy: direction to scan
            own_bodies: list of lists of positions for own snake body parts
            opponent_bodies: list of lists of positions
            for opponent snake bodies
            apples: list of apple objects
        """
        head_x, head_y = head_pos
        object_detected = "none"

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

            for body in opponent_bodies:
                if position in body:
                    object_detected = "opponent_snake"
                    break

            for apple in apples:
                if position == apple.position:
                    object_detected = f"{apple.color}_apple"
                    break

            x += dx
            y += dy
            distance += 1

        if not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT):
            object_detected = "wall"

        # Update state
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
