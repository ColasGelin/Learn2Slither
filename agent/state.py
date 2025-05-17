import numpy as np
from src.constants import BOARD_WIDTH, BOARD_HEIGHT, CellType, AGENT_STATE_SIZE

class State:
    def __init__(self):
        # We'll still use this for normalization, but only based on what the snake sees
        self.max_possible_distance = max(BOARD_WIDTH, BOARD_HEIGHT)
        
    def get_state(self, game_manager):
        state = np.zeros(AGENT_STATE_SIZE)
        snake = game_manager.snake
        head_x, head_y = snake.head

        # Current direction one-hot encoding (4 values)
        state[0] = 1 if snake.direction == (-1, 0) else 0  # LEFT
        state[1] = 1 if snake.direction == (1, 0) else 0   # RIGHT
        state[2] = 1 if snake.direction == (0, -1) else 0  # UP
        state[3] = 1 if snake.direction == (0, 1) else 0   # DOWN
        
        # Define the four directions to scan (limited to what the snake can see)
        directions = [
            # (dx, dy, state_offset)
            (-1, 0, 4),   # LEFT
            (1, 0, 9),   # RIGHT
            (0, -1, 14),  # UP
            (0, 1, 19)    # DOWN
        ]
        
        for dx, dy, offset in directions:
            # Default: nothing detected in this direction
            object_detected = "none"
            distance = 0
            
            # Start looking from the adjacent cell in the given direction
            x, y = head_x + dx, head_y + dy
            distance = 1  # Start with distance 1 (adjacent cell)
            
            # Continue looking in this direction until we hit something
            while 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT:
                # Check for snake body
                if (x, y) in snake.body[1:]:
                    object_detected = "snake"
                    break
                
                # Check for apples
                apple_found = False
                for apple in game_manager.apples:
                    if (x, y) == apple.position:
                        if apple.color == "green":
                            object_detected = "green_apple"
                        else:
                            object_detected = "red_apple"
                        apple_found = True
                        break
                
                if apple_found:
                    break
                
                # Move to next cell in this direction
                x += dx
                y += dy
                distance += 1
            
            # If we exited the loop without finding anything, it's a wall
            if object_detected == "none" and not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT):
                object_detected = "wall"
            
            # Set binary flags based on what was detected
            state[offset] = 1 if object_detected == "snake" else 0
            state[offset + 1] = 1 if object_detected == "green_apple" else 0
            state[offset + 2] = 1 if object_detected == "red_apple" else 0
            state[offset + 3] = 1 if object_detected == "wall" else 0
            
            # Add normalized distance (0 if nothing found)
            if object_detected == "none":
                state[offset + 4] = 0
            else:
                # Normalize distance to [0,1] range with higher values indicating closer objects
                state[offset + 4] = self._normalize_distance(distance)
        
        return state
    
    def _normalize_distance(self, distance):
        # Inverse distance scaling: closer objects have higher values
        return 1.0 / distance