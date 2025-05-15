import numpy as np
from src.constants import BOARD_WIDTH, BOARD_HEIGHT, CellType

AGENT_STATE_SIZE = 20  # 4 for direction + 4*4 for vision (4 directions, each with snake/green_apple/red_apple/nothing)

class State:
    def __init__(self):
        pass
        
    def get_state(self, game_manager):
        state = np.zeros(AGENT_STATE_SIZE)
        snake = game_manager.snake
        head_x, head_y = snake.head

        # Current direction one-hot encoding (4 values)
        state[0] = 1 if snake.direction == (-1, 0) else 0  # LEFT
        state[1] = 1 if snake.direction == (1, 0) else 0   # RIGHT
        state[2] = 1 if snake.direction == (0, -1) else 0  # UP
        state[3] = 1 if snake.direction == (0, 1) else 0   # DOWN
        
        # Define the four directions to scan
        directions = [
            # (dx, dy, state_offset)
            (-1, 0, 4),   # LEFT
            (1, 0, 8),    # RIGHT
            (0, -1, 12),  # UP
            (0, 1, 16)    # DOWN
        ]
        
        for dx, dy, offset in directions:
            # Variables to track what the snake can see in this direction
            sees_snake_body = False
            sees_green_apple = False
            sees_red_apple = False
            distance_to_snake = float('inf')
            distance_to_green = float('inf')
            distance_to_red = float('inf')
            
            x, y = head_x, head_y
            distance = 0
            
            while True:
                x += dx
                y += dy
                distance += 1
                
                # Check if we're out of bounds before calling get_cell_type
                if x < 0 or x >= BOARD_WIDTH or y < 0 or y >= BOARD_HEIGHT:
                    break
                    
                cell_type = game_manager.board.get_cell_type(x, y)
                
                if cell_type == CellType.WALL:
                    break
                
                if (x, y) in snake.body[1:]:
                    sees_snake_body = True
                    distance_to_snake = distance
                    break
                
                for apple in game_manager.apples:
                    if (x, y) == apple.position:
                        if apple.color == "green":
                            sees_green_apple = True
                            distance_to_green = distance
                        else: 
                            sees_red_apple = True
                            distance_to_red = distance
            
            state[offset] = 1 if sees_snake_body else 0
            state[offset + 1] = 1 if sees_green_apple else 0
            state[offset + 2] = 1 if sees_red_apple else 0
            
            min_distance = min(
                distance_to_snake if sees_snake_body else float('inf'),
                distance_to_green if sees_green_apple else float('inf'),
                distance_to_red if sees_red_apple else float('inf')
            )
            
            # If nothing was seen, use a small value close to 0
            if min_distance == float('inf'):
                state[offset + 3] = 0.01
            else:
                # Normalize to emphasize close objects
                state[offset + 3] = 1.0 / min_distance
        
        return state
                