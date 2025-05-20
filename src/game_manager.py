from src.constants import BOARD_WIDTH, BOARD_HEIGHT, CellType
from src.snake import Snake
from src.board import Board
from src.apple import Apple


class GameManager:
    def __init__(self, board_width=BOARD_WIDTH, board_height=BOARD_HEIGHT, num_green_apples=2, num_red_apples=1):
        self.board = Board(board_width, board_height)
        self.num_green_apples = num_green_apples
        self.num_red_apples = num_red_apples
        self.reset_game()
        
    def reset_game(self):
        self.board = Board(self.board.width, self.board.height)
        self.snake = Snake(self.board.width, self.board.height)
        self.apples = []
        self.place_initial_apples()
        self.score = 3
        self.game_over = False
        self.time_steps = 0
        
    def get_current_apple_coords(self):
        return [apple.position for apple in self.apples]
        
    def place_apple(self, color):
        x, y = self.board.get_random_empty_cell(self.snake.body,
                                                self.get_current_apple_coords())
        self.apples.append(Apple(x, y, color))
        
    def place_initial_apples(self):
        for _ in range(self.num_red_apples):
            self.place_apple(color="red")
        for _ in range(self.num_green_apples):
            self.place_apple(color="green")

    def step(self, action):
        if self.game_over:
            return True, self.score
        
        if action:
            self.snake.set_direction(action)
        
        
        self.snake.move()
        self.time_steps += 1
        self.max_duration_this_session = self.time_steps
        
        if (self.snake.check_collision_wall() or
            self.snake.check_collision_self() or
            self.snake.length <= 0):
            self.game_over = True
            return True, self.score

        if self.snake.head in self.get_current_apple_coords():
            apple = next(apple for apple in self.apples if apple.position == self.snake.head)
            if apple.color == "red":
                self.snake.shrink()
                self.score -= 1
            elif apple.color == "green":
                self.snake.grow()
                self.score += 1
            
            # Debug info for apple removal
            self.apples.remove(apple)
            self.place_apple(apple.color)  # Replace with same type of apple
            
        return False, self.score  # Return game_over=False and current score 