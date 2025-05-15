from src.constants import BOARD_WIDTH, BOARD_HEIGHT, CellType
from src.snake import Snake
from src.board import Board
from src.apple import Apple


class GameManager:
    def __init__(self):
        self.board = Board()
        self.reset_game()
        
    def reset_game(self):
        self.board = Board()
        self.snake = Snake()
        self.apples = []
        self.place_initial_apples()
        self.score = 0
        self.game_over = False
        self.time_steps = 0
        
    def get_current_apple_coords(self):
        return [apple.position for apple in self.apples]
        
    def place_apple(self, color):
        x, y = self.board.get_random_empty_cell(self.snake.body,
                                                self.get_current_apple_coords())
        self.apples.append(Apple(x, y, color))
        
    def place_initial_apples(self):
        for _ in range(1):  # NUM_RED_APPLES = 1
            self.place_apple(color="red")
        for _ in range(2):  # NUM_GREEN_APPLES = 2
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

        # Debug info for apple detection
        print(f"Snake head: {self.snake.head}, Apples: {self.get_current_apple_coords()}")
        
        if self.snake.head in self.get_current_apple_coords():
            print(f"COLLISION DETECTED! Snake head: {self.snake.head}")
            apple = next(apple for apple in self.apples if apple.position == self.snake.head)
            if apple.color == "red":
                self.snake.shrink()
                self.score -= 1
                print(f"Ate red apple! Score: {self.score}, Length: {self.snake.length}")
            elif apple.color == "green":
                self.snake.grow()
                self.score += 1
                print(f"Ate green apple! Score: {self.score}, Length: {self.snake.length}")
            
            # Debug info for apple removal
            print(f"Removing apple at {apple.position}, color: {apple.color}")
            self.apples.remove(apple)
            self.place_apple(apple.color)  # Replace with same type of apple
            print(f"New apples list: {self.get_current_apple_coords()}")
            
        return False, self.score  # Return game_over=False and current score 