import random
from constants import BOARD_WIDTH, BOARD_HEIGHT, CellType
from snake import Snake
from board import Board
from apple import Apple


class GameManager:
    def __init__(self):
        self.board = Board()
        self.reset_game()
        
    def reset_game(self):
        self.board = self.board.create_board(BOARD_WIDTH, BOARD_HEIGHT)
        self.snake = Snake()
        self.apples = [Apple()]
        self.place_initial_apples()
        self.score = 0
        self.game_over = False
        self.time_steps = 0
        
    def get_current_apple_coords(self):
        return [apple.position for apple in self.apples]
        
    def place_apple(self):
        x, y = self.board.get_random_empty_cell(self.snake.body,
                                                self.get_current_apple_coords)
        self.apples.append(Apple(x, y))
        
    def place_initial_apples(self):
        for _ in range(2):
            self.place_apple(color="red")
        self.place_apple(color="green")

    def step(self):
        self.snake.move()
        if self.snake.head in self.get_current_apple_coords():
            apple = next(apple for apple in self.apples if apple.position == self.snake.head)
            if apple.color == "red":
                self.snake.shrink()
                self.score -= 1
            elif apple.color == "green":
                self.snake.grow()
                self.score += 1
            self.apples.remove(apple)
            self.place_apple()
        self.time_steps += 1