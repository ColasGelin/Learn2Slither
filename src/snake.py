from src.constants import INITIAL_SNAKE_LENGTH, RIGHT, BOARD_WIDTH, BOARD_HEIGHT
import random


class Snake:
    def __init__(self, board_width=BOARD_WIDTH, board_height=BOARD_HEIGHT):
        self.length = INITIAL_SNAKE_LENGTH
        self.body = []
        self.direction = RIGHT
        start_x = random.randint(0, board_width - INITIAL_SNAKE_LENGTH)
        start_y = random.randint(0, board_height - 1)
        self.body = [(start_x + x, start_y) for x in range(INITIAL_SNAKE_LENGTH)]
        self.head = self.body[0]
        self.board_width = board_width
        self.board_height = board_height

    def move(self):
        if not self.body:
            raise ValueError("Snake has no body to move")
        
        head_x, head_y = self.body[0]
        new_head_x = head_x + self.direction[0]
        new_head_y = head_y + self.direction[1]
        
        self.body.insert(0, (new_head_x, new_head_y))
        self.head = self.body[0]  # Update the head property
        
        if len(self.body) > self.length:
            self.body.pop()

    def grow(self, amount=1):
        self.length += amount

    def shrink(self, amount=1):
        if self.length > 0:
            self.length -= amount

    def set_direction(self, new_direction):
        if self.direction[0] + new_direction[0] == 0 and \
           self.direction[1] + new_direction[1] == 0:
            return
        self.direction = new_direction

    def check_collision_wall(self):
        head_x, head_y = self.body[0]
        return not (0 <= head_x < self.board_width and 0 <= head_y < self.board_height)

    def check_collision_self(self):
        head = self.body[0]
        return head in self.body[1:]