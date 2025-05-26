from src.constants import BOARD_WIDTH, BOARD_HEIGHT, CellType
import random


class Board:

    def __init__(self, width=BOARD_WIDTH, height=BOARD_HEIGHT):
        self.grid = [[CellType.EMPTY for _ in range(width)]
                     for _ in range(height)]
        self.width = width
        self.height = height

    def get_random_empty_cell(self, snake_body_coords, apple_coords_list):
        empty_cells = [
            (x, y) for x in range(self.width) for y in range(self.height)
            if self.get_cell_type(x, y) == CellType.EMPTY and
            (x, y) not in snake_body_coords and (x, y) not in apple_coords_list
        ]

        if empty_cells:
            return random.choice(empty_cells)
        raise ValueError("No empty cells available")

    def get_cell_type(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return CellType.WALL
