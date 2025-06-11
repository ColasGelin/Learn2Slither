from enum import Enum

AGENT_STATE_SIZE = 24
AGENT_ACTION_SIZE = 4

BOARD_WIDTH = 10
BOARD_HEIGHT = 10

CELL_SIZE = 70
SCREEN_WIDTH = BOARD_WIDTH * CELL_SIZE
SCREEN_HEIGHT = BOARD_HEIGHT * CELL_SIZE
SPEED = 24

INITIAL_SNAKE_LENGTH = 3

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

NUM_GREEN_APPLES = 2
NUM_RED_APPLES = 1


class CellType(Enum):
    EMPTY = 0
    WALL = 1


COLOR_BLACK = (0, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)

PLAYER_COLORS = [
    {
        "body": (0, 0, 255),
        "head": (0, 0, 150)
    },
    {
        "body": (255, 0, 255),
        "head": (150, 0, 150)
    },
    {
        "body": (255, 165, 0),
        "head": (200, 130, 0)
    },
    {
        "body": (0, 255, 255),
        "head": (0, 200, 200)
    },
]
