from constants import INITIAL_SNAKE_LENGTH, RIGHT, BOARD_WIDTH, BOARD_HEIGHT


class Snake:
    def __init__(self):
        self.length = INITIAL_SNAKE_LENGTH
        self.body = []
        self.direction = RIGHT

    def move(self):
        if not self.body:
            raise ValueError("Snake has no body to move")
        
        head_x, head_y = self.body[0]
        new_head_x = head_x + self.direction[0]
        new_head_y = head_y + self.direction[1]
        
        self.body.insert(0, (new_head_x, new_head_y))
        
        if len(self.body) > self.length:
            self.body.pop()
        pass

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
        return not (0 <= head_x < BOARD_WIDTH and 0 <= head_y < BOARD_HEIGHT)
    
    def check_collision_self(self):
        head = self.body[0]
        return head in self.body[1:]