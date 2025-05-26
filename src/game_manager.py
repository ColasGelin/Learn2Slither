from src.constants import BOARD_WIDTH, BOARD_HEIGHT
from src.snake import Snake
from src.board import Board
from src.apple import Apple


class GameManager:

    def __init__(self,
                 board_width=BOARD_WIDTH,
                 board_height=BOARD_HEIGHT,
                 num_green_apples=1,
                 num_red_apples=0,
                 num_players=1):
        self.board = Board(board_width, board_height)
        self.num_green_apples = num_green_apples
        self.num_red_apples = num_red_apples
        self.num_players = num_players
        self.multiple_snake = num_players > 1  # For backward compatibility
        self.reset_game()

    def reset_game(self):
        self.board = Board(self.board.width, self.board.height)

        # Initialize players
        self.snakes = []
        self.scores = []
        occupied_positions = []

        # Create snakes for each player
        for i in range(self.num_players):
            new_snake = Snake(self.board.width, self.board.height)

            # Ensure snakes don't overlap
            attempts = 0
            while attempts < 10 and any(pos in occupied_positions
                                        for pos in new_snake.body):
                new_snake = Snake(self.board.width, self.board.height)
                attempts += 1

            self.snakes.append(new_snake)
            self.scores.append(3)  # Initial score
            occupied_positions.extend(new_snake.body)

        # For backward compatibility
        if self.num_players > 0:
            self.snake = self.snakes[0]
            self.score = self.scores[0]

        if self.num_players > 1:
            self.snake_1 = self.snakes[0]
            self.snake_2 = self.snakes[1]
            self.score_1 = self.scores[0]
            self.score_2 = self.scores[1]

        # Game state
        self.apples = []
        self.place_initial_apples()
        self.game_over = False
        self.time_steps = 0
        self.snake_alive = [True] * self.num_players

    def get_current_apple_coords(self):
        return [apple.position for apple in self.apples]

    def place_apple(self, color):
        # Get all occupied positions from all snakes
        occupied = []
        for snake in self.snakes:
            occupied.extend(snake.body)

        # Add current apple positions
        occupied.extend(self.get_current_apple_coords())

        # Get empty position for new apple
        x, y = self.board.get_random_empty_cell(occupied, [])
        self.apples.append(Apple(x, y, color))

    def place_initial_apples(self):
        for _ in range(self.num_red_apples):
            self.place_apple(color="red")
        for _ in range(self.num_green_apples):
            self.place_apple(color="green")

    def step(self, action):
        """
        Single player step function
        (preserved for backward compatibility)
        """
        if self.num_players > 1:
            return self.step_multi_player([action])

        if self.game_over:
            return True, self.score

        if action:
            self.snake.set_direction(action)

        self.snake.move()
        self.time_steps += 1

        # Check collisions
        if self.snake.check_collision_wall():
            self.game_over = True
        elif self.snake.check_collision_self():
            self.game_over = True
        elif self.snake.length <= 0:
            self.game_over = True

        # Process apple collisions
        if not self.game_over and self.handle_apple_collision(0):
            pass  # Apple collision was handled in the method

        return self.game_over, self.score

    def handle_apple_collision(self, snake_index):
        """Process apple collision for a specific snake"""
        if not self.snake_alive[snake_index]:
            return False

        snake = self.snakes[snake_index]

        if snake.head in self.get_current_apple_coords():
            apple = next(apple for apple in self.apples
                         if apple.position == snake.head)

            if apple.color == "red":
                snake.shrink()
                self.scores[snake_index] -= 1
            elif apple.color == "green":
                snake.grow()
                self.scores[snake_index] += 1

            # Update the main score variables for backward compatibility
            if self.num_players > 0 and snake_index == 0:
                self.score = self.scores[0]
            if self.num_players > 1:
                self.score_1 = self.scores[0]
                self.score_2 = self.scores[1]

            # Remove and replace the apple
            self.apples.remove(apple)
            self.place_apple(apple.color)
            return True

        return False

    def check_snake_collisions(self, snake_index):
        """Check various collision types for a specific snake"""
        if not self.snake_alive[snake_index]:
            return None

        snake = self.snakes[snake_index]

        # Check wall and self collisions
        if snake.check_collision_wall():
            self.snake_alive[snake_index] = False
            return 0
        elif snake.check_collision_self():
            self.snake_alive[snake_index] = False
            return 1
        elif snake.length <= 0:
            self.snake_alive[snake_index] = False
            return 2

        # Check collisions with other snakes
        for other_index, other_snake in enumerate(self.snakes):
            if other_index == snake_index or not self.snake_alive[other_index]:
                continue

            # If head hit another snake's body
            if snake.head in other_snake.body:
                # Check if it's a head-to-head collision
                if snake.head == other_snake.head:
                    # Longer snake wins; equal length means both lose
                    if snake.length > other_snake.length:
                        self.snake_alive[other_index] = False
                    elif snake.length < other_snake.length:
                        self.snake_alive[snake_index] = False
                        return 1
                    else:
                        # Equal length: both lose
                        self.snake_alive[snake_index] = False
                        self.snake_alive[other_index] = False
                        return 1
                else:
                    # Normal body collision
                    self.snake_alive[snake_index] = False
                    return 1

        return None  # No collision

    def step_multi_player(self, actions):
        """Generalized step function for multiple players"""
        if self.game_over:
            return True, tuple(self.scores), tuple([None] * self.num_players)

        # Apply actions to each snake
        for i, action in enumerate(actions):
            if i < self.num_players and action and self.snake_alive[i]:
                self.snakes[i].set_direction(action)

        # Move all snakes
        for i, snake in enumerate(self.snakes):
            if self.snake_alive[i]:
                snake.move()

        self.time_steps += 1

        # Check for collisions
        death_reasons = [None] * self.num_players
        for i in range(self.num_players):
            death_reasons[i] = self.check_snake_collisions(i)

        # Process apple collisions for all snakes
        for i in range(self.num_players):
            self.handle_apple_collision(i)

        # Check if there is only one snake left alive or if all snakes are dead
        alive_count = sum(self.snake_alive)
        if alive_count <= 1:  # Game ends when 0 or 1 snake remains
            self.game_over = True

        # For backward compatibility with dual snake mode
        if self.num_players > 1:
            return self.game_over, tuple(self.scores), tuple(death_reasons)

        # For standard single-player mode
        return self.game_over, self.scores[0], death_reasons[0]

    def step_two_snakes(self, action_1, action_2):
        """
        Legacy method for two-snake game mode,
        now redirects to the generalized method
        """
        return self.step_multi_player([action_1, action_2])
