import pickle
import os
import pygame
import numpy as np
from .constants import COLOR_BLACK, COLOR_GREEN, COLOR_RED, COLOR_BLUE, COLOR_HEAD

CELL_SIZE = 40

class ReplayManager:
    def __init__(self):
        self.best_replay = []
        self.best_score = 0
        self.current_replay = []
    
    def record_state(self, game_manager):
        """Record current game state"""
        state = {
            'snake_body': list(game_manager.snake.body),
            'apples': [(a.position, a.color) for a in game_manager.apples],
            'score': game_manager.score
        }
        self.current_replay.append(state)
    
    def end_episode(self, score):
        """Check if this is the best replay"""
        if score > self.best_score:
            self.best_score = score
            self.best_replay = self.current_replay.copy()
            self._save_replay()
            return True
        self.current_replay = []
        return False
    
    def start_episode(self):
        """Start a new episode recording"""
        self.current_replay = []
    
    def _save_replay(self, filename="replays/best_replay.pkl"):
        """Save best replay to file"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(self.best_replay, f)
    
    def play_best(self, speed=24):
        """Play the best recorded replay in a loop until user presses 'q' or closes the window"""
        if not self.best_replay:
            print("No replay available")
            return

        pygame.init()
        screen_width = max([max(x for x, _ in state['snake_body']) for state in self.best_replay]) * CELL_SIZE + CELL_SIZE*2
        screen_height = max([max(y for _, y in state['snake_body']) for state in self.best_replay]) * CELL_SIZE + CELL_SIZE*2

        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(f"Best Replay - Score: {self.best_score}")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont('Arial', 20, bold=True)

        running = True
        while running:
            for frame, state in enumerate(self.best_replay):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        running = False
                        break
                if not running:
                    break

                screen.fill(COLOR_BLACK)

                # Draw snake
                for pos in state['snake_body']:
                    rect = pygame.Rect(pos[0] * CELL_SIZE, pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(screen, COLOR_BLUE, rect)
                    if pos == state['snake_body'][0]:  # Head
                        pygame.draw.rect(screen, COLOR_HEAD, rect)

                # Draw apples
                for pos, color in state['apples']:
                    apple_color = COLOR_GREEN if color == "green" else COLOR_RED
                    apple_rect = pygame.Rect(pos[0] * CELL_SIZE, pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(screen, apple_color, apple_rect)

                # Show score
                score_text = font.render(f"Score: {state['score']}", True, (255, 255, 255))
                screen.blit(score_text, (10, 10))

                pygame.display.flip()
                clock.tick(speed)

            # Show final screen for a moment
            pygame.time.wait(2000)
        pygame.quit()