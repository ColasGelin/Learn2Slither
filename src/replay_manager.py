import pickle
import os
import pygame
import numpy as np
from .constants import COLOR_BLACK, COLOR_GREEN, COLOR_RED, COLOR_BLUE, COLOR_HEAD, SCREEN_WIDTH, SCREEN_HEIGHT, CELL_SIZE

class ReplayManager:
    def __init__(self):
        self.best_replay = []
        self.best_score = 0
        self.current_replay = []
        self.episodes = []
        self.best_combined_score = 0
        self.best_combined_replay = []
        
        # Define player colors for rendering
        self.player_colors = [
            {"body": COLOR_BLUE, "head": COLOR_HEAD},         
            {"body": (255, 0, 255), "head": (150, 0, 150)},   
            {"body": (255, 165, 0), "head": (200, 130, 0)},  
            {"body": (0, 255, 255), "head": (0, 200, 200)},   
        ]
        
        # Apple colors
        self.apple_colors = {
            "green": COLOR_GREEN,
            "red": COLOR_RED
        }
    
    def record_state(self, game_manager):
        """Record current game state for any number of players"""
        state = {}
        
        # Check for multi-player mode
        if hasattr(game_manager, 'multiple_snake') and game_manager.multiple_snake:
            state['multi_player'] = True
            state['players'] = []
            
            # Track alive status for each snake in multiplayer
            state['snake_alive'] = game_manager.snake_alive.copy() if hasattr(game_manager, 'snake_alive') else [True] * len(game_manager.snakes)
            
            # Add all players (not just hardcoded ones)
            for i, snake in enumerate(game_manager.snakes):
                state['players'].append({
                    'body': list(snake.body),
                    'score': game_manager.scores[i]
                })
            
            # Calculate combined score
            state['combined_score'] = sum(player['score'] for player in state['players'])
        else:
            # Single player mode
            state['multi_player'] = False
            state['players'] = [{
                'body': list(game_manager.snake.body),
                'score': game_manager.score
            }]
            state['snake_alive'] = [True]  # Single player is always alive until game over
        
        # Record apples (common for all modes)
        state['apples'] = [(a.position, a.color) for a in game_manager.apples]
        
        self.current_replay.append(state)
    
    def end_episode(self, score):
        """Check if this is the best replay"""
        is_best = False
        
        # Check for best individual score
        if score > self.best_score:
            self.best_score = score
            self.best_replay = self.current_replay.copy()
            self._save_replay()
            is_best = True
        
        # Check for best mean score in multi-player mode
        if self.current_replay and self.current_replay[0].get('multi_player', False):
            final_state = self.current_replay[-1]
            player_scores = [player['score'] for player in final_state['players']]
            if player_scores and all(score > 0 for score in player_scores):
                harmonic_mean = len(player_scores) / sum(1 / score for score in player_scores)
                if harmonic_mean > self.best_combined_score:
                    self.best_combined_score = harmonic_mean
                    self.best_combined_replay = self.current_replay.copy()
                    self._save_replay("replays/best_combined_replay.pkl")
                    is_best = True
        
        # Save episode
        if is_best:
            self.episodes.append(self.current_replay)
            
        # Reset current replay
        self.current_replay = []
        
        return is_best
    
    def start_episode(self):
        """Start a new episode recording"""
        self.current_replay = []
    
    def _save_replay(self, filename="replays/best_replay.pkl"):
        """Save best replay to file"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(self.best_replay if "combined" not in filename else self.best_combined_replay, f)
    
    def play_best(self, speed=24):
        """Play the best recorded replay based on mode"""
        if hasattr(self, 'best_combined_replay') and self.best_combined_replay:
            # For multi-player mode, show the best combined score replay
            if self.best_combined_replay and self.best_combined_replay[0].get('multi_player', False):
                return self._play_replay(self.best_combined_replay, self.best_combined_score, speed, is_combined=True)
        
        # Fall back to regular best replay
        if not self.best_replay:
            print("No replay available")
            return
        
        return self._play_replay(self.best_replay, self.best_score, speed)
    
    def _play_replay(self, replay, score, speed=24, is_combined=False):
        """Helper method to play a specific replay"""
        if not replay:
            print("No replay available")
            return

        pygame.init()
        
        # Check if this is a multi-player replay
        is_multi_player = replay[0].get('multi_player', False) if replay else False
        
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        title = f"Best {'Combined ' if is_combined else ''}Replay"
        pygame.display.set_caption(title)
        clock = pygame.time.Clock()
        font = pygame.font.SysFont('Arial', 20, bold=True)

        running = True
        while running:
            for frame, state in enumerate(replay):
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

                # Draw all snakes
                player_count = len(state['players'])
                for i, player in enumerate(state['players']):
                    # Skip drawing dead snakes
                    if 'snake_alive' in state and i < len(state['snake_alive']) and not state['snake_alive'][i]:
                        continue
                        
                    # Get colors for this player (loop through available colors)
                    color_data = self.player_colors[i % len(self.player_colors)]
                    
                    # Draw each segment of the snake
                    for pos in player['body']:
                        rect = pygame.Rect(pos[0] * CELL_SIZE, pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                        pygame.draw.rect(screen, color_data["body"], rect)
                        if pos == player['body'][0]:  # Head
                            pygame.draw.rect(screen, color_data["head"], rect)
                
                # Draw apples
                for pos, color in state['apples']:
                    apple_color = self.apple_colors.get(color, COLOR_GREEN)
                    apple_rect = pygame.Rect(pos[0] * CELL_SIZE, pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(screen, apple_color, apple_rect)

                pygame.display.flip()
                clock.tick(speed)

            # Show final screen for a moment before looping
            pygame.time.wait(2000)
            
        pygame.quit()

    def get_latest_episode(self):
        """Return the states from the most recently completed episode"""
        if not self.episodes:
            return None
        return self.episodes[-1]