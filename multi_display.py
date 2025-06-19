import pygame
import sys
import os
import numpy as np
from src.game_manager import GameManager
from src.constants import (COLOR_BLACK, COLOR_GREEN, COLOR_RED, CELL_SIZE,
                           SCREEN_WIDTH, SCREEN_HEIGHT, SPEED, PLAYER_COLORS,
                           BOARD_WIDTH, BOARD_HEIGHT)
from agent.snake_agent import SnakeAgent
from agent.state import State

# Grid configuration
GRID_SIZE = 3  # 3x3 grid
GAMES_TOTAL = 9
MINI_CELL_SIZE = CELL_SIZE // 3  # Smaller cells for mini games
MINI_BOARD_WIDTH = SCREEN_WIDTH // GRID_SIZE
MINI_BOARD_HEIGHT = SCREEN_HEIGHT // GRID_SIZE

class MultiGameDisplay:
    def __init__(self, model_paths=None, speed=SPEED):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Snake Training Showcase - 9 Sessions")
        self.clock = pygame.time.Clock()
        self.speed = speed
        
        # Initialize 9 game managers and agents
        self.game_managers = []
        self.agents = []
        self.state_processor = State()
        self.game_finished = [False] * GAMES_TOTAL
        
        # Create progressively better agents
        for i in range(GAMES_TOTAL):
            game_manager = GameManager(num_players=1)
            agent = SnakeAgent()
            
            # Load models with progressively better performance
            if model_paths and len(model_paths) > i:
                try:
                    agent.load_model(model_paths[i])
                    print(f"Loaded model {i+1}: {model_paths[i]}")
                except:
                    print(f"Could not load model {i+1}, using random agent")
            
            agent.epsilon = max(0.1 - (i * 0.01), 0.0)  # Less exploration for later agents
            
            self.game_managers.append(game_manager)
            self.agents.append(agent)
        
        # Reset all games
        for gm in self.game_managers:
            gm.reset_game()
    
    def draw_mini_game(self, screen, game_manager, grid_x, grid_y, game_index):
        """Draw a single mini game in the grid"""
        # Calculate offset for this mini game
        offset_x = grid_x * MINI_BOARD_WIDTH
        offset_y = grid_y * MINI_BOARD_HEIGHT
        
        # Create a surface for this mini game
        mini_surface = pygame.Surface((MINI_BOARD_WIDTH, MINI_BOARD_HEIGHT))
        mini_surface.fill(COLOR_BLACK)
        
        # Draw border
        border_color = (100, 100, 100) if not self.game_finished[game_index] else (255, 0, 0)
        pygame.draw.rect(mini_surface, border_color, 
                        (0, 0, MINI_BOARD_WIDTH, MINI_BOARD_HEIGHT), 2)
        
        # Scale factor for mini display
        scale_x = (MINI_BOARD_WIDTH - 20) / (BOARD_WIDTH * CELL_SIZE)
        scale_y = (MINI_BOARD_HEIGHT - 60) / (BOARD_HEIGHT * CELL_SIZE)
        scale = min(scale_x, scale_y)
        
        start_x = 10
        start_y = 40
        
        # Draw snake
        for i, snake in enumerate(game_manager.snakes):
            if not game_manager.snake_alive[i]:
                continue
                
            color_data = PLAYER_COLORS[i] if i < len(PLAYER_COLORS) else PLAYER_COLORS[0]
            
            for x, y in snake.body:
                rect_x = start_x + int(x * CELL_SIZE * scale)
                rect_y = start_y + int(y * CELL_SIZE * scale)
                rect_size = max(2, int(CELL_SIZE * scale))
                
                rect = pygame.Rect(rect_x, rect_y, rect_size, rect_size)
                
                if (x, y) == snake.head:
                    pygame.draw.rect(mini_surface, color_data["head"], rect)
                else:
                    pygame.draw.rect(mini_surface, color_data["body"], rect)
        
        # Draw apples
        for apple in game_manager.apples:
            x, y = apple.position
            color = COLOR_GREEN if apple.color == "green" else COLOR_RED
            
            rect_x = start_x + int(x * CELL_SIZE * scale)
            rect_y = start_y + int(y * CELL_SIZE * scale)
            rect_size = max(2, int(CELL_SIZE * scale))
            
            apple_rect = pygame.Rect(rect_x, rect_y, rect_size, rect_size)
            pygame.draw.rect(mini_surface, color, apple_rect)
        
        # Draw game info
        font = pygame.font.Font(None, 24)
        game_text = font.render(f"Game {game_index + 1}", True, (255, 255, 255))
        score_text = font.render(f"Score: {game_manager.scores[0]}", True, (255, 255, 255))
        
        mini_surface.blit(game_text, (5, 5))
        mini_surface.blit(score_text, (5, 20))
        
        # Draw "FINISHED" if game is over
        if self.game_finished[game_index]:
            finished_text = font.render("FINISHED", True, (255, 0, 0))
            text_rect = finished_text.get_rect(center=(MINI_BOARD_WIDTH//2, MINI_BOARD_HEIGHT//2))
            mini_surface.blit(finished_text, text_rect)
        
        # Blit mini surface to main screen
        screen.blit(mini_surface, (offset_x, offset_y))
    
    def run_showcase(self):
        """Run the showcase with all 9 games"""
        running = True
        step_counter = 0
        
        print("Starting Snake Training Showcase...")
        print("9 agents with progressively better training will play simultaneously")
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        # Reset all games
                        self.game_finished = [False] * GAMES_TOTAL
                        for gm in self.game_managers:
                            gm.reset_game()
                        step_counter = 0
                        print("All games reset!")
            
            # Clear screen
            self.screen.fill(COLOR_BLACK)
            
            # Update and draw each game
            active_games = 0
            for i in range(GAMES_TOTAL):
                if not self.game_finished[i]:
                    game_manager = self.game_managers[i]
                    agent = self.agents[i]
                    
                    if not game_manager.game_over:
                        # Get action from agent
                        current_state = self.state_processor.get_state(game_manager)
                        action = agent.get_action(current_state)
                        
                        # Step the game
                        game_over, score = game_manager.step(action)
                        game_manager.game_over = game_over
                        game_manager.scores[0] = score
                        
                        if game_over:
                            self.game_finished[i] = True
                            print(f"Game {i+1} finished with score: {score}")
                    else:
                        self.game_finished[i] = True
                        active_games += 1
                
                # Calculate grid position
                grid_x = i % GRID_SIZE
                grid_y = i // GRID_SIZE
                
                # Draw the mini game
                self.draw_mini_game(self.screen, self.game_managers[i], grid_x, grid_y, i)
            
            # Draw controls
            font_small = pygame.font.Font(None, 24)
            controls_text = font_small.render("Press 'R' to reset all games, 'Q' to quit", 
                                            True, (200, 200, 200))
            self.screen.blit(controls_text, (10, SCREEN_HEIGHT - 20))
            
            pygame.display.flip()
            self.clock.tick(self.speed)
            
            step_counter += 1
            
            # Check if all games are finished
            if all(self.game_finished):
                print("\nAll games finished!")
                scores = [gm.scores[0] for gm in self.game_managers]
                for i, score in enumerate(scores):
                    print(f"Game {i+1}: {score} points")
                print(f"Average score: {np.mean(scores):.2f}")
                print("Press 'R' to restart or 'Q' to quit")
        
        pygame.quit()

def create_progressive_agents(base_model_path=None):
    """Create 9 model paths with progressive improvement"""
    if base_model_path and os.path.exists(base_model_path):
        # If we have a base model, use it for all (they'll have different epsilon values)
        return [base_model_path] * GAMES_TOTAL
    
    # Look for existing models in models directory
    models_dir = "models"
    model_paths = []
    
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        model_files.sort()  # Sort to get progression
        
        # Take up to 9 models, or repeat the best one
        for i in range(GAMES_TOTAL):
            if i < len(model_files):
                model_paths.append(os.path.join(models_dir, model_files[i]))
            elif model_files:
                # Use the last (presumably best) model for remaining slots
                model_paths.append(os.path.join(models_dir, model_files[-1]))
            else:
                model_paths.append(None)
    else:
        print("No models directory found, using random agents")
        model_paths = [None] * GAMES_TOTAL
    
    return model_paths

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Snake Training Showcase - 9 Sessions Display')
    parser.add_argument('-mp', '--model-path', type=str, default=None,
                       help='Base model path (will be used for all games)')
    parser.add_argument('-s', '--speed', type=int, default=10,
                       help='Game speed (default: 10)')
    
    args = parser.parse_args()
    
    # Create progressive model paths
    model_paths = create_progressive_agents(args.model_path)
    
    # Run the showcase
    showcase = MultiGameDisplay(model_paths=model_paths, speed=args.speed)
    showcase.run_showcase()

if __name__ == "__main__":
    main()