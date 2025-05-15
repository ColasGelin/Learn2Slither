import pygame
import sys
import pickle
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from game_manager import GameManager
from constants import (BOARD_WIDTH, BOARD_HEIGHT, CellType, VISION_CHAR_MAP, 
                      COLOR_BLACK, COLOR_HEAD, COLOR_GREEN, COLOR_RED, COLOR_BLUE,
                      UP, DOWN, LEFT, RIGHT)
from agent.snake_agent import SnakeAgent
from agent.state import State
import numpy as np

# Pygame configuration
CELL_SIZE = 40  
SCREEN_WIDTH = BOARD_WIDTH * CELL_SIZE
SCREEN_HEIGHT = BOARD_HEIGHT * CELL_SIZE
SPEED = 24

def draw_board(screen, game_manager):
    screen.fill(COLOR_BLACK)
    
    for x, y in game_manager.snake.body:
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, COLOR_BLUE, rect)
        if (x, y) == game_manager.snake.head:
            pygame.draw.rect(screen, COLOR_HEAD, rect)
    
    for apple in game_manager.apples:
        x, y = apple.position
        color = COLOR_GREEN if apple.color == "green" else COLOR_RED
        apple_rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, color, apple_rect)
    
    pygame.display.flip()

def train_agent(episodes=100, render=True, render_every=10, save_every=50, load_model=False):
    if render:
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Snake Game")
        clock = pygame.time.Clock()
        
    agent = SnakeAgent()
    stateProcessor = State()
    scores = []
    max_score = 0
    
    for episode in range(episodes):
        game_manager = GameManager()
        game_manager.reset_game()
        steps = 0  # Initialize step counter
        
        current_state = stateProcessor.get_state(game_manager)
        
        should_render = render and (episode % render_every == 0)
    
        while not game_manager.game_over:
            if should_render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
            
            action = agent.get_action(current_state)
            action_idx = agent.action_to_idx(action)
            
            prev_score = game_manager.score
            game_over, score = game_manager.step(action)
            
            # Update game manager's game_over state with the result from step()
            game_manager.game_over = game_over
            
            if game_over:
                reward = -10
            elif score > prev_score:
                reward = 10
            elif score < prev_score:
                reward = -5
            else:
                reward = -0.01      
                
            next_state = stateProcessor.get_state(game_manager)
            agent.train(current_state, action_idx, reward, next_state, game_over)
            
            current_state = next_state
            game_manager.score = score
            
            if should_render:
                draw_board(screen, game_manager)
                clock.tick(SPEED)
          
            steps += 1
            # Print step info only occasionally to reduce console spam
            if steps % 10 == 0:
                print(f"Episode {episode + 1}/{episodes} - Step: {steps}, "
                      f"Action: {action}, Score: {game_manager.score}")
            
        scores.append(game_manager.score)
        avg_score = np.mean(scores[-100:]) if len(scores) > 0 else 0

        if game_manager.score > max_score:
            max_score = game_manager.score
            
        print(f"Episode {episode + 1}/{episodes} - Score: {game_manager.score}, "
                f"Avg Score: {avg_score:.2f}, Max Score: {max_score}")
        
        if (episode + 1) % save_every == 0:
            agent.save_model(f"model_episode_{episode + 1}.pth")
            print(f"Model saved at episode {episode + 1}")
            
    agent.save_model("final_model.pth")
    print("Training completed. Final model saved.")

def main():
    training_mode = True
    
    if training_mode:
        train_agent()
        
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()