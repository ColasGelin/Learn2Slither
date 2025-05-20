import pygame
import sys
import os
import argparse
import matplotlib.pyplot as plt


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from src.game_manager import GameManager
from src.constants import (BOARD_WIDTH, BOARD_HEIGHT, CellType, VISION_CHAR_MAP, 
                      COLOR_BLACK, COLOR_HEAD, COLOR_GREEN, COLOR_RED, COLOR_BLUE,
                      UP, DOWN, LEFT, RIGHT, AGENT_STATE_SIZE, AGENT_ACTION_SIZE)
from src.replay_manager import ReplayManager
from agent.snake_agent import SnakeAgent
from agent.state import State
from agent.reward_system import RewardSystem
from agent.curriculum_learning import CurriculumLearning
import numpy as np

# Pygame configuration
CELL_SIZE = 40  
SCREEN_WIDTH = BOARD_WIDTH * CELL_SIZE
SCREEN_HEIGHT = BOARD_HEIGHT * CELL_SIZE
SPEED = 24

def draw_board(screen, game_manager, episode=None, max_score=None):
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
    
    # Add text display for episode and score information
    if episode is not None:
        # Initialize font
        font = pygame.font.SysFont('Arial', 20, bold=True)
        
        # Render episode text
        episode_text = font.render(f"Episode: {episode}", True, (255, 255, 255))
        screen.blit(episode_text, (10, 10))
        
        # Render score text
        score_text = font.render(f"Score: {game_manager.score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 40))
        
        # Render max score if available
        if max_score is not None:
            max_score_text = font.render(f"Max Score: {max_score}", True, (255, 255, 255))
            screen.blit(max_score_text, (10, 70))
    
    pygame.display.flip()

def train_agent(sessions=100, render=True, render_every=100, save_every=100000, model_path=None, speed=SPEED, use_curriculum=False):
    if render:
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Snake Game")
        clock = pygame.time.Clock()

    agent = SnakeAgent()
    curriculum_manager = CurriculumLearning() if use_curriculum else None
    stateProcessor = State()
    reward_system = RewardSystem()
    replay_manager = ReplayManager()
    scores = []
    max_score = 3
    
    for episode in range(sessions):
        
        if curriculum_manager:
            config = curriculum_manager.get_current_config()
            game_manager = GameManager(
                num_green_apples=config['green_apples'],
                num_red_apples=config['red_apples']
            )
            reward_params = curriculum_manager.get_modified_reward_system_params()
            reward_system = RewardSystem(**reward_params)
        else:
            game_manager = GameManager()
        
        game_manager.reset_game()
        steps = 0  # Initialize step counter
        replay_manager.start_episode()
        replay_manager.record_state(game_manager)
        
        current_state = stateProcessor.get_state(game_manager)
        
        render = render and (episode % render_every == 0)
    
        while not game_manager.game_over:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
            
            action = agent.get_action(current_state)
            action_idx = agent.action_to_idx(action)
            
            prev_score = game_manager.score
            game_over, score = game_manager.step(action)
            game_manager.game_over = game_over
            
            replay_manager.record_state(game_manager)
            
            # Use reward system instead of inline logic
            reward = reward_system.calculate_reward(
                game_manager, game_over, prev_score, score)
                
            next_state = stateProcessor.get_state(game_manager)
            agent.train(current_state, action_idx, reward, next_state, game_over)
            
            current_state = next_state
            game_manager.score = score
            
            if render:
                draw_board(screen, game_manager)
                clock.tick(speed)
          
            steps += 1

        replay_manager.end_episode(game_manager.score)

        if game_manager.score > max_score:
            max_score = game_manager.score
            
        scores.append(game_manager.score)
            
        print(f"Episode {episode}/{sessions} - Score: {game_manager.score}, "
                f"Max Score: {max_score}, Stage: {curriculum_manager.get_current_config()['name']}")
    
        if (episode + 1) % save_every == 0:
            agent.save_model(f"models/model_episode_{episode + 1}.pth")
            print(f"Model saved at episode {episode + 1}")
        
        if curriculum_manager:
            stage_advanced = curriculum_manager.update_stage(
                episode_score=game_manager.score,
                episode_duration=steps
            )
            
            if stage_advanced:
                # Update reward system with new parameters
                reward_params = curriculum_manager.get_modified_reward_system_params()
                reward_system = RewardSystem(**reward_params)

    agent.save_model(f"models/final_model_{max_score}.pth")
    print("Training completed. Final model saved.")
    
    plot_training_results(scores, mean_scores=np.mean(scores))
    
    replay_manager.play_best(speed)
    
def plot_training_results(scores, mean_scores, window_size=100):
    plt.figure(figsize=(12, 6))
    
    plt.plot(scores, label='Score per Episode', alpha=0.3, color='blue')
    
    plt.plot(mean_scores, label='Mean Score', color='green', linewidth=2)
    
    if len(scores) >= window_size:
        moving_avg = [np.mean(scores[max(0, i-window_size):i+1]) for i in range(len(scores))]
        plt.plot(moving_avg, label=f'Moving Average (window={window_size})', color='red', linewidth=1.5)
    
    overall_mean = np.mean(scores)
    plt.axhline(y=overall_mean, color='purple', linestyle='--', 
                label=f'Overall Mean: {overall_mean:.2f}')
    
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.title('Snake Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('training_results.png')
    print(f"Plot saved as 'training_results.png'")
    
    plt.show()

def play_game(model_path=None, speed=SPEED, step_by_step=False):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake Game")
    clock = pygame.time.Clock()
    
    while True:
        game_manager = GameManager()
        game_manager.reset_game()
        
        if model_path:
            agent = SnakeAgent()
            agent.load_model(model_path)
            agent.epsilon = 0 
            stateProcessor = State()
            print(f"AI playing with model: {model_path}")

            advance_step = not step_by_step

            while not game_manager.game_over:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    # Check for keypress in step-by-step mode
                    elif step_by_step and event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_s:
                            advance_step = True
                        elif event.key == pygame.K_q:
                            print("Quitting game...")
                            pygame.quit()
                            sys.exit()

                # Only advance the game if not in step-by-step mode or if 's' was pressed
                if advance_step:
                    current_state = stateProcessor.get_state(game_manager)
                    action = agent.get_action(current_state)
                    
                    game_over, score = game_manager.step(action)
                    game_manager.game_over = game_over
                    game_manager.score = score
                    
                    # Reset the flag for step-by-step mode
                    if step_by_step:
                        advance_step = False

                draw_board(screen, game_manager)
                if step_by_step:
                    font = pygame.font.SysFont('Arial', 20, bold=True)
                    hint_text = font.render("Press 's' for next step, 'q' to quit", True, (255, 255, 255))
                    screen.blit(hint_text, (10, SCREEN_HEIGHT - 30))
                    pygame.display.flip()
                
                clock.tick(speed)
            
            print(f"Game Over! Final Score: {game_manager.score}")

def curriculum_learning():
    # This function will be empty for now
    # Will be implemented to provide progressive learning difficulty
    print("Curriculum learning activated")
    pass

def main():
    
    parser = argparse.ArgumentParser(description='Snake Game with Reinforcement Learning')
    
    parser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'play'],
                        help='Mode to run: train or play (default: train)')
    parser.add_argument('-s', '--sessions', type=int, default=1000,
                        help='Number of training sessions (default: 1000)')
    parser.add_argument('-r', '--render', action='store_true',
                        help='Enable rendering during training')
    parser.add_argument('-ds', '--display-speed', type=int, default=SPEED,
                        help='Speed of the game (default: 24)')
    parser.add_argument('-re', '--render-every', type=int, default=100,
                        help='Render every N sessions (default: 10)')
    parser.add_argument('-se', '--save-every', type=int, default=100000,
                        help='Save model every N sessions (default: episode / 10)')
    parser.add_argument('-mp', '--model-path', type=str, default=None,
                        help='Path to the model file')
    parser.add_argument('-st', '--step-by-step', action='store_true',
                        help='Enable step-by-step mode in play mode (press s to advance)')
    parser.add_argument('-c', '--curriculum', action='store_true',
                        help='Enable curriculum learning')

    if parser.parse_args().mode == 'train':
        train_agent(
            sessions=parser.parse_args().sessions,
            render=parser.parse_args().render,
            render_every=parser.parse_args().render_every,
            save_every=parser.parse_args().save_every,
            model_path=parser.parse_args().model_path,
            speed=parser.parse_args().display_speed ,
            use_curriculum=parser.parse_args().curriculum
        )
    else:
        play_game(
            model_path=parser.parse_args().model_path,
            speed=parser.parse_args().display_speed,
            step_by_step=parser.parse_args().step_by_step
        )

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()