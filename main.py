import pygame
import sys
import os
import argparse
import matplotlib.pyplot as plt
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from src.game_manager import GameManager
from src.constants import ( 
                      COLOR_BLACK, COLOR_HEAD, COLOR_GREEN, COLOR_RED, COLOR_BLUE,
                      UP, DOWN, LEFT, RIGHT, AGENT_STATE_SIZE, AGENT_ACTION_SIZE, 
                      CELL_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT, SPEED)
from src.replay_manager import ReplayManager
from agent.snake_agent import SnakeAgent
from agent.state import State
from agent.reward_system import RewardSystem
import numpy as np

# Define player colors for consistent visualization
PLAYER_COLORS = [
    {"body": COLOR_BLUE, "head": COLOR_HEAD},           # Player 1 - Blue
    {"body": (255, 0, 255), "head": (150, 0, 150)},     # Player 2 - Purple
    {"body": (255, 165, 0), "head": (200, 130, 0)},     # Player 3 - Orange (if needed)
    {"body": (0, 255, 255), "head": (0, 200, 200)},     # Player 4 - Cyan (if needed)
]

def draw_board(screen, game_manager, episode=None, max_score=None):
    """Draw the game board for a single player game"""
    screen.fill(COLOR_BLACK)
    
    # Draw snake
    for x, y in game_manager.snake.body:
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, COLOR_BLUE, rect)
        if (x, y) == game_manager.snake.head:
            pygame.draw.rect(screen, COLOR_HEAD, rect)
    
    # Draw apples
    for apple in game_manager.apples:
        x, y = apple.position
        color = COLOR_GREEN if apple.color == "green" else COLOR_RED
        apple_rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, color, apple_rect)
    
    # Add text display for episode and score information
    if episode is not None:
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

def draw_multi_player_board(screen, game_manager, episode=None, max_scores=None):
    """Draw the game board for a multi-player game with n players"""
    screen.fill(COLOR_BLACK)
    
    # Draw all snakes - only draw snakes that are alive
    for i, snake in enumerate(game_manager.snakes):
        # Skip drawing if the snake is dead
        if not game_manager.snake_alive[i]:
            continue
            
        if i < len(PLAYER_COLORS):
            color_data = PLAYER_COLORS[i]
        else:
            # Fallback color if we have more players than defined colors
            color_data = {"body": (200, 200, 200), "head": (150, 150, 150)}
            
        # Draw snake body
        for x, y in snake.body:
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, color_data["body"], rect)
            # Highlight the head
            if (x, y) == snake.head:
                pygame.draw.rect(screen, color_data["head"], rect)
    
    # Draw apples
    for apple in game_manager.apples:
        x, y = apple.position
        color = COLOR_GREEN if apple.color == "green" else COLOR_RED
        apple_rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, color, apple_rect)
    
    pygame.display.flip()

def train_agent(sessions=100, render=True, render_every=100, save_every=100000, 
               model_path=None, speed=SPEED, num_players=1, use_smart_exploration=False):
    """Unified training function that handles any number of players"""
    if render:
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(f"Snake Game - {num_players} {'Player' if num_players == 1 else 'Players'}")
        clock = pygame.time.Clock()

    # Initialize agents - for multiplayer we always create fresh agents
    # For single player, we can optionally load a model
    agents = []
    for i in range(num_players):
        agent = SnakeAgent(use_smart_exploration=use_smart_exploration)
        # For single player only, we can load a pre-existing model
        if i == 0 and num_players == 1 and model_path:
            agent.load_model(model_path)
            print(f"Loaded model from: {model_path}")
        agents.append(agent)
    
    state_processor = State()
    reward_system = RewardSystem()
    replay_manager = ReplayManager()
    
    # Track scores and stats for each agent
    all_scores = [[] for _ in range(num_players)]
    max_scores = [3] * num_players
    death_counters = [[0, 0, 0] for _ in range(num_players)]
    
    start_time = time.time()
    
    for episode in range(sessions):
        elapsed_time = time.time() - start_time
        
        # Create a game with the specified number of players
        game_manager = GameManager(num_players=num_players)
        game_manager.reset_game()
        
        steps = 0
        replay_manager.start_episode()
        replay_manager.record_state(game_manager)
        
        # Get initial states for all agents
        current_states = [state_processor.get_state(game_manager, i) for i in range(num_players)]
        
        render_this_episode = render and (episode % render_every == 0)
        
        while not game_manager.game_over:
            if render_this_episode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
            
            # Get actions from all agents
            actions = []
            action_indices = []
            
            for i, agent in enumerate(agents):
                if i < num_players and (num_players == 1 or game_manager.snake_alive[i]):
                    action = agent.get_action(current_states[i])
                    action_idx = agent.action_to_idx(action)
                    actions.append(action)
                    action_indices.append(action_idx)
                else:
                    actions.append(None)
                    action_indices.append(None)
            
            # Store previous scores
            if num_players == 1:
                prev_score = game_manager.score
            else:
                prev_scores = game_manager.scores.copy()
            
            # Execute actions based on number of players
            if num_players == 1:
                # Single player mode
                game_over, score = game_manager.step(actions[0])
                game_manager.game_over = game_over
                scores = [score]
                
            else:
                # Multi-player mode
                game_over, scores, deaths = game_manager.step_multi_player(actions)
                game_manager.game_over = game_over
                
                # Update death counters
                for i, death in enumerate(deaths):
                    if death is not None:
                        death_counters[i][death] += 1
            
            replay_manager.record_state(game_manager)
            
            # Calculate rewards and update agents
            for i in range(num_players):
                # Only process active snakes or those that just died
                if num_players == 1 or game_manager.snake_alive[i] or (deaths is not None and i < len(deaths) and deaths[i] is not None):
                    # Calculate reward
                    if num_players == 1:
                        reward = reward_system.calculate_reward(
                            game_manager, game_over, prev_score, scores[i])
                    else:
                        # Make sure we have a valid death status for this player
                        is_dead = deaths is not None and i < len(deaths) and deaths[i] is not None
                        
                        # Make sure we have valid score values - handle potential IndexError
                        player_prev_score = prev_scores[i] if i < len(prev_scores) else 0
                        player_new_score = scores[i] if i < len(scores) else 0
                        
                        reward = reward_system.calculate_reward(
                            game_manager, is_dead, 
                            player_prev_score, player_new_score, 
                            num_players=num_players, player_index=i)
                    
                    # Get next state
                    next_state = state_processor.get_state(game_manager, i)
                    
                    # Train the agent if it was active
                    if action_indices[i] is not None:
                        # For done status, use appropriate game over condition
                        is_done = game_over if num_players == 1 else (deaths is not None and i < len(deaths) and deaths[i] is not None)
                        agents[i].train(
                            current_states[i], 
                            action_indices[i], 
                            reward, 
                            next_state, 
                            is_done
                        )
                    
                    # Update for next iteration
                    current_states[i] = next_state
            
            if render_this_episode:
                if num_players == 1:
                    draw_board(screen, game_manager, episode=episode, max_score=max_scores[0])
                else:
                    draw_multi_player_board(screen, game_manager, episode=episode, max_scores=max_scores)
                clock.tick(speed)
          
            steps += 1
        
        # Update game scores for tracking
        if num_players == 1:
            game_manager.score = scores[0]
        
        # Track the best episode for replay
        if num_players == 1:
            replay_manager.end_episode(game_manager.score)
        else:
            max_score = max(game_manager.scores) if game_manager.scores else 0
            replay_manager.end_episode(max_score)

        # Update max scores and collect scores for plotting
        for i in range(num_players):
            score = game_manager.score if num_players == 1 else game_manager.scores[i]
            all_scores[i].append(score)
            if score > max_scores[i]:
                max_scores[i] = score
        
        # Print progress information
        if num_players == 1:
            print(f"{elapsed_time:.2f} - Episode {episode}/{sessions} - Score: {game_manager.score}, "
                  f"Max Score: {max_scores[0]}")
        else:
            print(f"{elapsed_time:.2f} - Episode {episode}/{sessions}")
            for i in range(num_players):
                print(f"Agent {i+1} - Score: {game_manager.scores[i]}, Max Score: {max_scores[i]}")

        # Save agent models periodically
        if (episode + 1) % save_every == 0:
            for i, agent in enumerate(agents):
                if num_players == 1:
                    agent.save_model(f"models/model_episode_{episode + 1}.pth")
                else:
                    agent.save_model(f"models/agent{i+1}_episode_{episode + 1}.pth")
            print(f"Model{'s' if num_players > 1 else ''} saved at episode {episode + 1}")
    
    # Save final models
    if num_players == 1:
        agent.save_model(f"models/final_model_{max_scores[0]}.pth")
    
    # Plot results
    if num_players == 1:
        plot_training_results(all_scores[0], np.mean(all_scores[0]))
    
    # Show the best replay at the end of training
    replay_manager.play_best(speed)

def plot_training_results(scores, mean_scores, window_size=100):
    plt.figure(figsize=(12, 8))
    
    # Plot scores in the main subplot
    plt.subplot(2, 1, 1)
    plt.plot(scores, label='Score per Episode', alpha=0.3, color='blue')
    
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
    
    plt.tight_layout()
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

def play_multi_player_game(model_paths=[], num_players=2, speed=SPEED, step_by_step=False):
    """Play a game with multiple AI players - no human players"""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"Snake Game - {num_players} Players")
    clock = pygame.time.Clock()
    
    # Ensure we have enough model paths
    while len(model_paths) < num_players:
        model_paths.append(None)  # Use default agent if no model specified
    
    # Initialize agents - all AI
    agents = []
    for i in range(num_players):
        agent = SnakeAgent()
        if model_paths[i]:
            agent.load_model(model_paths[i])
        agent.epsilon = 0  # No exploration, just use the model
        agents.append(agent)
        print(f"Agent {i+1} playing with model: {model_paths[i] if model_paths[i] else 'default model'}")
    
    state_processor = State()
    
    while True:
        game_manager = GameManager(num_players=num_players)
        game_manager.reset_game()
        
        advance_step = not step_by_step

        while not game_manager.game_over:
            actions = [None] * num_players
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif step_by_step and event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        advance_step = True
                    elif event.key == pygame.K_q:
                        print("Quitting game...")
                        pygame.quit()
                        sys.exit()

            # Only advance the game if not in step-by-step mode or if space was pressed
            if advance_step:
                # Get actions for all AI players
                for i, agent in enumerate(agents):
                    if game_manager.snake_alive[i]:
                        current_state = state_processor.get_state(game_manager, i)
                        actions[i] = agent.get_action(current_state)
                
                # Execute all actions
                game_over, scores, _ = game_manager.step_multi_player(actions)
                game_manager.game_over = game_over
                
                # Reset the flag for step-by-step mode
                if step_by_step:
                    advance_step = False

            # Draw the game board without score information
            draw_multi_player_board(screen, game_manager)
            
            if step_by_step:
                font = pygame.font.SysFont('Arial', 20, bold=True)
                hint_text = font.render("Press SPACE for next step, Q to quit", True, (255, 255, 255))
                screen.blit(hint_text, (10, SCREEN_HEIGHT - 30))
                pygame.display.flip()
            
            clock.tick(speed)
        
def main():
    parser = argparse.ArgumentParser(description='Snake Game with Reinforcement Learning')
    
    parser.add_argument('-m', '--mode', type=str, default='train', 
                        choices=['train', 'play'],
                        help='Mode to run: train (1-4 players) or play (1-4 players)')
    parser.add_argument('-s', '--sessions', type=int, default=1000,
                        help='Number of training sessions (default: 1000)')
    parser.add_argument('-r', '--render', action='store_true',
                        help='Enable rendering during training')
    parser.add_argument('-ds', '--display-speed', type=int, default=SPEED,
                        help='Speed of the game (default: 24)')
    parser.add_argument('-re', '--render-every', type=int, default=100,
                        help='Render every N sessions (default: 100)')
    parser.add_argument('-se', '--save-every', type=int, default=100000,
                        help='Save model every N sessions (default: episode / 10)')
    parser.add_argument('-mp', '--model-path', type=str, default=None,
                        help='Path to the model file (single player mode)')
    parser.add_argument('-np', '--num-players', type=int, default=1,
                        help='Number of players (1-4) for training or play mode (default: 1)')
    parser.add_argument('-st', '--step-by-step', action='store_true',
                        help='Enable step-by-step mode in play mode (press space to advance)')
    parser.add_argument('-sm', '--smart-exploration', action='store_true',
                        help='Enable smart exploration during training')
    args = parser.parse_args()
    
    # Validate number of players
    if args.num_players < 1 or args.num_players > 4:
        print("Error: Number of players must be between 1 and 4")
        sys.exit(1)

    if args.mode == 'train':
        # Single unified train function for both single and multiplayer
        train_agent(
            sessions=args.sessions,
            render=args.render,
            render_every=args.render_every,
            save_every=args.save_every,
            model_path=args.model_path if args.num_players == 1 else None,
            speed=args.display_speed,
            num_players=args.num_players,
            use_smart_exploration=args.smart_exploration
        )
    elif args.mode == 'play':
        # Play mode with any number of players
        if args.num_players == 1:
            # Single player mode
            play_game(
                model_path=args.model_path,
                speed=args.display_speed,
                step_by_step=args.step_by_step
            )

    pygame.quit()
    print("Game exited successfully.")
    sys.exit()
    
if __name__ == "__main__":
    main()