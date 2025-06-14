import pygame
import sys
import os
import argparse
import matplotlib.pyplot as plt
import time
from src.game_manager import GameManager
from src.constants import (COLOR_BLACK, COLOR_GREEN, COLOR_RED, CELL_SIZE,
                           SCREEN_WIDTH, SCREEN_HEIGHT, SPEED, PLAYER_COLORS,
                           BOARD_WIDTH, BOARD_HEIGHT)
from src.replay_manager import ReplayManager
from agent.snake_agent import SnakeAgent
from agent.state import State
from agent.reward_system import RewardSystem
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)


def draw_board(screen, game_manager, episode=None, max_score=None):
    screen.fill(COLOR_BLACK)

    for i, snake in enumerate(game_manager.snakes):
        if not game_manager.snake_alive[i]:
            continue

        if i < len(PLAYER_COLORS):
            color_data = PLAYER_COLORS[i]

        # Draw snake
        for x, y in snake.body:
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE,
                               CELL_SIZE)
            pygame.draw.rect(screen, color_data["body"], rect)
            if (x, y) == snake.head:
                pygame.draw.rect(screen, color_data["head"], rect)

    # Draw apples
    for apple in game_manager.apples:
        x, y = apple.position
        color = COLOR_GREEN if apple.color == "green" else COLOR_RED
        apple_rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE,
                                 CELL_SIZE)
        pygame.draw.rect(screen, color, apple_rect)

    # Update window title with score(s)
    if hasattr(game_manager, "scores"):
        score_str = " | ".join([
            f"Player {i+1}: {score}"
            for i, score in enumerate(game_manager.scores)
        ])
        title = f"Snake Game - {score_str}"
        if episode is not None:
            title += f" | Episode: {episode}"
        if max_score is not None:
            title += f" | Max Score: {max_score}"
        pygame.display.set_caption(title)

    pygame.display.flip()


def print_terminal_state(game_manager, snake_index=0):
    """Print the snake's vision state in terminal as required by subject"""

    snake = game_manager.snakes[snake_index]
    head_x, head_y = snake.head

    vision_map = {}

    # mark snake body, head, and apples
    for pos in snake.body:
        vision_map[pos] = 'S'
    vision_map[snake.head] = 'H'
    for apple in game_manager.apples:
        vision_map[apple.position] = 'G' if apple.color == 'green' else 'R'

    # Find what the snake can see in each direction
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visible_positions = set()
    visible_positions.add(snake.head)

    # Scan in this direction until we hit a wall
    for dx, dy in directions:
        x, y = head_x + dx, head_y + dy

        while True:
            if not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT):
                vision_map[(x, y)] = 'W'
                visible_positions.add((x, y))
                break

            visible_positions.add((x, y))

            x += dx
            y += dy

    # Determine the bounds for printing
    min_x = min(pos[0] for pos in visible_positions)
    max_x = max(pos[0] for pos in visible_positions)
    min_y = min(pos[1] for pos in visible_positions)
    max_y = max(pos[1] for pos in visible_positions)

    print("\n=== Snake Vision ===")
    for y in range(min_y, max_y + 1):
        line = ""
        for x in range(min_x, max_x + 1):
            pos = (x, y)
            if pos in visible_positions:
                if pos in vision_map:
                    line += vision_map[pos] + " "
                else:
                    line += "0 "
            else:
                line += "  "
        print(line.rstrip())


def train_agent(sessions=100,
                render=True,
                render_every=None,
                save_every=100000,
                model_path=None,
                speed=SPEED,
                num_players=1,
                use_smart_exploration=False):
    if render:
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        caption_mode = 'Player' if num_players == 1 else 'Players'
        pygame.display.set_caption(
            f"Snake Game - {num_players} {caption_mode}")
        clock = pygame.time.Clock()

    # Initialize agents
    agents = []
    for i in range(num_players):
        agent = SnakeAgent(use_smart_exploration=use_smart_exploration)
        agents.append(agent)
    if model_path and num_players == 1:
        agents[0].load_model(model_path)

    state_processor = State()
    reward_system = RewardSystem()
    replay_manager = ReplayManager()

    # Scores
    all_scores = [[] for _ in range(num_players)]
    max_scores = [3] * num_players
    epsilon_history = []

    start_time = time.time()
    epsilon = 1

    # EPISODE LOOP
    for episode in range(sessions):
        elapsed_time = time.time() - start_time

        game_manager = GameManager(num_players=num_players)
        game_manager.reset_game()

        steps = 0
        replay_manager.start_episode_recording()
        replay_manager.record_state(game_manager)

        current_states = [
            state_processor.get_state(game_manager, i)
            for i in range(num_players)
        ]
        if render_every is not None:
            render_this_episode = render and (episode % render_every == 0)
        else:
            render_this_episode = False

        # STEP LOOP
        while not game_manager.game_over:
            if render_this_episode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

            actions = []
            action_indices = []

            # 1 get actions from agents
            for i, agent in enumerate(agents):
                if i < num_players and (num_players == 1
                                        or game_manager.snake_alive[i]):
                    action = agent.get_action(current_states[i])
                    action_idx = agent.action_to_idx(action)
                    actions.append(action)
                    action_indices.append(action_idx)
                else:
                    # Dead snakes
                    actions.append(None)
                    action_indices.append(None)

            # 2 Make the move
            if num_players == 1:
                prev_scores = [game_manager.scores[0]]
                game_over, score = game_manager.step(actions[0])
                game_manager.game_over = game_over
                scores = [score]
            else:
                prev_scores = game_manager.scores.copy()
                game_over, scores, deaths = game_manager.step_multi_player(
                    actions)
                game_manager.game_over = game_over

            replay_manager.record_state(game_manager)

            # 3 Calculate rewards (snakes alive or that just died)
            for i in range(num_players):
                if num_players == 1 or game_manager.snake_alive[i] or (
                        deaths is not None and i < len(deaths)
                        and deaths[i] is not None):
                    if num_players == 1:
                        just_died = game_over
                    else:
                        just_died = deaths is not None and i < len(
                            deaths) and deaths[i] is not None

                    player_prev_score = prev_scores[i] if i < len(
                        prev_scores) else 0
                    player_new_score = scores[i] if i < len(scores) else 0

                    reward = reward_system.calculate_reward(
                        game_manager,
                        just_died,
                        player_prev_score,
                        player_new_score,
                        num_players=num_players,
                        player_index=i)

                next_state = state_processor.get_state(game_manager, i)

                # 4 Train the agent when agent dies
                if action_indices[i] is not None:
                    is_done = game_over if num_players == 1 else (
                        deaths is not None and i < len(deaths)
                        and deaths[i] is not None)
                    _, new_epsilon = agents[i].train(current_states[i],
                                                     action_indices[i], reward,
                                                     next_state, is_done)
                    if new_epsilon is not None:
                        epsilon = new_epsilon

                # 5 get next state
                current_states[i] = next_state

            if render_this_episode:
                draw_board(screen,
                           game_manager,
                           episode=episode,
                           max_score=max_scores[0])
                print_terminal_state(game_manager)
                clock.tick(speed)

            steps += 1

        # Track the best episode for replay
        if num_players == 1:
            game_manager.scores[0] = scores[0]
            replay_manager.end_episode(game_manager.scores[0])
        else:
            max_score = max(game_manager.scores) if game_manager.scores else 0
            replay_manager.end_episode(max_score)

        # Update max scores and collect scores for plotting
        for i in range(num_players):

            score = game_manager.scores[0] if num_players == 1 \
                                    else game_manager.scores[i]
            all_scores[i].append(score)
            if score > max_scores[i]:
                max_scores[i] = score

        epsilon_history.append(epsilon)

        print(f"{elapsed_time:.2f} - Episode {episode}/{sessions}")
        for i in range(num_players):
            print(f"Agent {i+1} - Score: {game_manager.scores[i]}, "
                  f"Max Score: {max_scores[i]}, epsilon: {epsilon:.2f}")

        # Save agent models periodically
        if (episode + 1) % save_every == 0:
            for i, agent in enumerate(agents):
                if num_players == 1:
                    agent.save_model(f"models/model_episode_{episode + 1}.pth")
                else:
                    agent.save_model(
                        f"models/agent{i+1}_episode_{episode + 1}.pth")
            print(f"Model{'s' if num_players > 1 else ''} "
                  f"saved at episode {episode + 1}")

    # Save final models
    if num_players == 1:
        agent.save_model(f"models/sess{episode + 1}-"
                         f"max{max_scores[0]}-"
                         f"sm{1 if use_smart_exploration else 0}.pth")
        plot_training_results(all_scores[0], epsilon_history=epsilon_history)

    end_time = time.time()
    training_duration = end_time - start_time
    print(f"Training duration: {training_duration:.2f} seconds")
    replay_manager.set_training_stats(all_scores[0], training_duration)

    # Show the best replay at the end of training
    replay_manager.play_best(speed)


def plot_training_results(scores, epsilon_history=None, window_size=100):
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(scores, label='Score per Episode', alpha=0.3, color='blue')

    if len(scores) >= window_size:
        moving_avg = [
            np.mean(scores[max(0, i - window_size):i + 1])
            for i in range(len(scores))
        ]
        plt.plot(moving_avg,
                 label=f'Moving Average (window={window_size})',
                 color='red',
                 linewidth=1.5)

    overall_mean = np.mean(scores)
    plt.axhline(y=overall_mean,
                color='purple',
                linestyle='--',
                label=f'Overall Mean: {overall_mean:.2f}')

    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.title('Snake Training Progress - Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if epsilon_history:
        plt.subplot(2, 1, 2)
        plt.plot(epsilon_history,
                 label='Epsilon (Exploration Rate)',
                 color='green',
                 linewidth=1.5)

        final_epsilon = epsilon_history[-1]
        plt.axhline(y=final_epsilon,
                    color='orange',
                    linestyle='--',
                    label=f'Final Epsilon: {final_epsilon:.3f}')

        plt.xlabel('Episodes')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Decay Over Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)  # Set y-axis limits for better visualization

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    print("Plot saved as 'training_results.png'")
    plt.show()


def play_game(model_path=None,
              num_players=1,
              speed=SPEED,
              step_by_step=False,
              games=100):
    if (speed < 1):
        speed = 1
    if (speed < 100):
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        caption_mode = 'Player' if num_players == 1 else 'Players'
        pygame.display.set_caption(
            f"Snake Game - {num_players} {caption_mode}")
        clock = pygame.time.Clock()

    # Initialize agents for all players
    agents = []
    for i in range(num_players):
        agent = SnakeAgent()
        if model_path:
            print(f"Loading model for player {i + 1} from {model_path}")
            agent.load_model(model_path)
        agent.epsilon = 0  # No exploration
        agents.append(agent)

    state_processor = State()
    max_score = 0

    for i in range(games):
        game_manager = GameManager(num_players=num_players)
        game_manager.reset_game()

        advance_step = not step_by_step
        while not game_manager.game_over:
            # Process events
            if (speed < 100):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif step_by_step and event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_s, pygame.K_SPACE):
                            advance_step = True
                        elif event.key == pygame.K_q:
                            print("Quitting game...")
                            pygame.quit()
                            sys.exit()

            if advance_step:
                if num_players == 1:
                    current_state = state_processor.get_state(game_manager)
                    action = agents[0].get_action(current_state)
                    game_over, score = game_manager.step(action)
                    game_manager.game_over = game_over
                    game_manager.scores[0] = score
                else:
                    actions = [None] * num_players
                    for i, agent in enumerate(agents):
                        if game_manager.snake_alive[i]:
                            s = state_processor.get_state(game_manager, i)
                            actions[i] = agent.get_action(s)

                    game_over, _, _ = game_manager.step_multi_player(actions)
                    game_manager.game_over = game_over

                if (speed < 100):
                    print(f"Action : {[direction_to_string(action)]}")
                    draw_board(screen, game_manager)
                    print_terminal_state(game_manager)

                # Reset the flag for step-by-step mode
                if step_by_step:
                    advance_step = False

            if step_by_step and speed < 100:
                font = pygame.font.SysFont('Arial', 20, bold=True)
                key_hint = "s" if num_players == 1 else "SPACE"
                hint_text = font.render(
                    f"Press '{key_hint}' for next step, 'q' to quit", True,
                    (255, 255, 255))
                screen.blit(hint_text, (10, SCREEN_HEIGHT - 30))
                pygame.display.flip()

            if (speed < 100):
                clock.tick(speed)

        if (game_manager.scores[0] > max_score):
            max_score = game_manager.scores[0]
        print(
            f"Game {i}/{games}| Score: {game_manager.scores[0]}, "
            f"Max Score: {max_score}"
        )


def direction_to_string(action):
    dx, dy = action
    if dx == -1 and dy == 0:
        return "LEFT"
    elif dx == 1 and dy == 0:
        return "RIGHT"
    elif dx == 0 and dy == -1:
        return "UP"
    elif dx == 0 and dy == 1:
        return "DOWN"
    else:
        return f"({dx}, {dy})"


def main():
    parser = argparse.ArgumentParser(
        description='Snake Game with Reinforcement Learning')

    parser.add_argument(
        '-m',
        '--mode',
        type=str,
        default='train',
        choices=['train', 'play'],
        help='Mode to run: train (1-4 players) or play (1-4 players)')
    parser.add_argument('-s',
                        '--sessions',
                        type=int,
                        default=1000,
                        help='Number of training sessions (default: 1000)')
    parser.add_argument('-ds',
                        '--display-speed',
                        type=int,
                        default=SPEED,
                        help='Speed of the game (default: 24)')
    parser.add_argument('-r',
                        '--render',
                        type=int,
                        default=None,
                        help='Render every N sessions (default: 100)')
    parser.add_argument(
        '-se',
        '--save-every',
        type=int,
        default=100000,
        help='Save model every N sessions (default: episode / 10)')
    parser.add_argument('-mp',
                        '--model-path',
                        type=str,
                        default=None,
                        help='Path to the model file (single player mode)')
    parser.add_argument(
        '-np',
        '--num-players',
        type=int,
        default=1,
        help='Number of players (1-4) for training or play mode (default: 1)')
    parser.add_argument(
        '-st',
        '--step-by-step',
        action='store_true',
        help='Enable step-by-step mode in play mode (press space to advance)')
    parser.add_argument('-sm',
                        '--smart-exploration',
                        action='store_true',
                        help='Enable smart exploration during training')
    args = parser.parse_args()

    if args.num_players < 1 or args.num_players > 4:
        print("Error: Number of players must be between 1 and 4")
        sys.exit(1)

    if args.model_path:
        if not os.path.isfile(
                args.model_path) or not args.model_path.endswith('.pth'):
            print(f"Error: Model file '{args.model_path}' "
                  f"does not exist or is not a .pth file.")
            sys.exit(1)

    if args.mode == 'train':
        train_agent(
            sessions=args.sessions,
            render=args.render,
            render_every=args.render,
            save_every=args.save_every,
            model_path=args.model_path if args.num_players == 1 else None,
            speed=args.display_speed,
            num_players=args.num_players,
            use_smart_exploration=args.smart_exploration)
    elif args.mode == 'play':
        play_game(model_path=args.model_path,
                  num_players=args.num_players,
                  speed=args.display_speed,
                  step_by_step=args.step_by_step,
                  games=args.sessions)

    pygame.quit()
    print("Game exited successfully.")
    sys.exit()


if __name__ == "__main__":
    main()
