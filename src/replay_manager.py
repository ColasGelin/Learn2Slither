import pygame
from .constants import (COLOR_BLACK, COLOR_GREEN, COLOR_RED,
                        SCREEN_WIDTH, SCREEN_HEIGHT, CELL_SIZE,
                        PLAYER_COLORS)
from .stats_overlay import StatsOverlay


class ReplayManager:

    def __init__(self):
        self.best_replay = []
        self.best_score = 0
        self.current_replay = []
        self.best_combined_score = 0
        self.best_combined_replay = []
        self.apple_colors = {"green": COLOR_GREEN, "red": COLOR_RED}
        
        # Stats tracking
        self.training_scores = []
        self.training_duration = 0
        self.stats_overlay = None
        self.show_stats = False


    def record_state(self, game_manager):
        """Record current game state for any number of players"""
        state = {}

        multiplayer = hasattr(game_manager, 'snakes') and len(
            game_manager.snakes) > 1

        if multiplayer:
            state['multi_player'] = True
            state['players'] = []

            state['snake_alive'] = game_manager.snake_alive.copy() if hasattr(
                game_manager,
                'snake_alive') else [True] * len(game_manager.snakes)

            for i, snake in enumerate(game_manager.snakes):
                state['players'].append({
                    'body': list(snake.body),
                    'score': game_manager.scores[i]
                })

            state['combined_score'] = sum(player['score']
                                          for player in state['players'])
        else:
            state['multi_player'] = False
            state['players'] = [{
                'body': list(game_manager.snake.body),
                'score': game_manager.score
            }]
            state['snake_alive'] = [
                True
            ]

        state['apples'] = [(a.position, a.color) for a in game_manager.apples]
        self.current_replay.append(state)

    def end_episode(self, score):
        """Check if this is the best replay"""
        is_best = False

        # If multiplayer get harmonic mean to find most interesting replay
        if (self.current_replay and
           self.current_replay[0].get('multi_player', False)):
            final_state = self.current_replay[-1]
            player_scores = [
                player['score'] for player in final_state['players']
            ]
            if player_scores and all(score > 0 for score in player_scores):
                harmonic_mean = len(player_scores) / sum(
                    1 / score for score in player_scores)
                if harmonic_mean > self.best_score:
                    self.best_score = harmonic_mean
                    self.best_replay = self.current_replay.copy()
                    is_best = True
        else:
            if score > self.best_score:
                self.best_score = score
                self.best_replay = self.current_replay.copy()
                is_best = True

        self.current_replay = []

        return is_best

    def start_episode_recording(self):
        """Start a new episode recording"""
        self.current_replay = []

    def play_best(self, speed=24):
        """Play the best recorded replay based on mode"""
        if not self.best_replay:
            print("No replay available")
            return

        # Determine if this is a multiplayer replay
        is_multiplayer = (self.best_replay and
                          self.best_replay[0].get('multi_player', False))

        return self._play_replay(self.best_replay, speed,
                                 is_combined=is_multiplayer,
                                 score=self.best_score)

    def set_training_stats(self, scores, training_duration=0):
        """Set training statistics for display"""
        self.training_scores = scores
        self.training_duration = training_duration
        self.stats_overlay = StatsOverlay(scores, training_duration, len(scores))
    
    def _play_replay(self, replay, speed=24, is_combined=False, score=0):
        """Helper method to play a specific replay"""
        if not replay:
            print("No replay available")
            return

        pygame.init()

        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        title = f"Best {'Combined ' if is_combined else ''}Replay - {score}"
        pygame.display.set_caption(title)
        clock = pygame.time.Clock()

        running = True
        while running:
            for frame, state in enumerate(replay):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    if (event.type == pygame.KEYDOWN and
                            event.key == pygame.K_q):
                        running = False
                        break
                    if (event.type == pygame.KEYDOWN and
                            event.key == pygame.K_h):
                        self.show_stats = not self.show_stats
                        break
                if not running:
                    break

                screen.fill(COLOR_BLACK)

                # Draw all snakes but dead snakes
                for i, player in enumerate(state['players']):
                    if 'snake_alive' in state and i < len(
                            state['snake_alive']
                    ) and not state['snake_alive'][i]:
                        continue

                    color_data = PLAYER_COLORS[i % len(PLAYER_COLORS)]

                    for pos in player['body']:
                        rect = pygame.Rect(pos[0] * CELL_SIZE,
                                           pos[1] * CELL_SIZE, CELL_SIZE,
                                           CELL_SIZE)
                        pygame.draw.rect(screen, color_data["body"], rect)
                        if pos == player['body'][0]:  # Head
                            pygame.draw.rect(screen, color_data["head"], rect)

                # Draw apples
                for pos, color in state['apples']:
                    apple_color = self.apple_colors.get(color, COLOR_GREEN)
                    apple_rect = pygame.Rect(pos[0] * CELL_SIZE,
                                             pos[1] * CELL_SIZE, CELL_SIZE,
                                             CELL_SIZE)
                    pygame.draw.rect(screen, apple_color, apple_rect)

                # Draw stats overlay if enabled
                if self.show_stats and self.stats_overlay:
                    self.stats_overlay.draw(screen)

                pygame.display.flip()
                clock.tick(speed)

            # Show final screen for a moment before looping
            pygame.time.wait(2000)

        pygame.quit()
