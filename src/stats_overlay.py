import pygame
import numpy as np


class StatsOverlay:
    def __init__(self, scores, training_duration=0, episodes=0):
        """
        Initialize the statistics overlay
        
        Args:
            scores: List of scores from training episodes
            training_duration: Total training time in seconds
            episodes: Total number of episodes
        """
        self.scores = scores
        self.training_duration = training_duration
        self.episodes = episodes
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        self.stats_calculated = False
        self.stats = {}
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLACK = (0, 0, 0)
        self.BLUE = (100, 150, 255)
        
    def _init_fonts(self):
        """Initialize fonts if not already done"""
        if self.font_large is None:
            pygame.font.init()
            self.font_large = pygame.font.SysFont('Arial', 32, bold=True)
            self.font_medium = pygame.font.SysFont('Arial', 22, bold=True)
            self.font_small = pygame.font.SysFont('Arial', 18)
    
    def _calculate_stats(self):
        """Calculate all statistics from the scores"""
        if self.stats_calculated or not self.scores:
            return
            
        scores_array = np.array(self.scores)
        
        self.stats = {
            'total_episodes': len(self.scores),
            'max_score': int(np.max(scores_array)),
            'min_score': int(np.min(scores_array)),
            'mean_score': float(np.mean(scores_array)),
            'median_score': float(np.median(scores_array)),
            'training_duration': self.training_duration,
            'avg_episode_duration': self.training_duration / len(self.scores) if len(self.scores) > 0 else 0,
            'scores_above_10': int(np.sum(scores_array >= 10)),
            'scores_above_20': int(np.sum(scores_array >= 20)),
            'scores_above_30': int(np.sum(scores_array >= 30)),
            'scores_above_40': int(np.sum(scores_array >= 40)),
            'scores_above_50': int(np.sum(scores_array >= 50)),
            'improvement_rate': self._calculate_improvement_rate(scores_array),
            'last_100_avg': float(np.mean(scores_array[-100:])) if len(scores_array) >= 100 else float(np.mean(scores_array))
        }
        
        self.stats_calculated = True
    
    def _calculate_improvement_rate(self, scores_array):
        if len(scores_array) < 4:
            return 0.0
            
        quarter_size = len(scores_array) // 4
        first_quarter_avg = np.mean(scores_array[:quarter_size])
        last_quarter_avg = np.mean(scores_array[-quarter_size:])
        
        if first_quarter_avg == 0:
            return 0.0
            
        return ((last_quarter_avg - first_quarter_avg) / first_quarter_avg) * 100
    
    def _format_time(self, seconds):
        """Format seconds into readable time format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def draw(self, screen):
        """Draw the statistics directly on the screen (no overlay)"""
        self._init_fonts()
        self._calculate_stats()
        
        # Clear screen with black background
        screen.fill(self.BLACK)
        
        # Get screen dimensions
        screen_width, screen_height = screen.get_size()
        
        y_offset = 80
        line_height = 25
        left_margin = 50
        
        # Title
        title_text = self.font_large.render("TRAINING STATISTICS - PLAYER 1", True, self.GREEN)
        title_rect = title_text.get_rect(centerx=screen_width // 2)
        screen.blit(title_text, (title_rect.x, y_offset))
        y_offset += 120
        
        # Create two columns for better layout
        col1_x = left_margin
        col2_x = screen_width // 2 + 50
        col1_y = y_offset
        col2_y = y_offset
        
        # Score Statistics
        section_text = self.font_medium.render("SCORE STATISTICS", True, self.BLUE)
        screen.blit(section_text, (col1_x, col1_y))
        col1_y += 35
        
        score_stats = [
            f"Max Score: {self.stats['max_score']}",
            f"Min Score: {self.stats['min_score']}",
            f"Average Score: {self.stats['mean_score']:.2f}",
            f"Median Score: {self.stats['median_score']:.2f}",
        ]
        
        for stat in score_stats:
            stat_text = self.font_small.render(stat, True, self.WHITE)
            screen.blit(stat_text, (col1_x + 20, col1_y))
            col1_y += line_height
        
        col1_y += 30
        
        # Performance
        section_text = self.font_medium.render("PERFORMANCE", True, self.BLUE)
        screen.blit(section_text, (col1_x, col1_y))
        col1_y += 35
        
        performance_stats = [
            f"Total Episodes: {self.stats['total_episodes']}",
            f"Last 100 Episodes Avg: {self.stats['last_100_avg']:.2f}",
            f"Improvement Rate: {self.stats['improvement_rate']:.1f}%",
        ]
        
        for stat in performance_stats:
            stat_text = self.font_small.render(stat, True, self.WHITE)
            screen.blit(stat_text, (col1_x + 20, col1_y))
            col1_y += line_height
        
        #Achievements
        section_text = self.font_medium.render("ACHIEVEMENTS", True, self.BLUE)
        screen.blit(section_text, (col2_x, col2_y))
        col2_y += 35
        
        achievement_stats = [
            f"Scores >= 10: {self.stats['scores_above_10']} ({self.stats['scores_above_10']/self.stats['total_episodes']*100:.1f}%)",
            f"Scores >= 20: {self.stats['scores_above_20']} ({self.stats['scores_above_20']/self.stats['total_episodes']*100:.1f}%)",
            f"Scores >= 30: {self.stats['scores_above_30']} ({self.stats['scores_above_30']/self.stats['total_episodes']*100:.1f}%)",
            f"Scores >= 40: {self.stats['scores_above_40']} ({self.stats['scores_above_40']/self.stats['total_episodes']*100:.1f}%)",
            f"Scores >= 50: {self.stats['scores_above_50']} ({self.stats['scores_above_50']/self.stats['total_episodes']*100:.1f}%)",
        ]
        
        for stat in achievement_stats:
            stat_text = self.font_small.render(stat, True, self.WHITE)
            screen.blit(stat_text, (col2_x + 20, col2_y))
            col2_y += line_height
        
        col2_y += 30
        
        # Training TIme 
        section_text = self.font_medium.render("TRAINING TIME", True, self.BLUE)
        screen.blit(section_text, (col2_x, col2_y))
        col2_y += 35
        
        time_stats = [
            f"Total Training Time: {self._format_time(self.stats['training_duration'])}",
            f"Episodes per Hour: {3600/self.stats['avg_episode_duration']:.0f}" if self.stats['avg_episode_duration'] > 0 else "Episodes per Hour: N/A",
        ]
        
        for stat in time_stats:
            stat_text = self.font_small.render(stat, True, self.WHITE)
            screen.blit(stat_text, (col2_x + 20, col2_y))
            col2_y += line_height
            
        instruction_y = screen_height - 60
        instruction_text = self.font_small.render("Press 'H' to return to replay | Press 'Q' to quit", True, self.RED)
        instruction_rect = instruction_text.get_rect(centerx=screen_width // 2)
        screen.blit(instruction_text, (instruction_rect.x, instruction_y))
        
    def update_stats(self, scores, training_duration=0, episodes=0):
        """Update statistics with new data"""
        self.scores = scores
        self.training_duration = training_duration
        self.episodes = episodes
        self.stats_calculated = False