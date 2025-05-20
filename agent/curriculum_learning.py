import numpy as np
from src.constants import BOARD_WIDTH, BOARD_HEIGHT

class CurriculumLearning:
    def __init__(self, stage_progress_threshold=10):
        self.stage_progress_threshold = stage_progress_threshold
    
        # Current curriculum stage (0-indexed)
        self.current_stage = 0
        
        # Tracking variables for progression
        self.success_count = 0
        self.total_episodes = 0
        self.best_score = 0
        self.best_duration = 0
        
        # Stage-specific configurations
        self.stage_configs = self._create_stage_configs()
        
    def _create_stage_configs(self):
        return [
            {
                'name': 'Beginner',
                'green_apples': 1,
                'red_apples': 0,
                'target_length': 8,
                'reward_multiplier': 1.0,
                'base_move_penalty': -0.5,
                'collision_penalty': -500
            },
            {
                'name': 'Intermediate',
                'green_apples': 2,
                'red_apples': 0,
                'target_length': 13,
                'reward_multiplier': 1,
                'base_move_penalty': -0.5,
                'collision_penalty': -100
            },
            {
                'name': 'Advanced',
                'green_apples': 2,
                'red_apples': 1,
                'target_length': 18,
                'reward_multiplier': 1,
                'base_move_penalty': -0.5,
                'collision_penalty': -100
            }
            ]
    
    def get_current_config(self):
        return self.stage_configs[self.current_stage]
    
    def update_stage(self, episode_score, episode_duration):
        """
        Update curriculum stage based on agent performance.
        
        Args:
            episode_score: Score achieved in the episode
            episode_duration: Duration of the episode
            
        Returns:
            bool: Whether the stage was advanced
        """
        self.total_episodes += 1
        self.best_score = max(self.best_score, episode_score)
        self.best_duration = max(self.best_duration, episode_duration)
        
        # Check if we've met the criteria to advance to the next stage
        current_config = self.get_current_config()
        if episode_score >= current_config['target_length']:
            self.success_count += 1
            
            # If we've had enough successful episodes, advance to the next stage
            if self.success_count >= self.stage_progress_threshold and self.current_stage < len(self.stage_configs) - 1:
                self.current_stage += 1
                self.success_count = 0
                print(f"🎓 Advancing to curriculum stage {self.current_stage + 1}/{len(self.stage_configs)}: {self.get_current_config()['name']}")
                print(f"New config: {self.get_current_config()}")
                return True
                
        return False
    
    def get_modified_reward_system_params(self):
        """
        Get modified reward system parameters based on current stage.
        """
        config = self.get_current_config()
        return {
            'base_move_penalty': config['base_move_penalty'],
            'collision_penalty': config['collision_penalty'],
            'apple_reward': 10 * config['reward_multiplier'],
            'bad_apple_penalty': -25 * config['reward_multiplier'],
            'approach_reward_factor': 0.5 * config['reward_multiplier']
        }
    
    def get_statistics(self):
        """
        Get statistics for the current curriculum state.
        """
        return {
            'stage': self.current_stage + 1,
            'stage_name': self.get_current_config()['name'],
            'total_stages': len(self.stage_configs),
            'success_count': self.success_count,
            'total_episodes': self.total_episodes,
            'best_score': self.best_score,
            'best_duration': self.best_duration,
            'current_config': self.get_current_config()
        }