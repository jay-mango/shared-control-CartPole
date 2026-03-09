import numpy as np

from stable_baselines3 import PPO # or SAC, depending on what you train with

class AdvancedAgent:
    """PyTorch RL Agent using Stable-Baselines3."""
    
    def __init__(self, model_path=None):
        if model_path:
            # Load your trained PyTorch model
            self.model = PPO.load(model_path)
            self.is_trained = True
        else:
            print("Warning: No model path provided. Agent will output 0.0 force.")
            self.is_trained = False

    def get_action(self, observation):
        if not self.is_trained:
            return 0.0
            
        # SB3 predict expects the observation array
        action, _states = self.model.predict(observation, deterministic=True)
        
        # SB3 continuous actions are usually returned as numpy arrays (e.g., [force])
        # We extract the float to match the BasicAgent interface
        if isinstance(action, np.ndarray):
            return float(action[0])
            
        return float(action)