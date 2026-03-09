import gymnasium as gym
import numpy as np

class SharedControlWrapper(gym.Wrapper):
    def __init__(self, env, input_handler):
        super().__init__(env)
        self.input_handler = input_handler
        self.last_observation = None
    
    def step(self, action):
        # 1. Get Human Action (controls horizontal movement)
        human_action = self.input_handler.get_action()
        
        # 2. Determine Agent Force and Alpha
        # Check if action is a tuple/list [force, alpha]
        if hasattr(action, "__len__") and len(action) == 2:
            agent_action = float(action[0])
            alpha = float(action[1])
        else:
            # Fallback: Action is just force, calculate alpha internally (Legacy behavior)
            agent_action = float(action)
            alpha = 0.0
            if self.last_observation is not None and abs(human_action) > 0:
                pole_angle = self.last_observation[2]
                safe_threshold = 0.20
                risk = min(abs(pole_angle) / safe_threshold, 1.0)
                alpha = 0.6 * (1.0 - risk)
        
        # 3. Blend Actions into a single net force
        blended_force = ((1.0 - alpha) * agent_action) + (alpha * human_action)
        
        # Step the environment with continuous force (passed as a list)
        observation, reward, terminated, truncated, info = self.env.step([blended_force])
        self.last_observation = observation
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset environment and tracking state"""
        observation, info = self.env.reset(**kwargs)
        self.last_observation = observation
        return observation, info
