import gymnasium as gym
import numpy as np

class SharedControlWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_observation = None
    
    def step(self, actions):
        """
        Steps the environment by blending agent and human actions.
        :param actions: A tuple of (agent_action, human_action).
        """
        # 1. Unpack agent and human actions
        agent_action, human_action = actions
        agent_action = float(agent_action)
        human_action = float(human_action)

        # Handle Edge Case: If wrapper was initialized after env.reset(), last_observation is None.
        # We try to fetch the state from the unwrapped environment to allow human control.
        if self.last_observation is None and hasattr(self.env.unwrapped, "state"):
            if self.env.unwrapped.state is not None:
                self.last_observation = np.array(self.env.unwrapped.state, dtype=np.float32)

        # 2. Calculate alpha (human authority) based on risk
        alpha = 0.0
        # Only grant human control if they are providing input and we have an observation
        if abs(human_action) > 0 and self.last_observation is not None:
            pole_angle = self.last_observation[2]
            # Risk is 0 when pole is upright, 1 when at the safe_threshold
            safe_threshold = 0.20  # Corresponds to ~11.5 degrees
            risk = min(abs(pole_angle) / safe_threshold, 1.0)
            
            # Alpha is high (e.g., 0.6) when risk is low, and 0 when risk is high.
            # This gives control to the human when it's safe.
            max_human_authority = 0.85
            alpha = max_human_authority * (1.0 - risk)
        
        # 3. Blend actions. If alpha is 0, only agent_action is used.
        blended_force = ((1.0 - alpha) * agent_action) + (alpha * human_action)
        
        # 4. Step the underlying environment with the blended force
        observation, reward, terminated, truncated, info = self.env.step([blended_force])
        self.last_observation = observation
        
        # Add alpha to info dict for debugging/display
        info['alpha'] = alpha
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset environment and tracking state"""
        observation, info = self.env.reset(**kwargs)
        self.last_observation = observation
        return observation, info
