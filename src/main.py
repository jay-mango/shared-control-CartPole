import gymnasium as gym
import pygame
import math
from human.input_handler import InputHandler
from envs.shared_wrapper import SharedControlWrapper
from agents.basic_agent import BasicAgent

FPS = 60  # Target frame rate

def patch_env_to_continuous(env):
    """
    Monkey-patch the CartPole environment to accept continuous force actions.
    Standard CartPole-v1 only accepts 0 or 1. This allows float inputs (e.g., 4.5 N).
    """
    unwrapped = env.unwrapped
    
    # Store the original step to preserve other logic if needed, 
    # but we need to rewrite the physics part for continuous force.
    # We'll replace the step method with one that handles float actions.
    
    def continuous_step(self, action):
        # Extract force from list/array or float
        force = float(action[0]) if hasattr(action, "__len__") else float(action)
        
        # --- Physics from standard CartPole, adapted for continuous force ---
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # Physics
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Euler integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)
        
        # Termination conditions
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if self.render_mode == "human":
            self.render()
            
        return np.array(self.state, dtype=np.float32), 1.0, terminated, False, {}

    import types
    import numpy as np
    # Bind the new method to the unwrapped environment instance
    unwrapped.step = types.MethodType(continuous_step, unwrapped)
    return env

def main():
    # Initialize pygame before creating any pygame components
    pygame.init()
    
    # 1. Setup Environment
    env = gym.make("CartPole-v1", render_mode="human")

    # Increase limits for easier manual testing
    env.unwrapped.x_threshold = 5.0  # Wider track (Default 2.4)
    env.unwrapped.theta_threshold_radians = math.radians(45)  # 45 degrees (Default 12)
    
    # Patch environment to be continuous
    patch_env_to_continuous(env)
    
    # 2. Setup Input Handler
    input_handler = InputHandler()
    
    # 3. Wrap environment for shared control
    env = SharedControlWrapper(env, input_handler)

    # 4. Setup Agent (The "Self Balancer")
    agent = BasicAgent()

    # 5. Setup frame rate control
    clock = pygame.time.Clock()
    observation, info = env.reset()
    
    running = True
    while running:
        # Handle all input for this frame
        if not input_handler.process_frame():
            running = False
            break

        # Agent always calculates action for stabilization
        agent_action = agent.get_action(observation)

        # --- Dynamic Alpha Calculation ---
        # Calculate alpha here so we can eventually replace it with an RL model
        human_force = input_handler.get_action()
        alpha = 0.0
        
        if abs(human_force) > 0:
            pole_angle = observation[2]
            safe_threshold = 0.20  # ~12 degrees
            risk = min(abs(pole_angle) / safe_threshold, 1.0)
            alpha = 0.6 * (1.0 - risk)

        # Pass both [force, alpha] to the environment
        observation, reward, terminated, truncated, info = env.step([agent_action, alpha])

        # Reset if episode is done
        if terminated or truncated:
            observation, info = env.reset()
        # Disable reset so you can keep moving even if the pole falls or hits the edge
        # if terminated or truncated:
        #     observation, info = env.reset()
        
        # Control frame rate
        clock.tick(FPS)

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()
