import gymnasium as gym
import pygame
import math
import types
import numpy as np

from human.input_handler import InputHandler
from envs.shared_wrapper import SharedControlWrapper
from agents.basic_agent import BasicAgent
from agents.advanced_agent import AdvancedAgent
from game.target_manager import TargetManager

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

    # Bind the new method to the unwrapped environment instance
    unwrapped.step = types.MethodType(continuous_step, unwrapped)
    return env

def main():
    # Initialize Pygame first to avoid console input issues on some systems.
    pygame.init()
    env = None
    try:
        # --- Interactive Agent Selection ---
        print("\n" + "="*30)
        print(" Choose Your Balancing Agent")
        print("="*30)
        print("1. Basic Agent (Mathematical PD Controller)")
        print("2. Advanced Agent (PyTorch RL Model)")
        print("="*30)
        
        choice = ""
        try:
            while choice not in ["1", "2"]:
                choice = input("Enter 1 or 2: ").strip()
        except KeyboardInterrupt:
            print("\n\nSelection cancelled. Exiting.")
            return

        # Initialize the chosen agent
        if choice == "1":
            print("\n--> Initializing Basic PD Agent...")
            agent = BasicAgent()
        elif choice == "2":
            try:
                model_path = input("\nEnter model filename (Press Enter for 'model.zip'): ").strip()
                if not model_path:
                    model_path = "model.zip"
                
                print(f"\n--> Initializing Advanced PyTorch Agent from: {model_path}...")
                agent = AdvancedAgent(model_path=model_path)
            except KeyboardInterrupt:
                print("\n\nInput cancelled. Exiting.")
                return

        # --- Environment and Game Setup ---
        env = gym.make("CartPole-v1", render_mode="human")
        #env.unwrapped.x_threshold = 5.0
        #env.unwrapped.theta_threshold_radians = math.radians(45)
        
        patch_env_to_continuous(env)
        
        input_handler = InputHandler()
        env = SharedControlWrapper(env)

        # --- Game and UI Setup ---
        font = pygame.font.Font(None, 36)
        TARGET_COLOR = (255, 165, 0) # Orange
        TEXT_COLOR = (255, 255, 255)
        score = 0
        
        x_threshold = env.unwrapped.x_threshold
        target_manager = TargetManager(num_targets=1, x_threshold=x_threshold)

        # --- Main Game Loop ---
        clock = pygame.time.Clock()
        observation, info = env.reset()
        
        while True:
            if not input_handler.process_frame():
                break # Exit loop if quit is requested

            agent_action = agent.get_action(observation)
            human_action = input_handler.get_action()

            # Pass both actions to the wrapper, which will handle blending
            actions = (agent_action, human_action)
            observation, reward, terminated, truncated, info = env.step(actions)
            
            # --- Target Logic and Rendering ---
            cart_x = observation[0]
            targets_hit = target_manager.update(cart_x)
            if targets_hit > 0:
                score += targets_hit * 10

            screen = env.unwrapped.screen
            if screen is not None:
                # Draw overlays on the screen rendered by the environment
                world_width = env.unwrapped.x_threshold * 2
                scale = screen.get_width() / world_width
                
                for target_pos in target_manager.get_positions():
                    target_x_pixels = target_pos * scale + screen.get_width() / 2.0
                    target_top = screen.get_height() * 0.4
                    target_bottom = screen.get_height() * 0.8
                    pygame.draw.line(screen, TARGET_COLOR, (target_x_pixels, target_top), (target_x_pixels, target_bottom), 4)

                score_text = font.render(f"Score: {score}", True, TEXT_COLOR)
                screen.blit(score_text, (10, 10))

                pygame.display.flip()

            if terminated or truncated:
                print(f"Pole fell! Final Score: {score}")
                observation, info = env.reset()
                score = 0
                target_manager.reset()
                
            clock.tick(FPS)

    finally:
        # Ensure cleanup happens even if an error occurs or the user quits.
        if env:
            env.close()
        pygame.quit()

if __name__ == "__main__":
    main()
