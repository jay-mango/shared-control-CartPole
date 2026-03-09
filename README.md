# CartPole Shared Control

This project implements a shared control system for the classic CartPole environment, allowing a human player and an AI agent to collaborate in balancing the pole. The visualization is rendered using Pygame.

## Key Features

- **Shared Control System:** The core of the project is a shared control architecture where both the human player and an AI agent can simultaneously influence the cart's movement.
- **Human-in-the-Loop:** A human player can apply force to the cart using the left and right arrow keys.
- **AI Agent:** A **Proportional-Integral-Derivative (PID)** controller acts as the AI agent. It uses a **deadband strategy**: when the pole is stable, it applies a gentle force to center the cart. When the pole tilts beyond a safe threshold, it switches to aggressive balancing mode.
- **Risk-Aware Arbitration:** The system uses a dynamic `alpha` parameter to blend forces. If the pole is stable, the human is granted high authority (up to 85%). If the pole starts to fall (high risk), the system automatically reduces human authority and increases AI intervention to save the pole.
- **Continuous Action Space:** The standard discrete action space of the CartPole-v1 environment has been modified to support continuous force values, allowing for finer control.
- **Pygame Visualization:** The environment is rendered using Pygame, providing a real-time visual representation of the simulation.

## How It Works

The system integrates `Gymnasium` for the environment and `Pygame` for handling user input and rendering.

1.  **Environment Setup:** The `CartPole-v1` environment is initialized. Its physics are monkey-patched to handle continuous force inputs. The track length and pole angle limits are increased to make the task more manageable for human players.
2.  **Human Input:** The `InputHandler` class captures keyboard events (`K_LEFT`, `K_RIGHT`) from the human player.
3.  **AI Agent:** The `BasicAgent` class implements a tuned PID controller. It switches behaviors based on stability: "Gentle Centering" (using Integral terms to fix steady-state error) when safe, and "Emergency Balancing" when the pole is at risk.
4.  **Dynamic Alpha Calculation:** Inside the `SharedControlWrapper`, the system calculates a "Risk" metric based on the pole's angle.
    - **Safe:** Low angle -> High Alpha (Human has control).
    - **Critical:** High angle -> Low Alpha (Agent takes over).
5.  **Action Blending:** The wrapper blends the Agent's force and the Human's input using the calculated alpha: `blended_force = (1 - alpha) * agent_action + alpha * human_action`.

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the simulation:**
    ```bash
    python src/main.py
    ```

Controls:

- **Left Arrow Key:** Apply force to the left.
- **Right Arrow Key:** Apply force to the right.
- **Close the window or press ESC** to quit.
