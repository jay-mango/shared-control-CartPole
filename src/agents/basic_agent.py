class BasicAgent:
    """Continuous PD Controller agent for CartPole balancing.
    
    This is not a Neural Network. It is a mathematical formula (PD Controller)
    that calculates the 'optimal' force based on the physics of the system.
    """
    
    def __init__(self):
        # Gains for Goal 1: Keep Pole Upright (High Priority)
        self.kp = 120.0   # Proportional gain (responds to angle)
        self.kd = 25.0    # Derivative gain (responds to velocity)
        self.ki = 5.0     # Integral gain (responds to accumulated angle error)
        
        # Gains for Goal 2: Keep Cart Centered (Low Priority).
        # These must be very small to feel like a gentle "nudge", not a hard pull.
        self.k_cart_p = 0.5  # Position gain (pulls cart to center)
        self.k_cart_v = 1.0  # Velocity gain (dampens movement / braking)
        self.k_cart_i = 0.01 # Integral gain (pulls cart to center over time)
        
        # Deadband: A "safe zone" where the agent's behavior changes.
        self.angle_threshold = 0.05 # radians (~2.8 degrees)
        
        # Integral state storage (Accumulators)
        self.pole_integral = 0.0
        self.cart_integral = 0.0
    
    def get_action(self, observation):
        cart_position, cart_velocity, pole_angle, pole_velocity = observation
        
        # --- Agent Logic ---
        # The agent's behavior depends on whether the pole is in the "safe zone".

        if abs(pole_angle) < self.angle_threshold:
            # --- BEHAVIOR 1: GENTLE CENTERING ---
            # If the pole is stable, the agent's only job is to provide a gentle
            # pull towards the center. This force is small and easily overridden.
            
            # Accumulate error for Cart Integral, reset Pole Integral (since we are safe)
            self.cart_integral += cart_position
            self.pole_integral = 0.0 
            
            centering_force = -1.0 * ((cart_position * self.k_cart_p) + (cart_velocity * self.k_cart_v) + (self.cart_integral * self.k_cart_i))
            return centering_force
        else:
            # --- BEHAVIOR 2: EMERGENCY BALANCING ---
            # If the pole is at risk, the agent's ONLY priority is to balance it.
            # It ignores centering to focus on the critical task.
            
            # Accumulate error for Pole Integral, reset Cart Integral
            self.pole_integral += pole_angle
            self.cart_integral = 0.0
            
            # Clamp integral to prevent "Windup" (instability if error gets too large)
            self.pole_integral = max(min(self.pole_integral, 5.0), -5.0)
            
            balance_force = (pole_angle * self.kp) + (pole_velocity * self.kd) + (self.pole_integral * self.ki)
            return balance_force
