class BasicAgent:
    """Continuous PD Controller agent for CartPole balancing.
    
    This is not a Neural Network. It is a mathematical formula (PD Controller)
    that calculates the 'optimal' force based on the physics of the system.
    """
    
    def __init__(self):
        # Gains for Goal 1: Keep Pole Upright (High Priority)
        self.kp = 150.0   # Proportional gain (responds to angle)
        self.kd = 20.0    # Derivative gain (responds to velocity)
        
        # Gains for Goal 2: Keep Cart Centered (Low Priority)
        self.k_cart_p = 3.0  # Position gain (pulls cart to center)
        self.k_cart_v = 6.0  # Velocity gain (dampens movement / braking)
    
    def get_action(self, observation):
        cart_position, cart_velocity, pole_angle, pole_velocity = observation
        
        # 1. Balance Force: React to the pole falling
        # If pole tips Right (+), we push Right (+) to get under it.
        balance_force = (pole_angle * self.kp) + (pole_velocity * self.kd)
        
        # 2. Centering Force: React to the cart drifting away
        # If cart is on the Right (+), we push Left (-) to return to center.
        centering_force = -1.0 * ((cart_position * self.k_cart_p) + (cart_velocity * self.k_cart_v))
        
        # Total Force is the sum of both desires
        total_force = balance_force + centering_force
        
        return total_force
