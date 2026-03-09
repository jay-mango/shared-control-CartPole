import random

class TargetManager:
    """
    Manages the creation and state of multiple targets for the CartPole game.
    """
    def __init__(self, num_targets, x_threshold, reach_threshold=0.25, min_dist=1.0):
        """
        Initializes the TargetManager.

        Args:
            num_targets (int): How many targets should be on screen at once.
            x_threshold (float): The edge of the world space (e.g., 5.0 for a world of -5.0 to 5.0).
            reach_threshold (float): How close the cart must be to a target to "hit" it.
            min_dist (float): Minimum distance between newly spawned targets.
        """
        self.num_targets = num_targets
        self.x_threshold = x_threshold
        self.reach_threshold = reach_threshold
        self.min_dist = min_dist
        self.targets = []
        self.reset()

    def _generate_random_target(self, ignore_index=-1):
        """
        Generates a new target at a random location, ensuring it's not too
        close to other existing targets.

        Args:
            ignore_index (int): The index of a target in self.targets to ignore
                                during distance checks. Used when replacing a target.
        """
        while True:
            new_target_x = random.uniform(-self.x_threshold * 0.9, self.x_threshold * 0.9)
            
            is_valid = True
            for i, t in enumerate(self.targets):
                if i == ignore_index:
                    continue
                if abs(new_target_x - t) < self.min_dist:
                    is_valid = False
                    break
            if is_valid:
                return new_target_x

    def reset(self):
        """
        Resets and creates a fresh set of random targets.
        """
        self.targets.clear()
        for _ in range(self.num_targets):
            self.targets.append(self._generate_random_target()) # ignore_index is -1, checks all

    def update(self, cart_x):
        """
        Checks if the cart has hit any targets and replaces them if so.

        Args:
            cart_x (float): The current x-position of the cart.

        Returns:
            int: The number of targets hit in this update.
        """
        hit_count = 0
        for i, target_x in enumerate(self.targets):
            if abs(cart_x - target_x) < self.reach_threshold:
                # Replace the hit target, ignoring it for the distance check.
                self.targets[i] = self._generate_random_target(ignore_index=i)
                hit_count += 1
        return hit_count

    def get_positions(self):
        """
        Returns the current x-positions of all active targets.
        """
        return self.targets
