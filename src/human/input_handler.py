import pygame

class InputHandler:
    def __init__(self):
        self.human_action = 0.0
        self.quit_requested = False
        self.max_force = 15.0  # Newtons
    
    def handle_events(self):
        '''Process pygame events. Returns False if quit is requested.'''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True
                return False
        return True
    
    def update_input(self):
        '''Update human action based on current key state'''
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.human_action = -self.max_force
        elif keys[pygame.K_RIGHT]:
            self.human_action = self.max_force
        else:
            self.human_action = 0.0
            
    def process_frame(self):
        '''Handle all input processing for one frame'''
        if not self.handle_events():
            return False
        self.update_input()
        return True

    def get_action(self):
        return self.human_action
        
    def is_quit_requested(self):
        return self.quit_requested
