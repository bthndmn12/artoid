import pygame
import numpy as np
import sys
import random
from space import Space
from ar_physics.forces import arPhy


class Simulation:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.space = Space(width, height)
        self.physics = arPhy(objects=[])
        self.screen = pygame.display.set_mode((width, height))
        self.running = True
        self.clock = pygame.time.Clock()

    def add_object(self, obj, x=None, y=None):
        """Adds an object with its attributes to the simulation."""
        if 'mass' not in obj:
            obj['mass'] = 1  # Default mass
        if 'velocity' not in obj:
            obj['velocity'] = [0, 0]  # Default velocity
        if 'size' not in obj:
            obj['size'] = [20, 20]  # Default size

        self.space.add_object(obj, x, y)
        self.physics.objects = self.space.get_objects()

    def run(self):
        """Runs the simulation."""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            # # Apply physics and update object positions
            # for obj in self.physics.objects:
            #     self.physics.apply_gravitation(obj)

            for i in range(len(self.physics.objects)):
                for j in range(i + 1, len(self.physics.objects)):
                    self.physics.apply_gravity(self.physics.objects[i], self.physics.objects[j])
                    self.physics.apply_collision(self.physics.objects[i], self.physics.objects[j])
                    
                    

            # Update positions
            self.physics.update(dt=3600 * 24 * 50)

            # Check for border collisions with the screen edges
            for obj in self.physics.objects:
                # Check for collision with bottom border
                if obj['position'][1] + obj['size'][1] > self.height:
                    obj['position'][1] = self.height - obj['size'][1]
                    obj['velocity'][1] = 0  # Stop falling
                
                # Check for collision with top border
                if obj['position'][1] < 0:
                    obj['position'][1] = 0
                    obj['velocity'][1] = 0  # Stop moving up
                
                # Check for collision with right border
                if obj['position'][0] + obj['size'][0] > self.width:
                    obj['position'][0] = self.width - obj['size'][0]
                    obj['velocity'][0] = 0  # Stop moving right
                
                # Check for collision with left border
                if obj['position'][0] < 0:
                    obj['position'][0] = 0
                    obj['velocity'][0] = 0  # Stop moving left

            # Render updated objects
            self.space.render(self.screen)

            self.clock.tick(240)  # Limit to 60 FPS

        pygame.quit()

if __name__ == "__main__":
    pygame.init()
    simulation = Simulation(600, 400)

    # # Create objects with necessary physics properties
    # object1 = {
    #     'position': [50, 50],
    #     'size': [10, 10],
    #     'mass': 10000,
    #     'draw': lambda screen, obj: pygame.draw.rect(screen, (0, 255, 0), (int(obj['position'][0]), int(obj['position'][1]), obj['size'][0], obj['size'][1]))
    # }
    # object2 = {
    #     'position': [200, 200],
    #     'size': [10, 10],

    #     'mass': 1000,
    #     'draw': lambda screen, obj: pygame.draw.circle(screen, (0, 0, 255), (int(obj['position'][0]), int(obj['position'][1])), obj['size'][0] // 2)
    # }
    # Create objects with necessary physics properties
    objects = []

    for i in range(100):
        object_i = {
            'position': [50 + i * 10, 50 + i * 10],  # Increment position for each object
            'radius': 5,  # Radius of the circle
            'mass': random.randint(0, 10000),  # Random mass between 1000 and 10000
            'color': (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),  # Random color
            'draw': lambda screen, obj: pygame.draw.circle(screen, obj['color'], (int(obj['position'][0]), int(obj['position'][1])), obj['radius'])
        }
        objects.append(object_i)

    # Add objects to the simulation
    # simulation.add_object(object1, 100, 100)
        # Add objects to the simulation
    for obj in objects:
        simulation.add_object(obj)

    # Run the simulation
    simulation.run()