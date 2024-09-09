import pygame
import numpy as np
import sys
sys.path.append("../")
import os
import random


"""
This class represents the Space object in the game. 
The Space object is the environment in which the player and other objects exist.
Space is a 2D grid of cells, where each cell can be empty or occupied by an object.

functions:
- create_space: creates the 2D grid of cells
- add_object: adds an object to the space
- remove_object: removes an object from the space
- get_objects: returns a list of objects in the space


ToDo: Implement the Space class. The Space class should have the following methods:

- Physic class is goinf to be implemented in the future.



"""

import pygame
import numpy as np
import sys
import random

import pygame
import numpy as np
import random

class Space:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.objects = []

    def add_object(self, obj, x=None, y=None):
        """Adds an object at a specific position or at a random position if not provided."""
        if x is None or y is None:
            x, y = self.get_random_position()
        obj['position'] = [x, y]
        obj['velocity'] = [0, 0]
        self.objects.append(obj)

    def remove_object(self, obj):
        """Removes an object from the space."""
        self.objects.remove(obj)

    def get_objects(self):
        """Returns a list of objects in the space."""
        return self.objects

    def get_random_position(self):
        """Generates a random position within the bounds of the space."""
        x = random.uniform(0, self.width)
        y = random.uniform(0, self.height)
        return x, y

    def move_object(self, obj, dx, dy):
        """Moves an object by a delta in x and y."""
        obj['position'][0] += dx
        obj['position'][1] += dy
        # Ensure the object remains within bounds
        obj['position'][0] = min(max(0, obj['position'][0]), self.width)
        obj['position'][1] = min(max(0, obj['position'][1]), self.height)

    def check_collision(self, obj1, obj2):
        """Checks if two objects are overlapping."""
        distance = ((obj1['position'][0] - obj2['position'][0]) ** 2 + 
                    (obj1['position'][1] - obj2['position'][1]) ** 2) ** 0.5
        return distance < (obj1['radius'] + obj2['radius'])  # Assuming objects have 'radius' for collision

    def render(self, screen):
        """Renders the space and its objects on the screen."""
        screen.fill((0, 0, 0))  # Clear the screen
        for obj in self.objects:
            if 'draw' in obj and callable(obj['draw']):
                obj['draw'](screen, obj)  # Call the object's draw method
            else:
                # Default drawing if no 'draw' method is provided
                pygame.draw.circle(screen, (255, 0, 0), (int(obj['position'][0]), int(obj['position'][1])), obj.get('radius', 10))
        pygame.display.flip()