import numpy as np
import sys
import os
import random
import enum
import pygame
import math
"""
This class defines the forces acting on the objects in the game.
The forces are applied to the objects to simulate the artificial physics of the game.
The forces can be gravity, friction, or any other force that affects the movement of objects.

functions:
- apply_force: applies a force to an object
- update: updates the position of the objects based on the forces applied to them
- check_collision: checks if two objects have collided with each other
- resolve_collision: resolves the collision between two objects
- apply_gravity: applies gravity to an object
- apply_friction: applies friction to an object
- apply_drag: applies drag to an object
- apply_spring: applies spring force to an object
- apply_repulsion: applies repulsion force to an object
- apply_attraction: applies attraction force to an object

ToDo:
- Implement the Forces class. The Forces class should have the following methods:
"""

class Forces(enum.Enum):
    GRAVITY = 1
    FRICTION = 2
    # DRAG = 3
    # SPRING = 4
    # REPULSION = 5
    # ATTRACTION = 6
    ELECTROMAGNETIC = 7


class arPhy():
    def __init__(self, objects=[], forces=[]):
        self.objects = objects
        self.forces = forces
    
    # def apply_force(self, obj, force):
    #     # Apply a force to an object
    #     if 'acceleration' not in obj:
    #         obj['acceleration'] = [0, 0]
    #     obj['acceleration'][0] += force[0] / obj['mass']
    def apply_force(self, obj, force):
        # Skip objects with zero mass to avoid division by zero
        if obj.get('mass', 1) == 0:
            return
        obj['velocity'][0] += force[0] / obj['mass']
        obj['velocity'][1] += force[1] / obj['mass']
    
    
    # def update(self):
    #     # Update the state of all objects
    #     for obj in self.objects:
    #         # Update velocity based on acceleration
    #         obj['velocity'][0] += obj['acceleration'][0]
    #         obj['velocity'][1] += obj['acceleration'][1]
            
    #         # Update position based on velocity
    #         obj['position'][0] += obj['velocity'][0]
    #         obj['position'][1] += obj['velocity'][1]
            
    #         # Reset acceleration for the next frame
    #         obj['acceleration'] = [0, 0]
    def update(self, dt):
        for obj in self.objects:
            # Update position based on velocity
            obj['position'][0] += obj['velocity'][0] * dt
            obj['position'][1] += obj['velocity'][1] * dt


    def check_collision(self, obj1, obj2):
        # Check if two objects are colliding
        return (obj1['position'][0] < obj2['position'][0] + obj2['size'][0] and
                obj1['position'][0] + obj1['size'][0] > obj2['position'][0] and
                obj1['position'][1] < obj2['position'][1] + obj2['size'][1] and
                obj1['position'][1] + obj1['size'][1] > obj2['position'][1])
    
    def resolve_collision(self, obj1, obj2):
        # Resolve collision by swapping velocities
        temp_velocity = obj1['velocity']
        obj1['velocity'] = obj2['velocity']
        obj2['velocity'] = temp_velocity
    
    def apply_gravitation(self, obj):
        # Ensure object has mass before applying gravity
        if 'mass' in obj and obj['mass'] > 0:
            gravity_force = [0, 9.81 * obj['mass']]  # Assuming gravity is 9.81 m/s^2
            self.apply_force(obj, gravity_force)
    
    def apply_friction(self, obj):
        # Apply friction to an object
        friction_force = [-obj['velocity'][0] * obj['friction_coefficient'], 
                          -obj['velocity'][1] * obj['friction_coefficient']]
        self.apply_force(obj, friction_force)

    # def apply_gravity(self, obj1, obj2):
    #     # Calculate the distance between the two objects
    #     dx = obj2['position'][0] - obj1['position'][0]
    #     dy = obj2['position'][1] - obj1['position'][1]
    #     distance = math.sqrt(dx**2 + dy**2)
        
    #     # Calculate the gravitational force
    #     force_magnitude = (6.67430e-11 * obj1['mass'] * obj2['mass']) / (distance**2)
    #     force_x = force_magnitude * (dx / distance)
    #     force_y = force_magnitude * (dy / distance)
        
    #     # Apply the force to both objects
    #     self.apply_force(obj1, [force_x, force_y])
    #     self.apply_force(obj2, [-force_x, -force_y])
    def apply_gravity(self, obj1, obj2):
        G = 6.67430e-11
        dx = obj2['position'][0] - obj1['position'][0]
        dy = obj2['position'][1] - obj1['position'][1]
        distance = math.sqrt(dx**2 + dy**2)
        force_magnitude = G * obj1['mass'] * obj2['mass'] / (distance**2)
        force_x = force_magnitude * dx / distance
        force_y = force_magnitude * dy / distance
        self.apply_force(obj1, [force_x, force_y])
        self.apply_force(obj2, [-force_x, -force_y])

    def apply_collision(self, obj1, obj2):
        if self.check_collision(obj1, obj2):
            self.resolve_collision(obj1, obj2)
        
    

    # def apply_drag(self, obj):
    #     pass
    
    # def apply_spring(self, obj1, obj2):
    #     pass
    
    # def apply_repulsion(self, obj1, obj2):
    #     pass
    
    # def apply_attraction(self, obj1, obj2):
    #     pass


# ------------------------------------------DEMO------------------------------------------------------
# import sys
# sys.path.append('../')
# import screen.record.vidcap as vc

# # Initialize Pygame
# pygame.init()

# # Set up display
# WIDTH, HEIGHT = 1280, 800
# screen = pygame.display.set_mode((WIDTH, HEIGHT))
# pygame.display.set_caption("Orbital Simulation with Pygame")

# # Define colors
# BLACK = (0, 0, 0)
# YELLOW = (255, 255, 0)
# BLUE = (0, 0, 255)
# ORANGE = (255, 165, 0)
# WHITE = (255, 255, 255)
# RED = (255, 0, 0)

# # objects = [
# #     {'id': 0, 'mass': 100.989e30, 'velocity': [0, 0], 'position': [0, 0], 'size': 5, 'color': YELLOW},  # sun
# #     {'id': 1, 'mass': 59.453e18, 'velocity': [-60, 240], 'position': [50.6e9, 0], 'size': 3, 'color': BLUE},  # world
# #     {'id': 2, 'mass': 50.453e5, 'velocity': [-240, 240], 'position': [60.6e9, 0], 'size': 2, 'color': WHITE}, #moon
# #     {'id': 3, 'mass': 30.800e12, 'velocity': [-210, 190], 'position': [0, 90.6e9], 'size': 1, 'color': ORANGE},  # mars
# #     {'id': 4, 'mass': 500.800e29, 'velocity': [-300, 300], 'position': [0, 40.6e9], 'size': 1, 'color': RED},
# # ]


# # objects = [
# #     {'id': 0, 'mass': 100.989e30, 'velocity': [0, 0], 'position': [0, 0], 'size': 5, 'color': YELLOW},  # sun
# #     {'id': 1, 'mass': 400.453e29, 'velocity': [-100, 240], 'position': [50.6e9, 0], 'size': 3, 'color': BLUE},  # world
# #     {'id': 2, 'mass': 50.453e5, 'velocity': [-240, 240], 'position': [60.6e9, 0], 'size': 2, 'color': WHITE}, #moon
# #     {'id': 3, 'mass': 30.800e12, 'velocity': [-210, 190], 'position': [0, 90.6e9], 'size': 1, 'color': ORANGE},  # mars
# #     {'id': 4, 'mass': 500.800e29, 'velocity': [-300, 300], 'position': [0, 40.6e9], 'size': 1, 'color': RED},
# # ]

# objects = [
#     {'id': 0, 'mass': 100.989e30, 'velocity': [0, 0], 'position': [0, 0], 'size': 5, 'color': YELLOW},  # sun
#     {'id': 1, 'mass': 400.453e30, 'velocity': [-100, 240], 'position': [50.6e10, 0], 'size': 3, 'color': BLUE},  # world
#     {'id': 2, 'mass': 50.453e5, 'velocity': [-240, 240], 'position': [60.6e9, 0], 'size': 2, 'color': WHITE}, #moon
#     {'id': 3, 'mass': 30.800e12, 'velocity': [-210, 190], 'position': [0, 90.6e9], 'size': 1, 'color': ORANGE},  # mars
#     {'id': 4, 'mass': 500.800e28, 'velocity': [-410, 410], 'position': [0, 40.6e9], 'size': 1, 'color': RED},
# ]



# # Initialize physics engine
# physics_engine = arPhy(objects, [Forces.GRAVITY])

# # Initialize camera position
# camera_x, camera_y = WIDTH // 2, HEIGHT // 2
# camera_speed = 10  # Adjust camera speed as needed

# def scale_position(pos, camera_pos):
#     scale = 10e-11  # Adjusted scale to zoom out slightly
#     x = camera_pos[0] + pos[0] * scale
#     y = camera_pos[1] + pos[1] * scale
#     return int(x), int(y)
# # Main loop
# running = True
# clock = pygame.time.Clock()
# # vidcap = vc.VidCap("./planets.mp4", resolution=(WIDTH, HEIGHT), fps=240)
# dt = 3600 * 24 * 10  # Time step: 10 days in seconds
# trail_points = {obj['id']: [] for obj in objects}
# max_trail_length = 1000  # Maximum number of points in the trail

# # vidcap.start()

# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#     # Handle keyboard input for camera movement
#     keys = pygame.key.get_pressed()
#     if keys[pygame.K_LEFT]:
#         camera_x += camera_speed
#     if keys[pygame.K_RIGHT]:
#         camera_x -= camera_speed
#     if keys[pygame.K_UP]:
#         camera_y += camera_speed
#     if keys[pygame.K_DOWN]:
#         camera_y -= camera_speed

#     # Apply gravity between the star and the planet
#     for i in range(len(objects)):
#         for j in range(i + 1, len(objects)):
#             physics_engine.apply_gravity(objects[i], objects[j])

#     # Update physics
#     physics_engine.update(dt)

#     # Update camera position to follow the object with id = 0
#     obj_to_follow = next(obj for obj in objects if obj['id'] == 0)
#     camera_x = WIDTH // 2 - obj_to_follow['position'][0] * 10e-11
#     camera_y = HEIGHT // 2 - obj_to_follow['position'][1] * 10e-11

#     # Clear screen
#     screen.fill(BLACK)

#     # Draw objects and append their trail points
#     for obj in objects:
#         scaled_position = scale_position(obj['position'], (camera_x, camera_y))
#         pygame.draw.circle(screen, obj['color'], scaled_position, obj['size'])

#         # Append current position to trail points
#         trail_points[obj['id']].append(list(obj['position']))  # Ensure position is a new list

#         # Limit trail length
#         if len(trail_points[obj['id']]) > max_trail_length:
#             trail_points[obj['id']].pop(0)

#     # Draw orbit trails
#     for obj_id, points in trail_points.items():
#         if len(points) > 1:
#             # Subtract the position of the object the camera is following
#             scaled_trail_points = [
#                 scale_position(
#                     [point[0] - obj_to_follow['position'][0], point[1] - obj_to_follow['position'][1]], 
#                     (WIDTH // 2, HEIGHT // 2)
#                 ) 
#                 for point in points
#             ]
#             # Draw the trail for each object
#             pygame.draw.lines(screen, objects[obj_id]['color'], False, scaled_trail_points, 1)

#     # Update display
#     pygame.display.flip()

#     # Record the screen
#     # vidcap.record(screen)

#     # Cap the frame rate
#     clock.tick(240)

# # vidcap.stop()
# pygame.quit()