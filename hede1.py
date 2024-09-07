import pygame
import os
from player.p1 import Player
from food.f1 import Food, FoodType, FoodGroup
import random
import torch
import numpy as np

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Player Example")

# food = Food(25, 25)
foodgroup = FoodGroup(15, 25, 25)

# Create a player instance
player1 = Player(100, 100, os.path.join("./", "pacman-png-18.png"), 25,25, foodgroup, load_weights=True)
player2 = Player(200, 200, os.path.join("./", "pacman-png-18.png"), 25,25, foodgroup, load_weights=True)



# Mini screen dimensions
MINI_SCREEN_WIDTH = 120
MINI_SCREEN_HEIGHT = 120
mini_screen = pygame.Surface((MINI_SCREEN_WIDTH, MINI_SCREEN_HEIGHT))

# Main game loop
running = True
clock = pygame.time.Clock()
# Training loop
# Training loop
epochs = 1000
for epoch in range(epochs):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Randomly choose a target direction for training
    target_direction = random.randint(0, 3)
    target_direction2 = random.randint(0, 3)
    
    # Train the model
    loss = player1.train(screen, target_direction)
    loss2 = player2.train(screen, target_direction2)
    
    # Capture the old state
    old_state = (player1.rect.x, player1.rect.y)
    old_state2 = (player2.rect.x,   player2.rect.y)
    
    # Update player
    player1.update(screen)
    player2.update(screen)
    
    # Capture the new state
    new_state = (player1.rect.x, player1.rect.y)
    new_state2 = (player2.rect.x, player2.rect.y)
    
    # Check if the player ate the food
    reward = -2  # Default reward is -1 (penalty for not eating food)
    if player1.eat() or player2.eat():
        print("Player ate the food!")
        foodgroup.respawn_all()
        reward = 50  # Reward for eating food
    
    # Update the Q-table using Q-learning
    player1.q_learning_update(old_state, target_direction, reward, new_state)
    player2.q_learning_update(old_state2, target_direction, reward, new_state2)
    
    # Clear the screen
    screen.fill(BLACK)
    
    # Draw the player
    screen.blit(player1.image, player1.rect)
    screen.blit(player2.image, player2.rect)

    # Draw the food
    foodgroup.draw(screen)

    # Capture the player's vision and draw it on the mini screen
    mini_screen.fill(BLACK)
    grid_size = 120
    for i in range(grid_size):
        for j in range(grid_size):
            pixel_x = player1.rect.x + i - grid_size // 2
            pixel_y = player1.rect.y + j - grid_size // 2
            if 0 <= pixel_x < 800 and 0 <= pixel_y < 600:
                color = screen.get_at((pixel_x, pixel_y))[:3]
                mini_screen.set_at((i, j), color)
    
    # Draw a border around the mini screen
    pygame.draw.rect(mini_screen, WHITE, mini_screen.get_rect(), 1)
    
    # Blit the mini screen to the main screen
    screen.blit(mini_screen, (SCREEN_WIDTH - MINI_SCREEN_WIDTH, SCREEN_HEIGHT - MINI_SCREEN_HEIGHT))
    
    # Update the display
    pygame.display.flip()
    
    # Cap the frame rate
    clock.tick(120)

    # Print the loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss1: {loss}, Loss2: {loss2}")

    # Save model weights every 100 epochs
    #Compare the loss values of the two players and save the model of the player with the lower loss
    if epoch % 100 == 0:
        if loss < loss2:
            player1.save_model("player_model.pth")
        else:
            player2.save_model("player_model.pth")


pygame.quit()
# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
    
#     # Update player
#     player1.update(screen)
    
#     if player1.eat():
#         print("Player ate the food!")
#         food.respawn()
    
#     # Clear the screen
#     screen.fill(BLACK)
    
#     # Draw the player
#     screen.blit(player1.image, player1.rect)

#     # Draw the player's points
#     screen.blit(player1.text, player1.text_rect)

#     # Draw the food
#     screen.blit(food.image, food.rect)
    
#     # Update the display
#     pygame.display.flip()
    
#     # Cap the frame rate
#     clock.tick(60)

# pygame.quit()