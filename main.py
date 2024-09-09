import pygame
import os
from player.p1 import Player
from food.f1 import Food, FoodType, FoodGroup
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Food group
foodgroup = FoodGroup(15, 15, 15)

# Create player instance
player1 = Player(100, 100, os.path.join("./", "pacman-png-18.png"), 25, 25, foodgroup, load_weights=True)

# Main game loop
running = True
clock = pygame.time.Clock()

# Training parameters
epochs_per_episode = 500  # Number of epochs per episode
num_episodes = 1500 

# Curriculum Learning Functions


def train(player, screen, num_episodes, max_steps_per_episode=1000, log_interval=100):
    for episode in range(num_episodes):
        screen.fill((0, 0, 0))  # Clear the screen (black)

        # Reset the player and food group at the start of each episode
        player.rect.x, player.rect.y = 50, 50  # Reset player position
        player.points = 0  # Reset points
        player.food_group.respawn_all()  # Respawn food items for the new episode

        running = True
        step = 0  # Track the number of steps in the current episode
        total_loss = 0
        loss_count = 0
        
        while running and step < max_steps_per_episode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Player's update method will handle movement, action selection, and training
            player.update(screen)

            # Redraw the game entities
            screen.fill((0, 0, 0))  # Clear the screen for each frame
            player.food_group.draw(screen)
            screen.blit(player.image, player.rect)
            
            # Update the display
            pygame.display.flip()

            # Increment the step count
            step += 1  

            # After every 100 steps, log the average loss
            if step % log_interval == 0:
                if player.current_loss is not None:
                    total_loss += player.current_loss
                    loss_count += 1
                    avg_loss = total_loss / loss_count
                    print(f"Episode {episode + 1}, Step {step}, Avg Loss: {avg_loss:.4f}")
                  

            
            # End the episode if the player accumulates a high score
            if player.points >= 10:  # Example condition to end the episode early
                print(f"Episode {episode + 1} ended with {player.points} points")
                break

        
        player.save_model(f"player_model.pth")

        print(f"Episode {episode + 1} completed with {player.points} points after {step} steps")

train(player1, screen, num_episodes=100)

pygame.quit()
#----------------------------------------------------------------------------------
# # Initialize Pygame
# pygame.init()

# # Screen dimensions
# SCREEN_WIDTH = 800
# SCREEN_HEIGHT = 600

# # Colors
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)

# # Set up the display
# screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
# pygame.display.set_caption("Player Example")

# # Food group
# foodgroup = FoodGroup(15, 15, 15)

# # Create player instances
# player1 = Player(100, 100, os.path.join("./", "pacman-png-18.png"), 25, 25, foodgroup, load_weights=True)


# # Main game loop
# running = True
# clock = pygame.time.Clock()

# # Training loop
# epochs_per_episode = 500  # Number of epochs per episode
# num_episodes = 1500 

# # for episode in range(num_episodes):
# #     print(f"\nEpisode: {episode + 1}")

# #     # Reset player position at the start of each episode
# #     player1.rect.x = 100
# #     player1.rect.y = 100

# #     for epoch in range(epochs_per_episode):
# #         for event in pygame.event.get():
# #             if event.type == pygame.QUIT:
# #                 running = False

# #         # Update player
# #         player1.update(screen)

# #         # Get loss for printing (if available)
# #         loss = player1.current_loss if hasattr(player1, 'current_loss') else None

# #         # Clear the screen
# #         screen.fill(BLACK)
# #         # Draw the player
# #         screen.blit(player1.image, player1.rect)

# #         # Draw the food
# #         foodgroup.draw(screen)

# #         # Update the display
# #         pygame.display.flip()

# #         # Cap the frame rate
# #         clock.tick(60) 

# #         if epoch % 100 == 0:
# #             print(f"Epoch {epoch}, Loss: {loss}")


# #     player1.save_model("player_model.pth")
# # Main training loop

# def generate_curriculum(self, num_stages):
#     curriculum = []
#     for stage in range(num_stages):
#         num_food = 1 + stage  # Increase number of food items gradually
#         food_speed = stage * 0.1  # Increase food speed gradually
#         curriculum.append({'num_food': num_food, 'food_speed': food_speed})
#     return curriculum

# def train_curriculum(self, curriculum, epochs_per_stage):
#     for stage, params in enumerate(curriculum):
#         print(f"Starting stage {stage + 1}")
#         self.food_group.set_params(params['num_food'], params['food_speed'])
#         for epoch in range(epochs_per_stage):
#             self.train_episode()
#         print(f"Completed stage {stage + 1}")

# def evaluate(self, num_episodes=10):
#     total_reward = 0
#     for _ in range(num_episodes):
#         episode_reward = self.play_episode(training=False)
#         total_reward += episode_reward
#     average_reward = total_reward / num_episodes
#     return average_reward

# def train(self, total_epochs, eval_interval=100):
#     curriculum = self.generate_curriculum(5)  # 5 stages of curriculum
#     epochs_per_stage = total_epochs // len(curriculum)
    
#     for stage, params in enumerate(curriculum):
#         print(f"Starting stage {stage + 1}")
#         self.food_group.set_params(params['num_food'], params['food_speed'])
        
#         for epoch in range(epochs_per_stage):
#             loss = self.train_episode()
            
#             if epoch % eval_interval == 0:
#                 avg_reward = self.evaluate()
#                 print(f"Stage {stage + 1}, Epoch {epoch}, Loss: {loss:.4f}, Avg Reward: {avg_reward:.2f}")
        
#         print(f"Completed stage {stage + 1}")
    
#     self.save_model("final_model.pth")

# train(player1, 1000, 100)

# pygame.quit()
#----------------------------------------------------------------------------------

# import pygame
# import os
# from player.p1 import Player
# from food.f1 import Food, FoodType, FoodGroup
# import random
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# # Initialize Pygame
# pygame.init()

# # Screen dimensions
# SCREEN_WIDTH = 800
# SCREEN_HEIGHT = 600

# # Colors
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)

# # Set up the display
# screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
# pygame.display.set_caption("Player Example")

# # Food group
# foodgroup = FoodGroup(20, 25, 25)

# # Create player instances
# player1 = Player(100, 100, os.path.join("./", "pacman-png-18.png"), 25, 25, foodgroup, load_weights=False)
# # player2 = Player(200, 200, os.path.join("./", "pacman-png-18.png"), 25, 25, foodgroup, load_weights=False)

# # # Mini screen dimensions
# # MINI_SCREEN_WIDTH = 120
# # MINI_SCREEN_HEIGHT = 120
# # mini_screen = pygame.Surface((MINI_SCREEN_WIDTH, MINI_SCREEN_HEIGHT))

# # Main game loop
# running = True
# clock = pygame.time.Clock()

# # Training loop
# epochs = 1000
# best_loss = float('inf')
# for epoch in range(epochs):
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#     # Randomly choose a target direction for training
#     target_direction = random.randint(0, 3)
#     # target_direction2 = random.randint(0, 3)

#     # Train the model
#     loss = player1.train(screen, target_direction)
#     # loss2 = player2.train(screen, target_direction2)

#     # Capture the old state
#     old_state = (player1.rect.x, player1.rect.y)
#     # old_state2 = (player2.rect.x, player2.rect.y)

#     # Update player
#     player1.update(screen)
#     # player2.update(screen)

#     # Capture the new state
#     new_state = (player1.rect.x, player1.rect.y)
#     # new_state2 = (player2.rect.x, player2.rect.y)

#     # Check if the player ate the food
#     reward1 = -1  # Default reward is -1 (penalty for not eating food)
#     if player1.eat():
#         print("Player1 ate the food!")
#         foodgroup.respawn_all()
#         reward1 = 20  # Reward for eating food

#     # reward2 = -1  # Default reward is -1 (penalty for not eating food)
#     # if player2.eat():
#     #     print("Player2 ate the food!")
#     #     foodgroup.respawn_all()
#     #     reward2 = 40  # Reward for eating food

#     # Update the Q-table using Q-learning
#     player1.q_learning_update(old_state, target_direction, reward1, new_state)
#     # player2.q_learning_update(old_state2, target_direction2, reward2, new_state2)

#     # Clear the screen
#     screen.fill(BLACK)

#     # Draw the player
#     screen.blit(player1.image, player1.rect)
#     # screen.blit(player2.image, player2.rect)

#     # Draw the food
#     foodgroup.draw(screen)

#     # # Capture the player's vision and draw it on the mini screen
#     # mini_screen.fill(BLACK)
#     # grid_size = 120
#     # for i in range(grid_size):
#     #     for j in range(grid_size):
#     #         pixel_x = player1.rect.x + i - grid_size // 2
#     #         pixel_y = player1.rect.y + j - grid_size // 2
#     #         if 0 <= pixel_x < 800 and 0 <= pixel_y < 600:
#     #             color = screen.get_at((pixel_x, pixel_y))[:3]
#     #             mini_screen.set_at((i, j), color)

#     # # Draw a border around the mini screen
#     # pygame.draw.rect(mini_screen, WHITE, mini_screen.get_rect(), 1)

#     # # Blit the mini screen to the main screen
#     # screen.blit(mini_screen, (SCREEN_WIDTH - MINI_SCREEN_WIDTH, SCREEN_HEIGHT - MINI_SCREEN_HEIGHT))

#     # Update the display
#     pygame.display.flip()

#     # Cap the frame rate
#     clock.tick(120)

#     # Print the loss every 100 epochs
#     if epoch % 100 == 0:
#         print(f"Epoch {epoch}, Loss1: {loss}")

#     # Save model weights every 100 epochs if the loss is lower than the best loss
#     if epoch % 100 == 0:
#         player1.save_model("player_model.pth")

# pygame.quit()

# import pygame
# import os
# from player.p1 import Player
# from food.f1 import Food, FoodType, FoodGroup
# import random
# import torch
# import numpy as np

# # Initialize Pygame
# pygame.init()

# # Screen dimensions
# SCREEN_WIDTH = 800
# SCREEN_HEIGHT = 600

# # Colors
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)

# # Set up the display
# screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
# pygame.display.set_caption("Player Example")

# # food = Food(25, 25)
# foodgroup = FoodGroup(15, 25, 25)

# # Create a player instance
# player1 = Player(100, 100, os.path.join("./", "pacman-png-18.png"), 25,25, foodgroup, load_weights=True)
# player2 = Player(200, 200, os.path.join("./", "pacman-png-18.png"), 25,25, foodgroup, load_weights=True)



# # Mini screen dimensions
# MINI_SCREEN_WIDTH = 120
# MINI_SCREEN_HEIGHT = 120
# mini_screen = pygame.Surface((MINI_SCREEN_WIDTH, MINI_SCREEN_HEIGHT))

# # Main game loop
# running = True
# clock = pygame.time.Clock()
# # Training loop
# # Training loop
# epochs = 1000
# for epoch in range(epochs):
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
    
#     # Randomly choose a target direction for training
#     target_direction = random.randint(0, 3)
#     target_direction2 = random.randint(0, 3)
    
#     # Train the model
#     loss = player1.train(screen, target_direction)
#     loss2 = player2.train(screen, target_direction2)
    
#     # Capture the old state
#     old_state = (player1.rect.x, player1.rect.y)
#     old_state2 = (player2.rect.x,   player2.rect.y)
    
#     # Update player
#     player1.update(screen)
#     player2.update(screen)
    
#     # Capture the new state
#     new_state = (player1.rect.x, player1.rect.y)
#     new_state2 = (player2.rect.x, player2.rect.y)
    
#     # Check if the player ate the food
#     reward = -2  # Default reward is -1 (penalty for not eating food)
#     if player1.eat() or player2.eat():
#         print("Player ate the food!")
#         foodgroup.respawn_all()
#         reward = 50  # Reward for eating food
    
#     # Update the Q-table using Q-learning
#     player1.q_learning_update(old_state, target_direction, reward, new_state)
#     player2.q_learning_update(old_state2, target_direction, reward, new_state2)
    
#     # Clear the screen
#     screen.fill(BLACK)
    
#     # Draw the player
#     screen.blit(player1.image, player1.rect)
#     screen.blit(player2.image, player2.rect)

#     # Draw the food
#     foodgroup.draw(screen)

#     # Capture the player's vision and draw it on the mini screen
#     mini_screen.fill(BLACK)
#     grid_size = 120
#     for i in range(grid_size):
#         for j in range(grid_size):
#             pixel_x = player1.rect.x + i - grid_size // 2
#             pixel_y = player1.rect.y + j - grid_size // 2
#             if 0 <= pixel_x < 800 and 0 <= pixel_y < 600:
#                 color = screen.get_at((pixel_x, pixel_y))[:3]
#                 mini_screen.set_at((i, j), color)
    
#     # Draw a border around the mini screen
#     pygame.draw.rect(mini_screen, WHITE, mini_screen.get_rect(), 1)
    
#     # Blit the mini screen to the main screen
#     screen.blit(mini_screen, (SCREEN_WIDTH - MINI_SCREEN_WIDTH, SCREEN_HEIGHT - MINI_SCREEN_HEIGHT))
    
#     # Update the display
#     pygame.display.flip()
    
#     # Cap the frame rate
#     clock.tick(120)

#     # Print the loss every 100 epochs
#     if epoch % 100 == 0:
#         print(f"Epoch {epoch}, Loss1: {loss}, Loss2: {loss2}")

#     # Save model weights every 100 epochs
#     #Compare the loss values of the two players and save the model of the player with the lower loss
#     if epoch % 100 == 0:
#         if loss < loss2:
#             player1.save_model("player_model.pth")
#         else:
#             player2.save_model("player_model.pth")


# pygame.quit()
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