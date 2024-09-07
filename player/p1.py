import pygame
import torch
import numpy as np
from l_brain.p1_n import PlayerNN

class Player(pygame.sprite.Sprite):
    def __init__(self, x, y, image_path, width, height, food_group, points=0, load_weights=True):
        super().__init__()
        
        # Load the player's sprite from an image file and resize it
        original_image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(original_image, (width, height))
        
        # Get a rectangle that represents the dimensions of the image
        self.rect = self.image.get_rect()
        self.rect.x, self.rect.y = x, y
        
        self.velocity = [0, 0]
        self.points = points
        self.food_group = food_group

        # Initialize the neural network
        self.nn = PlayerNN().to("cuda" if torch.cuda.is_available() else "cpu")
        # self.optimizer = torch.optim.AdamW(self.nn.parameters(), lr=0.001)
        # self.criterion = torch.nn.MSELoss()
        # self.criterion = torch.nn.CrossEntropyLoss()
        # self.criterion = torch.nn.BCEWithLogitsLoss()
        # self.criterion = torch.nn.MSELoss()

        # Load model weights if specified
        if load_weights:
            self.load_model_weights("player_model.pth")


        # Q-learning parameters
        self.q_table = np.zeros((800, 600, 4))  # Q-table for 800x600 grid and 4 actions
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1

        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration

        # # Font for rendering points
        # self.font = pygame.font.Font(None, 36)
        # self.update_text()
    
    
    def move(self, dx, dy):
        """Move the player by changing its rect's position."""
        new_rect = pygame.Rect(self.rect.x + dx * 5, self.rect.y + dy * 5, self.rect.width, self.rect.height)
        
        # Check bounds to prevent player from moving off-screen
        if new_rect.x < 0:
            new_rect.x = 0
        elif new_rect.x + new_rect.width > 800:
            new_rect.x = 800 - new_rect.width
        
        if new_rect.y < 0:
            new_rect.y = 0
        elif new_rect.y + new_rect.height > 600:
            new_rect.y = 600 - new_rect.height
        
        self.rect = new_rect
    
    # def update(self):
    #     """Update the velocity and move accordingly."""
    #     keys = pygame.key.get_pressed()
        
    #     if keys[pygame.K_LEFT]:
    #         self.move(-0.5, 0)
    #     elif keys[pygame.K_RIGHT]:
    #         self.move(0.5, 0)
    #     if keys[pygame.K_UP]:
    #         self.move(0, -0.5)
    #     elif keys[pygame.K_DOWN]:
    #         self.move(0, 0.5)

    #     # Press Ctrl to move faster
    #     if keys[pygame.K_LCTRL]:
    #         self.move(self.velocity[0] * 5, self.velocity[1] * 5)

    def update(self, screen):
        """Update the velocity and move accordingly."""
        # Capture the 120x120 grid of pixel colors around the player's position
        grid_size = 120
        grid_colors = np.zeros((grid_size, grid_size, 3))
        for i in range(grid_size):
            for j in range(grid_size):
                pixel_x = self.rect.x + i - grid_size // 2
                pixel_y = self.rect.y + j - grid_size // 2
                if 0 <= pixel_x < 800 and 0 <= pixel_y < 600:
                    grid_colors[i, j] = screen.get_at((pixel_x, pixel_y))[:3]  # Get RGB values

        # Flatten and normalize the grid colors
        grid_colors /= 255.0
        input_tensor = torch.tensor(grid_colors.flatten(), dtype=torch.float32).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

        # Epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            direction = np.random.randint(0, 4)  # Exploration: random direction
        else:
            output = self.nn(input_tensor)  # Exploitation: neural network output
            direction = torch.argmax(output).item()  # Choose best direction based on NN

        # Decay epsilon after each update
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Move the player based on the direction
        if direction == 0:
            self.move(-1, 0)  # Left
        elif direction == 1:
            self.move(1, 0)   # Right
        elif direction == 2:
            self.move(0, -1)  # Up
        elif direction == 3:
            self.move(0, 1)   # Down

    def eat(self):
        """Check if the player collides with the foodgroup and update points."""
        if self.food_group.check_collision(self.rect):
            self.points += 1
            # self.update_text()
            return True
        return False
        # """Check if the player collides with the food and update points."""
        # if self.rect.colliderect(self.food_group.rect):s
        #     self.points += 1
        #     self.update_text()
        #     return True
        # return False
    
    # def update_text(self):
    #     """Update the text surface with the current points."""
    #     self.text = self.font.render(str(self.points), True, (255, 255, 255))
    #     self.text_rect = self.text.get_rect()
    #     self.text_rect.center = (self.rect.x + self.rect.width // 2, self.rect.y + self.rect.height // 2)

    # """Train the neural network using the 10x10 grid of pixel colors around the player's position"""
    # def train(self, screen, target_direction):
    #     """Train the neural network."""
    #     # Capture the 10x10 grid of pixel colors around the player's position
    #     grid_size = 10
    #     grid_colors = np.zeros((grid_size, grid_size, 3))
    #     for i in range(grid_size):
    #         for j in range(grid_size):
    #             pixel_x = self.rect.x + i - grid_size // 2
    #             pixel_y = self.rect.y + j - grid_size // 2
    #             if 0 <= pixel_x < 800 and 0 <= pixel_y < 600:
    #                 grid_colors[i, j] = screen.get_at((pixel_x, pixel_y))[:3]  # Get RGB values
        
    #     # Flatten the grid colors
    #     input_tensor = torch.tensor(grid_colors.flatten(), dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
    #     # Convert target direction to one-hot encoding
    #     target = torch.zeros(4)
    #     target[target_direction] = 1
    #     target = target.unsqueeze(0)
        
    #     # Forward pass
    #     output = self.nn(input_tensor)
        
    #     # Compute loss
    #     loss = self.criterion(output, target)
        
    #     # Backward pass and optimization
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
        
    #     return loss.item()

    def train(self, screen, target_direction):
        """Train the neural network."""
        # Capture the 10x10 grid of pixel colors around the player's position
        grid_size = 120
        grid_colors = np.zeros((grid_size, grid_size, 3))
        # Normalize pixel values by dividing by 255
        
        

        for i in range(grid_size):
            for j in range(grid_size):
                pixel_x = self.rect.x + i - grid_size // 2
                pixel_y = self.rect.y + j - grid_size // 2
                if 0 <= pixel_x < 800 and 0 <= pixel_y < 600:
                    grid_colors[i, j] = screen.get_at((pixel_x, pixel_y))[:3]  # Get RGB values
        
        # Flatten the grid colors
        grid_colors /= 255.0
        input_tensor = torch.tensor(grid_colors.flatten(), dtype=torch.float32).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convert target direction to one-hot encoding  
        target = torch.zeros(4).to("cuda" if torch.cuda.is_available() else "cpu")
        target[target_direction] = 1
        target = target.unsqueeze(0)
        
        # Forward pass
        output = self.nn(input_tensor)
        
        # Compute loss
        loss = self.criterion(output, target)
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def q_learning_update(self, old_state, action, reward, new_state):
        """Update the Q-table using Q-learning."""
        if 0 <= old_state[0] < 800 and 0 <= old_state[1] < 600 and \
        0 <= new_state[0] < 800 and 0 <= new_state[1] < 600:  # Prevent out-of-bounds
            old_q_value = self.q_table[old_state[0] // 10, old_state[1] // 10, action]
            max_future_q = np.max(self.q_table[new_state[0] // 10, new_state[1] // 10, :])
            new_q_value = (1 - self.learning_rate) * old_q_value + self.learning_rate * (reward + self.discount_factor * max_future_q)
            self.q_table[old_state[0] // 10, old_state[1] // 10, action] = new_q_value


    """ Save the model to a file """
    def save_model(self, path):
        torch.save(self.nn.state_dict(), path)

    def load_model_weights(self, path):
        """Load the model weights from a file."""
        self.nn.load_state_dict(torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))