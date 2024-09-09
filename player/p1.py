import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from l_brain.p1_n import DuelingDQN
from collections import deque, namedtuple
import random

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'priority'])

class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
    
    def add(self, state, action, reward, next_state):
        max_priority = max(self.priorities) if self.priorities else 1.0
        experience = Experience(state, action, reward, next_state, max_priority)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max_priority)
        else:
            idx = random.randint(0, self.capacity - 1)
            self.buffer[idx] = experience
            self.priorities[idx] = max_priority
    
    def sample(self, batch_size):
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        return samples, indices
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


class Player(pygame.sprite.Sprite):
    def __init__(self, x, y, image_path, width, height, food_group, points=0, load_weights=True):
        super().__init__()

        self.image = pygame.Surface((width, height))
        self.image.fill((255, 255, 0))  # Yellow for the player

        self.rect = self.image.get_rect()
        self.rect.x, self.rect.y = x, y

        self.velocity = [0, 0]
        self.points = points
        self.food_group = food_group

        self.nn = DuelingDQN().to("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.AdamW(self.nn.parameters(), lr=0.0001)
        self.criterion = torch.nn.MSELoss()

        if load_weights:
            self.load_model_weights("player_model.pth")

        # Deep Q-learning parameters
        self.batch_size = 64
        self.replay_memory = PrioritizedReplayBuffer(capacity=10000)  # Using PrioritizedReplayBuffer
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.target_update_frequency = 100
        self.current_step = 0

        self.target_nn = DuelingDQN().to("cuda" if torch.cuda.is_available() else "cpu")
        self.target_nn.load_state_dict(self.nn.state_dict())
        self.target_nn.eval()

        self.current_loss = None

    def move(self, dx, dy):
        """Move the player by changing its rect's position."""
        new_rect = pygame.Rect(self.rect.x + dx * 5, self.rect.y + dy * 5, self.rect.width, self.rect.height)

        if new_rect.x < 0:
            new_rect.x = 0
        elif new_rect.x + new_rect.width > 800:
            new_rect.x = 800 - new_rect.width

        if new_rect.y < 0:
            new_rect.y = 0
        elif new_rect.y + new_rect.height > 600:
            new_rect.y = 600 - new_rect.height

        self.rect = new_rect

    def get_state(self, screen):
        """Capture and process the game state as a 64x64 tensor."""
        # screen_array = pygame.surfarray.array3d(screen)
        resized_screen = pygame.transform.scale(screen, (64, 64))
        resized_array = pygame.surfarray.array3d(resized_screen)
        resized_array = resized_array.transpose((2, 0, 1)).astype(np.float32) / 255.0

        screen_tensor = torch.tensor(resized_array, dtype=torch.float32).unsqueeze(0)
        screen_tensor = screen_tensor.to("cuda" if torch.cuda.is_available() else "cpu")
        return screen_tensor

    def get_reward(self):
        """Calculate the reward for the player's actions."""
        reward = -0.01  # Small negative reward for each step
        if self.eat():
            reward += 10  # Positive reward for eating food
        elif self.check_collision_with_walls():
            reward -= 5  # Negative reward for hitting walls

        closest_food = min(self.food_group.foods, key=lambda food: ((food.rect.x - self.rect.x)**2 + (food.rect.y - self.rect.y)**2)**0.5)
        distance = ((closest_food.rect.x - self.rect.x)**2 + (closest_food.rect.y - self.rect.y)**2)**0.5
        reward += 5 / (distance + 1)  # Distance-based reward

        return reward

    def update(self, screen):
        old_state = self.get_state(screen)

        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, 4)
        else:
            with torch.no_grad():
                output = self.nn(old_state)
                action = torch.argmax(output).item()

        # Perform the chosen action
        if action == 0:
            self.move(-1, 0)
        elif action == 1:
            self.move(1, 0)
        elif action == 2:
            self.move(0, -1)
        elif action == 3:
            self.move(0, 1)

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.current_step += 1

        # Calculate reward and new state
        reward = self.get_reward()
        new_state = self.get_state(screen)

        # Store experience in replay memory
        self.replay_memory.add(old_state, action, reward, new_state)

        # Train the model if the replay memory has enough samples
        if len(self.replay_memory.buffer) > self.batch_size and self.current_step % 4 == 0:
            self.current_loss = self.train_double_dqn()  # Capture loss after training step

        # Update target network every few steps
        if self.current_step % self.target_update_frequency == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())

    def train_double_dqn(self):
        if len(self.replay_memory.buffer) < self.batch_size:
            return None

        experiences, indices = self.replay_memory.sample(self.batch_size)
        states = torch.cat([exp.state for exp in experiences])
        actions = torch.tensor([exp.action for exp in experiences], device=states.device)
        rewards = torch.tensor([exp.reward for exp in experiences], device=states.device)
        next_states = torch.cat([exp.next_state for exp in experiences])

        # Get current Q-values
        current_q_values = self.nn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get target Q-values
        with torch.no_grad():
            next_action_values = self.nn(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_nn(next_states).gather(1, next_action_values).squeeze(1)
            target_q_values = rewards + self.gamma * next_q_values

        # Compute the loss
        loss = self.criterion(current_q_values, target_q_values)

        # Prioritized experience replay: update priorities
        td_errors = abs(current_q_values - target_q_values).detach().cpu().numpy()
        new_priorities = td_errors + 1e-6  # Small constant to avoid zero priority
        self.replay_memory.update_priorities(indices, new_priorities)

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.nn.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()  # Return the loss value for logging

    def eat(self):
        """Check if the player collides with the food group and update points."""
        if self.food_group.check_collision(self.rect):
            self.points += 1
            return True
        return False

    def check_collision_with_walls(self):
        """Check if the player is colliding with the screen boundaries."""
        return (self.rect.left < 0 or self.rect.right > 800 or
                self.rect.top < 0 or self.rect.bottom > 600)

    def save_model(self, path):
        """Save the model to a file."""
        torch.save(self.nn.state_dict(), path)

    def load_model_weights(self, path):
        """Load the model weights from a file."""
        self.nn.load_state_dict(torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

#---------------------------------------------------------------------------------------------------------------------
# from collections import namedtuple
# import random

# Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'priority'])

# class PrioritizedReplayBuffer:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.buffer = []
#         self.priorities = []
    
#     def add(self, state, action, reward, next_state):
#         max_priority = max(self.priorities) if self.priorities else 1.0
#         experience = Experience(state, action, reward, next_state, max_priority)
#         if len(self.buffer) < self.capacity:
#             self.buffer.append(experience)
#             self.priorities.append(max_priority)
#         else:
#             idx = random.randint(0, self.capacity - 1)
#             self.buffer[idx] = experience
#             self.priorities[idx] = max_priority
    
#     def sample(self, batch_size):
#         probs = np.array(self.priorities) / sum(self.priorities)
#         indices = np.random.choice(len(self.buffer), batch_size, p=probs)
#         samples = [self.buffer[idx] for idx in indices]
#         return samples, indices
    
#     def update_priorities(self, indices, priorities):
#         for idx, priority in zip(indices, priorities):
#             self.priorities[idx] = priority


# class Player(pygame.sprite.Sprite):
#     def __init__(self, x, y, image_path, width, height, food_group, points=0, load_weights=True):
#         super().__init__()

#         # Create a food image (you can replace this with your own image)
#         self.image = pygame.Surface((width, height))
#         self.image.fill((255, 255, 0))  # Red color for food

#         # Get a rectangle that represents the dimensions of the image
#         self.rect = self.image.get_rect()
#         self.rect.x, self.rect.y = x, y

#         self.velocity = [0, 0]
#         self.points = points
#         self.food_group = food_group

#         # Initialize the neural network
#         self.nn = PlayerNN().to("cuda" if torch.cuda.is_available() else "cpu")
#         # self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=0.001)
#         self.optimizer = torch.optim.AdamW(self.nn.parameters(), lr=0.001)
#         # self.optimizer = torch.optim(self.nn.parameters(), lr=0.001)

#         self.criterion = torch.nn.MSELoss()
#         # self.criterion = torch.nn.SmoothL1Loss()

#         # Load model weights if specified
#         if load_weights:
#             self.load_model_weights("player_model.pth")

#         # Deep Q-learning parameters
#         self.batch_size = 64
#         self.replay_memory = deque(maxlen=10000)  # Experience replay buffer
#         self.gamma = 0.99  # Discount factor
#         self.epsilon = 1.0  # Exploration rate (start with full exploration)
#         self.epsilon_min = 0.01  # Minimum exploration rate
#         self.epsilon_decay = 0.995  # Decay rate for exploration
#         self.target_update_frequency = 5  # Update target network every 10 steps
#         self.current_step = 0

#         # Create target network (a copy of the main network)
#         self.target_nn = PlayerNN().to("cuda" if torch.cuda.is_available() else "cpu")
#         self.target_nn.load_state_dict(self.nn.state_dict())  # Initialize with the same weights
#         self.target_nn.eval()  # Set target network to evaluation mode

#         # Loss and current loss for logging
#         self.current_loss = None

#     def move(self, dx, dy):
#         """Move the player by changing its rect's position."""
#         new_rect = pygame.Rect(self.rect.x + dx * 5, self.rect.y + dy * 5, self.rect.width, self.rect.height)

#         # Check bounds to prevent player from moving off-screen
#         if new_rect.x < 0:
#             new_rect.x = 0
#         elif new_rect.x + new_rect.width > 800:
#             new_rect.x = 800 - new_rect.width

#         if new_rect.y < 0:
#             new_rect.y = 0
#         elif new_rect.y + new_rect.height > 600:
#             new_rect.y = 600 - new_rect.height

#         self.rect = new_rect

#     def get_state(self, screen):
#         """Capture and process the game state as a 64x64 tensor."""
#         screen_array = pygame.surfarray.array3d(screen)
        
#         # Resize the screen to 64x64
#         resized_screen = pygame.transform.scale(screen, (64, 64))
#         resized_array = pygame.surfarray.array3d(resized_screen)
        
#         # Convert to (C, H, W) format and normalize
#         resized_array = resized_array.transpose((2, 0, 1)).astype(np.float32) / 255.0
        
#         screen_tensor = torch.tensor(resized_array, dtype=torch.float32).unsqueeze(0)
#         screen_tensor = screen_tensor.to("cuda" if torch.cuda.is_available() else "cpu")
#         return screen_tensor
    
#     def update(self, screen):
#         old_state = self.get_state(screen)

#         if np.random.rand() < self.epsilon:
#             action = np.random.randint(0, 4)
#         else:
#             with torch.no_grad():
#                 output = self.nn(old_state)
#                 action = torch.argmax(output).item()

#         if action == 0:
#             self.move(-1, 0)
#         elif action == 1:
#             self.move(1, 0)
#         elif action == 2:
#             self.move(0, -1)
#         elif action == 3:
#             self.move(0, 1)

#         self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
#         self.current_step += 1

#         reward = -0.1
#         if self.eat():
#             print("Player1 ate the food!")
#             self.food_group.respawn_all()
#             reward = 1
#         elif self.check_collision_with_walls():
#             reward = -0.5

#         new_state = self.get_state(screen)
#         # self.replay_memory.append((old_state, action, reward, new_state))
        

#         if len(self.replay_memory) > self.batch_size and self.current_step % 4 == 0:
#             self.train_dqn()

#         if self.current_step % self.target_update_frequency == 0:
#             self.target_nn.load_state_dict(self.nn.state_dict())

#     def eat(self):
#         """Check if the player collides with the foodgroup and update points."""
#         if self.food_group.check_collision(self.rect):
#             self.points += 1
#             return True
#         return False

#     def check_collision_with_walls(self):
#         """Check if the player is colliding with the screen boundaries."""
#         return (self.rect.left < 0 or self.rect.right > 800 or
#                 self.rect.top < 0 or self.rect.bottom > 600)

#     def train_double_dqn(self):
#         if len(self.replay_buffer) < self.batch_size:
#             return

#         experiences, indices = self.replay_buffer.sample(self.batch_size)
#         states = torch.cat([exp.state for exp in experiences])
#         actions = torch.tensor([exp.action for exp in experiences], device=states.device)
#         rewards = torch.tensor([exp.reward for exp in experiences], device=states.device)
#         next_states = torch.cat([exp.next_state for exp in experiences])

#         current_q_values = self.nn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
#         with torch.no_grad():
#             next_action_values = self.nn(next_states).max(1)[1].unsqueeze(1)
#             next_q_values = self.target_nn(next_states).gather(1, next_action_values).squeeze(1)
#             target_q_values = rewards + self.gamma * next_q_values

#         loss = self.criterion(current_q_values, target_q_values)
        
#         # Update priorities
#         td_errors = abs(current_q_values - target_q_values).detach().cpu().numpy()
#         new_priorities = td_errors + 1e-6  # Small constant to avoid zero priority
#         self.replay_buffer.update_priorities(indices, new_priorities)

#         self.optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.nn.parameters(), max_norm=1.0)
#         self.optimizer.step()

#         return loss.item()
#     # def train_dqn(self):
#     #     if len(self.replay_memory) < self.batch_size:
#     #         return

#     #     minibatch = random.sample(self.replay_memory, self.batch_size)
#     #     states = torch.cat([transition[0] for transition in minibatch])
#     #     actions = torch.tensor([transition[1] for transition in minibatch], device=states.device)
#     #     rewards = torch.tensor([transition[2] for transition in minibatch], device=states.device)
#     #     next_states = torch.cat([transition[3] for transition in minibatch])

#     #     q_values = self.nn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
#     #     with torch.no_grad():
#     #         next_q_values = self.target_nn(next_states).max(1)[0]
#     #         target_q_values = rewards + self.gamma * next_q_values

#     #     loss = self.criterion(q_values, target_q_values)
#     #     self.current_loss = loss.item()

#     #     self.optimizer.zero_grad()
#     #     loss.backward()
#     #     torch.nn.utils.clip_grad_norm_(self.nn.parameters(), max_norm=1.0)  # Add gradient clipping
#     #     self.optimizer.step()

#     """ Save the model to a file """
#     def save_model(self, path):
#         torch.save(self.nn.state_dict(), path)

#     def load_model_weights(self, path):
#         """Load the model weights from a file."""
#         self.nn.load_state_dict(torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

#-------------------------------------------------------------------------
# class Player(pygame.sprite.Sprite):
#     def __init__(self, x, y, image_path, width, height, food_group, points=0, load_weights=True):
#         super().__init__()

#        # Create a food image (you can replace this with your own image)
#         self.image = pygame.Surface((width, height))
#         self.image.fill((255, 255, 0))  # Red color for food

#         # Get a rectangle that represents the dimensions of the image
#         self.rect = self.image.get_rect()
#         self.rect.x, self.rect.y = x, y

#         self.velocity = [0, 0]
#         self.points = points
#         self.food_group = food_group

#         # Initialize the neural network
#         self.nn = PlayerNN().to("cuda" if torch.cuda.is_available() else "cpu")
#         self.optimizer = torch.optim.AdamW(self.nn.parameters(), lr=0.001)
#         self.criterion = torch.nn.MSELoss()

#         # Load model weights if specified
#         if load_weights:
#             self.load_model_weights("player_model.pth")

#         # Deep Q-learning parameters
#         self.batch_size = 8
#         self.replay_memory = deque(maxlen=100)  # Experience replay buffer
#         self.gamma = 0.99  # Discount factor
#         self.epsilon = 1.0  # Exploration rate (start with full exploration)
#         self.epsilon_min = 0.01  # Minimum exploration rate
#         self.epsilon_decay = 0.995  # Decay rate for exploration
#         self.target_update_frequency = 10  # Update target network every 10 steps
#         self.current_step = 0

#         # Create target network (a copy of the main network)
#         self.target_nn = PlayerNN().to("cuda" if torch.cuda.is_available() else "cpu")
#         self.target_nn.load_state_dict(self.nn.state_dict())  # Initialize with the same weights
#         self.target_nn.eval()  # Set target network to evaluation mode

#         # Loss and current loss for logging
#         self.current_loss = None

#     def move(self, dx, dy):
#         """Move the player by changing its rect's position."""
#         new_rect = pygame.Rect(self.rect.x + dx * 5, self.rect.y + dy * 5, self.rect.width, self.rect.height)

#         # Check bounds to prevent player from moving off-screen
#         if new_rect.x < 0:
#             new_rect.x = 0
#         elif new_rect.x + new_rect.width > 800:
#             new_rect.x = 800 - new_rect.width

#         if new_rect.y < 0:
#             new_rect.y = 0
#         elif new_rect.y + new_rect.height > 600:
#             new_rect.y = 600 - new_rect.height

#         self.rect = new_rect


#     def get_state(self, screen):
#         """Capture and process the game state as a 64x64 tensor."""
#         screen_array = pygame.surfarray.array3d(screen)
        
#         # Resize the screen to 64x64
#         resized_screen = pygame.transform.scale(screen, (64, 64))
#         resized_array = pygame.surfarray.array3d(resized_screen)
        
#         # Convert to (C, H, W) format and normalize
#         resized_array = resized_array.transpose((2, 0, 1)).astype(np.float32) / 255.0
        
#         screen_tensor = torch.tensor(resized_array, dtype=torch.float32).unsqueeze(0)
#         screen_tensor = screen_tensor.to("cuda" if torch.cuda.is_available() else "cpu")
#         return screen_tensor
    
#     def update(self, screen):
#         old_state = self.get_state(screen)

#         if np.random.rand() < self.epsilon:
#             action = np.random.randint(0, 4)
#         else:
#             with torch.no_grad():
#                 output = self.nn(old_state)
#                 action = torch.argmax(output).item()

#         if action == 0:
#             self.move(-1, 0)
#         elif action == 1:
#             self.move(1, 0)
#         elif action == 2:
#             self.move(0, -1)
#         elif action == 3:
#             self.move(0, 1)

#         self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
#         self.current_step += 1

#         reward = -0.1
#         if self.eat():
#             print("Player1 ate the food!")
#             self.food_group.respawn_all()
#             reward = 1
#         elif self.check_collision_with_walls():
#             reward = -0.5

#         new_state = self.get_state(screen)
#         self.replay_memory.append((old_state, action, reward, new_state))

#         if len(self.replay_memory) > self.batch_size and self.current_step % 4 == 0:
#             self.train_dqn()

#         if self.current_step % self.target_update_frequency == 0:
#             self.target_nn.load_state_dict(self.nn.state_dict())

#     # def get_state(self, screen):
#     #     """Capture and process the game state as a tensor."""
#     #     grid_size = 120
#     #     grid_colors = np.zeros((grid_size, grid_size, 3))
#     #     for i in range(grid_size):
#     #         for j in range(grid_size):
#     #             pixel_x = self.rect.x + i - grid_size // 2
#     #             pixel_y = self.rect.y + j - grid_size // 2
#     #             if 0 <= pixel_x < 800 and 0 <= pixel_y < 600:
#     #                 grid_colors[i, j] = screen.get_at((pixel_x, pixel_y))[:3]

#     #     # Normalize the grid colors
#     #     grid_colors /= 255.0

#     #     # Reshape the grid colors to match the expected input shape for the neural network
#     #     input_tensor = torch.tensor(grid_colors, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
#     #     return input_tensor


#     def eat(self):
#         """Check if the player collides with the foodgroup and update points."""
#         if self.food_group.check_collision(self.rect):
#             self.points += 1
#             return True
#         return False

#     def check_collision_with_walls(self):
#         """Check if the player is colliding with the screen boundaries."""
#         return (self.rect.left < 0 or self.rect.right > 800 or
#                 self.rect.top < 0 or self.rect.bottom > 600)

#     def train_dqn(self):
#         if len(self.replay_memory) < self.batch_size:
#             return

#         minibatch = random.sample(self.replay_memory, self.batch_size)
#         states = torch.cat([transition[0] for transition in minibatch])
#         actions = torch.tensor([transition[1] for transition in minibatch], device=states.device)
#         rewards = torch.tensor([transition[2] for transition in minibatch], device=states.device)
#         next_states = torch.cat([transition[3] for transition in minibatch])

#         q_values = self.nn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
#         with torch.no_grad():
#             next_q_values = self.target_nn(next_states).max(1)[0]
#             target_q_values = rewards + self.gamma * next_q_values

#         loss = self.criterion(q_values, target_q_values)
#         self.current_loss = loss.item()

#         self.optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.nn.parameters(), max_norm=1.0)  # Add gradient clipping
#         self.optimizer.step()

#     """ Save the model to a file """
#     def save_model(self, path):
#         torch.save(self.nn.state_dict(), path)

#     def load_model_weights(self, path):
#         """Load the model weights from a file."""
#         self.nn.load_state_dict(torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

#---------------------------------------------------------------------------------------------------------
# import pygame
# import torch
# import numpy as np
# from l_brain.p1_n import PlayerNN

# class Player(pygame.sprite.Sprite):
#     def __init__(self, x, y, image_path, width, height, food_group, points=0, load_weights=True):
#         super().__init__()
        
#         # Load the player's sprite from an image file and resize it
#         original_image = pygame.image.load(image_path)
#         self.image = pygame.transform.scale(original_image, (width, height))
        
#         # Get a rectangle that represents the dimensions of the image
#         self.rect = self.image.get_rect()
#         self.rect.x, self.rect.y = x, y
        
#         self.velocity = [0, 0]
#         self.points = points
#         self.food_group = food_group

#         # Initialize the neural network
#         self.nn = PlayerNN().to("cuda" if torch.cuda.is_available() else "cpu")
#         self.optimizer = torch.optim.AdamW(self.nn.parameters(), lr=0.001)
#         self.criterion = torch.nn.MSELoss()
#         # self.criterion = torch.nn.CrossEntropyLoss()
#         # self.criterion = torch.nn.BCEWithLogitsLoss()
#         # self.criterion = torch.nn.MSELoss()

#         # Load model weights if specified
#         if load_weights:
#             self.load_model_weights("player_model.pth")


#         # Q-learning parameters
#         self.q_table = np.zeros((800, 600, 4))  # Q-table for 800x600 grid and 4 actions
#         self.learning_rate = 0.1
#         self.discount_factor = 0.95
#         self.epsilon = 0.1

#         self.epsilon_min = 0.01  # Minimum exploration rate
#         self.epsilon_decay = 0.995  # Decay rate for exploration

#         # # Font for rendering points
#         # self.font = pygame.font.Font(None, 36)
#         # self.update_text()
    
    
#     def move(self, dx, dy):
#         """Move the player by changing its rect's position."""
#         new_rect = pygame.Rect(self.rect.x + dx * 5, self.rect.y + dy * 5, self.rect.width, self.rect.height)
        
#         # Check bounds to prevent player from moving off-screen
#         if new_rect.x < 0:
#             new_rect.x = 0
#         elif new_rect.x + new_rect.width > 800:
#             new_rect.x = 800 - new_rect.width
        
#         if new_rect.y < 0:
#             new_rect.y = 0
#         elif new_rect.y + new_rect.height > 600:
#             new_rect.y = 600 - new_rect.height
        
#         self.rect = new_rect
    
#     # def update(self):
#     #     """Update the velocity and move accordingly."""
#     #     keys = pygame.key.get_pressed()
        
#     #     if keys[pygame.K_LEFT]:
#     #         self.move(-0.5, 0)
#     #     elif keys[pygame.K_RIGHT]:
#     #         self.move(0.5, 0)
#     #     if keys[pygame.K_UP]:
#     #         self.move(0, -0.5)
#     #     elif keys[pygame.K_DOWN]:
#     #         self.move(0, 0.5)

#     #     # Press Ctrl to move faster
#     #     if keys[pygame.K_LCTRL]:
#     #         self.move(self.velocity[0] * 5, self.velocity[1] * 5)

#     def update(self, screen):
#         """Update the velocity and move accordingly."""
#         # Capture the 120x120 grid of pixel colors around the player's position
#         grid_size = 120
#         grid_colors = np.zeros((grid_size, grid_size, 3))
#         for i in range(grid_size):
#             for j in range(grid_size):
#                 pixel_x = self.rect.x + i - grid_size // 2
#                 pixel_y = self.rect.y + j - grid_size // 2
#                 if 0 <= pixel_x < 800 and 0 <= pixel_y < 600:
#                     grid_colors[i, j] = screen.get_at((pixel_x, pixel_y))[:3]  # Get RGB values

#         # Normalize the grid colors
#         grid_colors /= 255.0

#         # Reshape the grid colors to match the expected input shape for the neural network
#         input_tensor = torch.tensor(grid_colors, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

#         # Epsilon-greedy policy
#         if np.random.rand() < self.epsilon:
#             direction = np.random.randint(0, 4)  # Exploration: random direction
#         else:
#             output = self.nn(input_tensor)  # Exploitation: neural network output
#             direction = torch.argmax(output).item()  # Choose best direction based on NN

#         # Decay epsilon after each update
#         self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

#         # Move the player based on the direction
#         if direction == 0:
#             self.move(-1, 0)  # Left
#         elif direction == 1:
#             self.move(1, 0)   # Right
#         elif direction == 2:
#             self.move(0, -1)  # Up
#         elif direction == 3:
#             self.move(0, 1)   # Down

#     def eat(self):
#         """Check if the player collides with the foodgroup and update points."""
#         if self.food_group.check_collision(self.rect):
#             self.points += 1
#             # self.update_text()
#             return True
#         return False
#         # """Check if the player collides with the food and update points."""
#         # if self.rect.colliderect(self.food_group.rect):s
#         #     self.points += 1
#         #     self.update_text()
#         #     return True
#         # return False
    
#     # def update_text(self):
#     #     """Update the text surface with the current points."""
#     #     self.text = self.font.render(str(self.points), True, (255, 255, 255))
#     #     self.text_rect = self.text.get_rect()
#     #     self.text_rect.center = (self.rect.x + self.rect.width // 2, self.rect.y + self.rect.height // 2)

#     # """Train the neural network using the 10x10 grid of pixel colors around the player's position"""
#     # def train(self, screen, target_direction):
#     #     """Train the neural network."""
#     #     # Capture the 10x10 grid of pixel colors around the player's position
#     #     grid_size = 10
#     #     grid_colors = np.zeros((grid_size, grid_size, 3))
#     #     for i in range(grid_size):
#     #         for j in range(grid_size):
#     #             pixel_x = self.rect.x + i - grid_size // 2
#     #             pixel_y = self.rect.y + j - grid_size // 2
#     #             if 0 <= pixel_x < 800 and 0 <= pixel_y < 600:
#     #                 grid_colors[i, j] = screen.get_at((pixel_x, pixel_y))[:3]  # Get RGB values
        
#     #     # Flatten the grid colors
#     #     input_tensor = torch.tensor(grid_colors.flatten(), dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
#     #     # Convert target direction to one-hot encoding
#     #     target = torch.zeros(4)
#     #     target[target_direction] = 1
#     #     target = target.unsqueeze(0)
        
#     #     # Forward pass
#     #     output = self.nn(input_tensor)
        
#     #     # Compute loss
#     #     loss = self.criterion(output, target)
        
#     #     # Backward pass and optimization
#     #     self.optimizer.zero_grad()
#     #     loss.backward()
#     #     self.optimizer.step()
        
#     #     return loss.item()

#     def train(self, screen, target_direction):
#         """Train the neural network."""
#         # Capture the 120x120 grid of pixel colors around the player's position
#         grid_size = 120
#         grid_colors = np.zeros((grid_size, grid_size, 3))
#         for i in range(grid_size):
#             for j in range(grid_size):
#                 pixel_x = self.rect.x + i - grid_size // 2
#                 pixel_y = self.rect.y + j - grid_size // 2
#                 if 0 <= pixel_x < 800 and 0 <= pixel_y < 600:
#                     grid_colors[i, j] = screen.get_at((pixel_x, pixel_y))[:3]  # Get RGB values

#         # Normalize the grid colors
#         grid_colors /= 255.0

#         # Reshape the grid colors to match the expected input shape for the neural network
#         input_tensor = torch.tensor(grid_colors, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

#         # Convert target direction to one-hot encoding
#         target = torch.zeros(4).to("cuda" if torch.cuda.is_available() else "cpu")
#         target[target_direction] = 1
#         target = target.unsqueeze(0)

#         # Forward pass
#         output = self.nn(input_tensor)

#         # Compute loss
#         loss = self.criterion(output, target)

#         # Backward pass and optimization
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         return loss.item()

#     def q_learning_update(self, old_state, action, reward, new_state):
#         """Update the Q-table using Q-learning."""
#         if 0 <= old_state[0] < 800 and 0 <= old_state[1] < 600 and \
#         0 <= new_state[0] < 800 and 0 <= new_state[1] < 600:  # Prevent out-of-bounds
#             old_q_value = self.q_table[old_state[0] // 10, old_state[1] // 10, action]
#             max_future_q = np.max(self.q_table[new_state[0] // 10, new_state[1] // 10, :])
#             new_q_value = (1 - self.learning_rate) * old_q_value + self.learning_rate * (reward + self.discount_factor * max_future_q)
#             self.q_table[old_state[0] // 10, old_state[1] // 10, action] = new_q_value


#     """ Save the model to a file """
#     def save_model(self, path):
#         torch.save(self.nn.state_dict(), path)

#     def load_model_weights(self, path):
#         """Load the model weights from a file."""
#         self.nn.load_state_dict(torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))