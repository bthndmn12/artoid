import pygame
import random

# Enum for food types
class FoodType:
    NORMAL = 0
    SPECIAL = 1
    POISON = 2


class Food(pygame.sprite.Sprite):
    def __init__(self, width, height, speed=0):
        super().__init__()
        
        # Create a food image (you can replace this with your own image)
        self.image = pygame.Surface((width, height))
        self.image.fill((255, 0, 0))  # Red color for food
        
        # Get a rectangle that represents the dimensions of the image
        self.rect = self.image.get_rect()
        
        # Randomly position the food on the screen with random FoodTypes
        self.rect.x = random.randint(0, 800 - self.rect.width)
        self.rect.y = random.randint(0, 600 - self.rect.height)
        self.food_type = random.choice([FoodType.NORMAL, FoodType.SPECIAL, FoodType.POISON])
        
        # Food speed (used for future movement or behavior)
        self.speed = speed

        # Randomly position the food on the screen
        self.respawn()

    """ Spawn a new food item at a random location. """
    def respawn(self):
        self.rect.x = random.randint(0, 800 - self.rect.width)
        self.rect.y = random.randint(0, 600 - self.rect.height)

    """ Check if the player has collided with the food. """
    def check_collision(self, player):
        return self.rect.colliderect(player.rect)
    
    """ If the food is edible, return True. If food is poisonous, return False. """
    def is_edible(self):
        return self.food_type in [FoodType.NORMAL, FoodType.SPECIAL]
    
    """ Update food position if movement is needed. """
    def update(self):
        # Move the food vertically (for example, could be extended)
        self.rect.y += self.speed
        if self.rect.y > 600:  # Wrap around the screen
            self.rect.y = 0


class FoodGroup:
    def __init__(self, num_foods, width, height, speed=0):
        self.num_foods = num_foods
        self.width = width
        self.height = height
        self.speed = speed
        self.foods = [Food(width, height, speed) for _ in range(num_foods)]
    
    """ Respawn all food items. """
    def respawn_all(self):
        for food in self.foods:
            food.respawn()
    
    """ Draw all food items on the screen. """
    def draw(self, screen):
        for food in self.foods:
            screen.blit(food.image, food.rect)

    """ Check if the player collides with any food item. """
    def check_collision(self, player_rect):
        for food in self.foods:
            if player_rect.colliderect(food.rect):
                food.respawn()
                return True
        return False

    """ Update food items (e.g., move them if needed). """
    def update(self):
        for food in self.foods:
            food.update()

    """ Set new parameters for food count and speed. """
    def set_params(self, num_food, food_speed):
        # Update the number of foods
        self.num_foods = num_food
        self.speed = food_speed

        # Adjust the list of foods
        if len(self.foods) > num_food:
            # If there are more food items than needed, remove the extra
            self.foods = self.foods[:num_food]
        else:
            # Add more food items if needed
            for _ in range(num_food - len(self.foods)):
                self.foods.append(Food(self.width, self.height, self.speed))

        # Update speed for all foods
        for food in self.foods:
            food.speed = food_speed
