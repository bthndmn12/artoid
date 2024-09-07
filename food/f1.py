import pygame
import random

# Enum for food types
class FoodType:
    NORMAL = 0
    SPECIAL = 1
    POISON = 2


class Food(pygame.sprite.Sprite):
    def __init__(self, width, height):
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
        if self.food_type == FoodType.NORMAL or self.food_type == FoodType.SPECIAL:
            return True
        else:
            return False
        
class FoodGroup:
    def __init__(self, num_foods, width, height):
        self.foods = [Food(width, height) for _ in range(num_foods)]
    
    def respawn_all(self):
        for food in self.foods:
            food.respawn()
    
    def draw(self, screen):
        for food in self.foods:
            screen.blit(food.image, food.rect)
    
    def check_collision(self, player_rect):
        for food in self.foods:
            if player_rect.colliderect(food.rect):
                food.respawn()
                return True
        return False
    
