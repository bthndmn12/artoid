import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
syy = sys.path.append("../")
from landaunet.landaulayer import LandauLayer
from landaunet.landaulayer import LangevinLandauOptimizer

"""This class is a Simple Neural Network for the Player class in the game. 
It has 2 hidden layers with 64 neurons each and uses ReLU activation function. 
The output layer has 4 neurons, one for each direction the player can move in. 
The network is trained using the Adam optimizer and Mean Squared Error loss function. 
The input to the network is the player's position and the food's position, and the 
output is the direction the player should move in to reach the food. 

Player sees the 10 pixels around it and the food's position to decide the direction to move in."""

class ExperimentalNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, beta_init):
        super(ExperimentalNet, self).__init__()
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            self.layers.append(LandauLayer(sizes[i], sizes[i+1], beta_init))
            if i < len(sizes) - 2:  # No BatchNorm for the last layer
                self.bn_layers.append(nn.BatchNorm1d(sizes[i+1]))

    def forward(self, x):
        for i, (layer, bn) in enumerate(zip(self.layers[:-1], self.bn_layers)):
            x = F.leaky_relu(bn(layer(x)), 0.2)
        x = self.layers[-1](x)
        return x

    def update_beta(self, x, target, dt):
        for layer in self.layers:
            layer.update_beta(x, target, dt)
            x = layer(x)

class PlayerNN:

    def __init__(self, input_size, hidden_sizes, output_size, beta_init):
        self.nn = ExperimentalNet(input_size, hidden_sizes, output_size, beta_init)
        
        self.optimizer = LangevinLandauOptimizer(self.nn.parameters())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn.to(self.device)


# class PlayerNN(nn.Module):
#     def __init__(self):
#         super(PlayerNN, self).__init__()
#         self.fc1 = nn.Linear(120*120*3, 32)  # Input size: 10x10x3 (RGB) + 2 (player and food positions)
#         self.fc2 = nn.Linear(32,32)
#         self.fc3 = nn.Linear(32, 4)  # Output size: 4 (directions: left, right, up, down)


#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)
