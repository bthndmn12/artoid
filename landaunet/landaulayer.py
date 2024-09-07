import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
import math

class LandauLayer(nn.Module):
    def __init__(self, input_size, output_size, beta_init, alpha=0.75, gamma=0.25):
        super(LandauLayer, self).__init__()
        self.w = nn.Parameter(torch.randn(output_size, input_size))
        nn.init.xavier_normal_(self.w)
        self.beta = nn.Parameter(torch.tensor(beta_init).float())
        self.alpha = alpha
        self.gamma = gamma
        self.t = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        t_dynamic = self.t * torch.sigmoid(torch.mean(x, dim=-1)).unsqueeze(-1)
        w2 = self.w * (torch.arange(x.size(-1), dtype=torch.float, device=x.device) * torch.cosh(5 * self.beta) + t_dynamic * torch.sinh(5 * self.beta)).unsqueeze(1)
        phi = torch.matmul(w2, x.unsqueeze(-1)).squeeze(-1)
        return phi
        
    def update_beta(self, phi, target, dt, kappa=0.01):
        d_loss = self.d_loss(phi, target).detach()
        d_loss_x = (torch.roll(d_loss, -1) - torch.roll(d_loss, 1)) / 2.0
        force = -kappa * d_loss_x
        
        with torch.no_grad():
            beta_xx = (torch.roll(self.beta, -1) - 2 * self.beta + torch.roll(self.beta, 1))
            beta_t = (self.beta - self.beta.clone().detach()) / dt
            self.beta.data += dt * (beta_t - beta_xx + self.beta - self.beta**3 + force)
            
    def d_loss(self, phi, target):
        max_abs_phi = torch.max(torch.abs(phi), dim=1, keepdim=True)[0]
        phi_norm = phi / (max_abs_phi + 1e-8)  # Added small epsilon to prevent division by zero
        
        phi_x = (torch.roll(phi_norm, -1, dims=1) - torch.roll(phi_norm, 1, dims=1)) / 2.0
        
        diff = phi_norm - target
        quadratic_term = self.alpha * diff**2
        quartic_term = self.gamma * diff**4
        spatial_term = 0.5 * phi_x**2
        d_loss = quadratic_term - quartic_term + spatial_term
        mean_d_loss = d_loss.mean()
        return torch.abs(mean_d_loss)

class LangevinLandauOptimizer:
    def __init__(self, params, lr=0.0002, damping=0.03, temperature=0.07):
        self.params = list(params)
        self.lr = lr
        self.damping = damping
        self.temperature = temperature
        self.velocities = [torch.zeros_like(p.data) for p in self.params]

    def step(self):
        for param, velocity in zip(self.params, self.velocities):
            if param.grad is None:
                continue
            
            force = -param.grad.data

            velocity = velocity.to(param.device)
            
            velocity.mul_(1 - self.damping).add_(force, alpha=self.lr)
            noise_scale = math.sqrt(2 * self.damping * self.temperature * self.lr)
            velocity.add_(torch.randn_like(velocity) * noise_scale)
            
            param.data.add_(velocity)

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

