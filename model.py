#!/usr/bin/env python
import torch
import torch.nn as nn


class ConcreteVAE(nn.Module):
    def __init__(self, hidden=400, z_dim=20):
        super(ConcreteVAE, self).__init__()

        self.fc1 = nn.Linear(784, hidden)
        self.mu = nn.Linear(hidden, z_dim)
        self.sigma = nn.Linear(hidden, z_dim)
        self.fc3 = nn.Linear(z_dim, hidden)
        self.fc4 = nn.Linear(hidden, 784)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.mu(h1), self.sigma(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
