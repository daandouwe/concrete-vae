#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.nn import functional as F


class BinaryConcrete:
    def __init__(self, alpha, temp):
        self.alpha = alpha
        self.gumbel = torch.distributions.Gumbel(
            torch.zeros(self.alpha.shape), torch.ones(self.alpha.shape))
        self.temp = temp
        self.sigmoid = nn.Sigmoid()

    def sample(self):
        return self.sigmoid((self.alpha + self.gumbel.sample()) / self.temp)


class Concrete:
    def __init__(self, alpha, temp):
        self.alpha = alpha
        self.gumbel = torch.distributions.Gumbel(
            torch.zeros(self.alpha.shape), torch.ones(self.alpha.shape))
        self.temp = temp
        self.softmax = nn.Softmax(dim=-1)

    def sample(self):
        return self.softmax((self.alpha + self.gumbel.sample()) / self.temp)


class Generative(nn.Module):
    def __init__(self, input=10, hidden=200, output=784):
        """Generative model for VAE."""
        super(Generative, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, output)
        self.act_fn = nn.Tanh()

    def forward(self, x):
        h = self.fc1(x)
        h = self.act_fn(h)
        return self.fc2(h)


class Inference(nn.Module):
    def __init__(self, input=784, hidden=200, output=10):
        super(Inference, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, output)
        self.act_fn = nn.Tanh()

    def forward(self, x):
        h = self.fc1(x)
        h = self.act_fn(h)
        return self.fc2(h)


class ConcreteVAE(nn.Module):
    def __init__(self, latent_dim):
        super(ConcreteVAE, self).__init__()
        self.generative = Generative(input=latent_dim)
        self.inference = Inference(output=latent_dim)
        self.initialize()

        self.softmax = nn.Softmax(dim=-1)

    def initialize(self):
        for param in self.parameters():
            if len(param.shape) > 2:
                nn.init.xavier_uniform_(param)

    def encode(self, x):
        return self.inference(x)

    def sample(self, alpha, temp=None):
        if self.training:
            # return BinaryConcrete(alpha, temp).sample()
            return Concrete(alpha, temp).sample()
        else:
            # return (alpha > 0.5).float()  # argmax for binary variable
            return torch.distributions.OneHotCategorical(logits=alpha).sample()

    def decode(self, x):
        h = self.generative(x)
        return torch.sigmoid(h)

    def forward(self, x, temp=None):
        alpha = self.encode(x)
        sample = self.sample(alpha, temp)
        x_ = self.decode(sample)
        return x_, alpha

    def loss(self, x, x_probs, alpha):
        bce = F.binary_cross_entropy(x_probs, x.view(-1, 784), reduction='sum')
        kl = (self.softmax(alpha) *
                # (alpha - torch.log(torch.tensor(1.) / 2))).sum()
                (alpha - torch.log(torch.tensor(1.) / alpha.size(-1)))).sum()
        return bce, kl
