import ans
import midi_utils
import os
import torch
import torch.distributions

PIANO_ROLL_SIZE = 96 * 60


def reparametrize(mean, sigma):
    eps = torch.rand_like(sigma)
    return mean + torch.exp(sigma) * eps


