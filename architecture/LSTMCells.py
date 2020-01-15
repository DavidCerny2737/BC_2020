import torch
from torch import nn


class LSTMCells(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(0, num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, inputs, hidden):
        result = (torch.zeros_like(hidden[0]), torch.zeros_like(hidden[1]))
        for i, layer in enumerate(self.layers):
            h, c = layer(inputs, (hidden[0][i], hidden[1][i]))
            inputs = h
            result[0][i] = h
            result[1][i] = c
        return result