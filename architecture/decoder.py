import torch.nn as nn
import utils
import torch
from LSTMCells import LSTMCells


class Decoder(nn.Module):

    def __init__(self, hidden_size, latent_size, input_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        self.fc = nn.Linear(latent_size, hidden_size * num_layers * 2)
        self.activation_fc = nn.ReLU()

        self.cells = LSTMCells(input_size, hidden_size, num_layers)

        self.fc_mapping = nn.Linear(hidden_size * num_layers * 2, input_size)
        self.activation = nn.Sigmoid()

    def forward(self, latent, seq_length):
        dec_hidden = self.fc(latent)
        dec_hidden = torch.reshape(dec_hidden, (self.num_layers, dec_hidden.shape[0], -1))

        dec_hidden = torch.chunk(dec_hidden, 2, 2)
        dec_hidden = [state.contiguous() for state in dec_hidden]
        dec_hidden = tuple(dec_hidden)

        outs = torch.zeros((seq_length, latent.shape[0], self.input_size)).cuda()
        for t in range(0, seq_length):
            # batch, input
            fc_input = torch.cat(dec_hidden, 2).reshape((dec_hidden[0].shape[1], -1))
            outs[t, :, :] = self.activation(self.fc_mapping(fc_input)).squeeze().clone()
            dec_hidden = self.cells(outs[t, :, :].clone(), dec_hidden)
        return outs
