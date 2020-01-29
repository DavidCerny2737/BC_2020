import torch.nn as nn
import utils
import torch
from LSTMCells import LSTMCells


class Decoder(nn.Module):

    '''def __init__(self, hidden_size, latent_size, input_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        self.fc = nn.Linear(latent_size, hidden_size * num_layers * 2)
        self.activation_fc = nn.ReLU()

        self.cells = LSTMCells(input_size, hidden_size, num_layers)

        self.fc_mapping = nn.Linear(hidden_size * num_layers * 2, input_size)
        self.activation = nn.Sigmoid()'''

    '''def forward(self, latent, seq_length):
        dec_hidden = self.fc(latent)
        dec_hidden = self.activation_fc(dec_hidden)
        if len(dec_hidden.shape) == 1:
            dec_hidden = torch.reshape(dec_hidden, (1, -1))
        batch_size = dec_hidden.shape[0]

        dec_hidden = torch.reshape(dec_hidden, (self.num_layers, batch_size, -1))

        dec_hidden = torch.chunk(dec_hidden, 2, len(dec_hidden.shape) - 1)
        dec_hidden = [state.contiguous() for state in dec_hidden]
        dec_hidden = tuple(dec_hidden)

        outs = torch.zeros((seq_length, batch_size, self.input_size)).cuda()

        for t in range(0, seq_length):
            # batch, input
            fc_input = torch.cat(dec_hidden, len(dec_hidden[0].shape) - 1).reshape((batch_size, -1))
            outs[t, :, :] = self.activation(self.fc_mapping(fc_input)).squeeze().clone()
            dec_hidden = self.cells(outs[t, :, :].clone(), dec_hidden)
        return outs'''

    def __init__(self, hidden_size, latent_size, input_size, num_layers=2, dtype=torch.float32):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = latent_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.dtype = dtype

        self.cells = LSTMCells(latent_size, hidden_size, num_layers)
        self.cells.type(dtype)

        self.fc_mapping = nn.Linear(hidden_size * num_layers * 2, input_size)
        self.activation = nn.Sigmoid()

    def forward(self, latent, seq_length):
        if len(latent.shape) == 1:
            latent = latent.reshape((1, latent.shape[0]))
        latent = latent.repeat(seq_length, 1, 1)

        batch_size = latent.shape[1]
        outs = torch.zeros((seq_length, batch_size, self.input_size)).type(self.dtype).cuda()
        hidden = (torch.zeros((self.num_layers, batch_size, self.hidden_size)).type(self.dtype).cuda(),
                  torch.zeros((self.num_layers, batch_size, self.hidden_size)).type(self.dtype).cuda())
        for t in range(0, seq_length):
            hidden = self.cells(latent[t, :, :], hidden)
            fc_input = torch.cat(hidden, len(hidden[0].shape) - 1).reshape((batch_size, -1))
            outs[t, :, :] = self.activation(self.fc_mapping(fc_input)).squeeze().clone()
        return outs
