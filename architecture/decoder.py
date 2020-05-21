import torch.nn as nn
import utils
import torch
from LSTMCells import LSTMCells


class Decoder(nn.Module):

    def __init__(self, hidden_size, latent_size, input_size, num_layers=2, dtype=torch.float32):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.latent_size = latent_size
        self.dtype = dtype

        self.fc = nn.Linear(latent_size, hidden_size * num_layers * 2)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.activation_fc = nn.Tanh()

        self.cells = LSTMCells(input_size, hidden_size, num_layers)
        # self.cells = nn.LSTM(input_size + latent_size, hidden_size, num_layers)
        self.cells.type(dtype)
        self.activation_rnn = nn.Tanh()

        self.fc_mapping = nn.Linear(hidden_size * 2 * num_layers, input_size)
        torch.nn.init.xavier_uniform_(self.fc_mapping.weight)
        self.activation = nn.Sigmoid()

    def forward(self, latent, seq_length, probability_red=None):
        dec_hidden = self.fc(latent)
        dec_hidden = self.activation_fc(dec_hidden)

        if len(dec_hidden.shape) == 1:
            dec_hidden = dec_hidden.view(1, -1)

        if len(latent.shape) == 1:
            latent = latent.view(1, -1)

        batch_size = dec_hidden.shape[0]
        dec_hidden = dec_hidden.view(self.num_layers, batch_size, -1)

        dec_hidden = torch.chunk(dec_hidden, 2, len(dec_hidden.shape) - 1)
        dec_hidden = [state.contiguous() for state in dec_hidden]
        dec_hidden = tuple(dec_hidden)


        outs = torch.zeros((seq_length, batch_size, self.input_size), dtype=self.dtype).cuda()
        out = torch.zeros((batch_size, self.input_size), dtype=self.dtype).cuda()

        for i in range(0, seq_length):
            fc_input = torch.cat(dec_hidden, len(dec_hidden[0].shape) - 1).view(batch_size, - 1)
            out = self.activation(self.fc_mapping(fc_input))

            if probability_red is not None:
                out -= probability_red

            outs[i, :, :] = out.clone()
            dec_hidden = self.cells(out, dec_hidden)
            dec_hidden = (self.activation_rnn(dec_hidden[0]), self.activation_fc(dec_hidden[1]))

        return outs

    '''def __init__(self, hidden_size, latent_size, input_size, num_layers=2, dtype=torch.float32):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = latent_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.dtype = dtype

        self.fc_latent = nn.Linear(latent_size, latent_size)
        self.activation_fc = nn.SELU()

        self.fc_latent1 = nn.Linear(latent_size, latent_size)
        self.activation_fc1 = nn.SELU()

        init_hidden = nn.init.xavier_normal_(torch.zeros(num_layers, 1, hidden_size).cuda())
        init_state = nn.init.xavier_normal_(torch.zeros(num_layers, 1, hidden_size).cuda())
        self.init_hidden = nn.Parameter(init_hidden, requires_grad=True)
        self.init_state = nn.Parameter(init_state, requires_grad=True)

        self.cells = LSTMCells(latent_size, hidden_size, num_layers)
        self.cells.type(dtype)

        self.fc_mapping = nn.Linear(hidden_size * num_layers * 2, input_size)
        nn.init.xavier_normal_(self.fc_mapping.weight)
        self.activation = nn.Sigmoid()

    def forward(self, latent, seq_length):
        latent = self.fc_latent(latent)
        latent = self.activation_fc(latent)

        latent = self.fc_latent1(latent)
        latent = self.activation_fc1(latent)

        if len(latent.shape) == 1:
            latent = latent.reshape((1, latent.shape[0]))
        latent = latent.repeat(seq_length, 1, 1)

        batch_size = latent.shape[1]
        outs = torch.zeros((seq_length, batch_size, self.input_size)).type(self.dtype).cuda()
        hidden = (self.init_hidden.repeat(1, batch_size, 1),
                  self.init_state.repeat(1, batch_size, 1))

        for t in range(0, seq_length):
            hidden = self.cells(latent[t, :, :], hidden)
            fc_input = torch.cat(hidden, len(hidden[0].shape) - 1)
            fc_input = fc_input.view(batch_size, -1)
            outs[t, :, :] = self.activation(self.fc_mapping(fc_input)).squeeze()
        return outs'''
