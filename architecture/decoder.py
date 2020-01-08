import torch.nn as nn
import utils
import torch


class Decoder(nn.Module):

    def __init__(self, hidden_size, latent_size, input_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        self.fc = nn.Linear(latent_size, hidden_size * num_layers * 2)
        self.activation_fc = nn.ReLU()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.cell = nn.LSTMCell(input_size, hidden_size)

        self.fc_mapping = nn.Linear(hidden_size * num_layers * 2, input_size)
        self.activation = nn.Sigmoid()

    def forward(self, latent, seq_length):
        dec_hidden = self.fc(latent)
        dec_hidden = torch.reshape(dec_hidden, (self.num_layers, 1, -1))
        dec_hidden = torch.chunk(dec_hidden, 2, 2)
        dec_hidden = [state.contiguous() for state in dec_hidden]
        dec_hidden = (torch.reshape(dec_hidden[0], (dec_hidden[0].shape[1], -1)),
                      torch.reshape(dec_hidden[1], (dec_hidden[1].shape[1], -1)))

        outs = torch.zeros((seq_length, latent.shape[0], self.input_size)).cuda()
        for t in range(0, seq_length):
            outs[t, 0, :] = self.activation(self.fc_mapping(torch.cat(dec_hidden, 1)).squeeze().clone())
            h_1, c_1 = self.cell(outs[t, :, :].clone(), dec_hidden)
            dec_hidden = (h_1, c_1)
        return outs
