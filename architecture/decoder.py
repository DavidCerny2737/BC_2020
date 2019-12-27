import torch.nn as nn
import utils
import torch

class Decoder(nn.Module):

    def __init__(self, hidden_size, latent_size, input_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.fc = nn.Linear(latent_size, hidden_size * num_layers * 2)
        self.activation_fc = nn.PReLU()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc_mapping = nn.Linear(hidden_size, input_size)
        self.activation = nn.Sigmoid()

    def forward(self, latent, inputs):
        dec_hidden = self.fc(latent)
        dec_hidden = self.activation_fc(dec_hidden)

        dec_hidden = torch.reshape(dec_hidden, (self.num_layers, 1, -1))
        dec_hidden = torch.chunk(dec_hidden, 2, 2)
        dec_hidden = [state.contiguous() for state in dec_hidden]

        dec_input = torch.zeros_like(inputs)

        output, (hn, cn) = self.rnn(dec_input, dec_hidden)
        output = self.fc_mapping(output)
        output = self.activation(output)
        return output
