import torch.nn as nn
import utils


class Encoder(nn.Module):

    def __init__(self, hidden_size, latent_size, num_layers=2, input_size=utils.PIANO_ROLL_SIZE):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size // 2, num_layers, bidirectional=True)
        self.activation = nn.Tanh()
        self.mean_net = nn.Linear(hidden_size, latent_size)
        self.sigma_net = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        output, (hn, cn) = self.rnn(x)
        output = self.activation(output)
        mean = self.mean_net(output)
        sigma = self.sigma_net(output)
        return output, mean, sigma
