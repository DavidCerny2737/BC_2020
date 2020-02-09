import torch.nn as nn
import torch
import utils


class Encoder(nn.Module):

    def __init__(self, hidden_size, latent_size, input_size, num_layers=2, bidirectional=True):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)

        hidden_dim = hidden_size * num_layers * 2
        if bidirectional:
            hidden_dim *= 2
        self.activation = nn.Tanh()

        self.mean_net = nn.Linear(hidden_dim, latent_size)
        self.sigma_net = nn.Linear(hidden_dim, latent_size)

    def forward(self, x):
        output, hn = self.rnn(x)

        hn = torch.cat((hn[0], hn[1]), len(hn[0].shape) - 1).transpose(0, 1).contiguous()
        hn = hn.view(hn.shape[0], -1)
        hn = self.activation(hn)

        mean = self.mean_net(hn)
        logvar = self.sigma_net(hn)
        return mean, logvar
