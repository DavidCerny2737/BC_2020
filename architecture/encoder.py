import torch.nn as nn
import torch
import utils


class Encoder(nn.Module):

    def __init__(self, hidden_size, latent_size, input_size, num_layers=2, bidirectional=True):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional)

        hidden_dim = hidden_size * num_layers * 2
        if bidirectional:
            hidden_dim *= 2

        self.mean_net = nn.Linear(hidden_dim, latent_size)
        self.sigma_net = nn.Linear(hidden_dim, latent_size)
        self.activation_mean = nn.PReLU()
        self.activation_logvar = nn.PReLU()

    def forward(self, x):
        output, hn = self.rnn(x)

        hn = torch.cat((hn[0], hn[1]), 2).transpose(0, 1).contiguous()
        hn = hn.view(hn.shape[0], -1)

        mean = self.mean_net(hn)
        mean = self.activation_mean(mean)
        logvar = self.sigma_net(hn)
        logvar = self.activation_logvar(logvar)
        return mean, logvar
