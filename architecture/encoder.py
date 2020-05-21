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

        init_hidden = nn.init.xavier_normal_(torch.zeros(num_layers * 2, 1,  hidden_size).cuda())
        init_state = nn.init.xavier_normal_(torch.zeros(num_layers * 2, 1,  hidden_size).cuda())
        self.init_hidden = nn.Parameter(init_hidden, requires_grad=True)
        self.init_state = nn.Parameter(init_state, requires_grad=True)

        self.activation = nn.Tanh()

        self.mean_net = nn.Linear(hidden_dim, latent_size)
        nn.init.xavier_uniform_(self.mean_net.weight)

        self.sigma_net = nn.Linear(hidden_dim, latent_size)
        nn.init.xavier_uniform_(self.sigma_net.weight)

    def forward(self, x):
        output, hn = self.rnn(x)

        hn = torch.cat((hn[0], hn[1]), len(hn[0].shape) - 1).permute(1, 0, 2).contiguous()
        hn = hn.view(hn.shape[0], -1)
        hn = self.activation(hn)

        mean = self.mean_net(hn)
        logvar = self.sigma_net(hn)

        return mean, logvar
