import torch.nn as nn
import utils


class Decoder(nn.Module):

    def __init__(self, latent_size, num_layers=2, output_size=utils.PIANO_ROLL_SIZE):
        super().__init__()
        self.rnn = nn.LSTM(latent_size, output_size, num_layers)
        self.activation = nn.Sigmoid()

    def forward(self, latent):
        output, (hn, cn) = self.rnn(latent)
        output = self.activation(output)
        return output
