import torch
import torch.nn as nn
import encoder
import decoder
import utils


class VRAE(nn.Module):

    def __init__(self, hidden_size, latent_size, input_size=utils.PIANO_ROLL_SIZE, num_layers_enc=2,
                 num_layers_dec=2):
        super().__init__()
        self.encoder = encoder.Encoder(hidden_size, latent_size, input_size, num_layers_enc)
        self.decoder = decoder.Decoder(hidden_size, latent_size, input_size, num_layers_dec)

    def forward(self, x):
        mean, sigma = self.encoder.forward(x)
        z = utils.reparametrize(mean, sigma)
        output = self.decoder(z, x)
        return output, mean, sigma
