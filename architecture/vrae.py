import torch
import torch.nn as nn
import encoder
import decoder
import utils


class VRAE(nn.Module):

    def __init__(self, hidden_size, latent_size, input_size=utils.PIANO_ROLL_SIZE,
                 num_layers_dec=2, dtype=torch.float32):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.input_size = input_size
        self.dtype = dtype

        self.decoder = decoder.Decoder(hidden_size, latent_size, input_size, num_layers_dec, dtype=dtype)
        self.decoder.type(dtype)
        self.encoder = None

    def train_prep(self, num_layers_enc=2, bidirectional_enc=True):
        self.encoder = encoder.Encoder(self.hidden_size, self.latent_size, self.input_size, num_layers_enc,
                                       bidirectional=bidirectional_enc)
        self.encoder.type(self.dtype)

    def generate_prep(self, decoder_file):
        self.decoder.state_dict = torch.load(decoder_file)

    def forward(self, x):
        mean, sigma = self.encoder.forward(x)
        z = utils.reparametrize(mean, sigma)
        output = self.decoder(z, x.shape[0])
        return output, mean, sigma

    def generate(self, seq_len):
        z = torch.randn(self.latent_size).type(self.dtype).cuda()
        output = self.decoder(z, seq_len)
        return output

    def save_weights(self, decoder_file, mean_file, sigma_file):
        torch.save({'state-dict': self.decoder.state_dict()}, decoder_file)
