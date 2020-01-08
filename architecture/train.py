import midi_utils
import vrae

import os
import torch
import numpy

ROOT_FOLDER = 'C:\\pycharmProjects\\BC_2020\\midi_data\\game\\transposed'
TARGET_FOLDER = 'C:\\pycharmProjects\\BC_2020\\midi_data\\game\\results'
FILE = 'aug_04woodman.mid'
COMPARE_FILE = 'orig'
ENC_HIDDEN = 64
LATENT_SIZE = 32
EPOCHS = 2000
LR = 0.001
beta_1 = 0.05
beta_2 = 0.001

def train():
    data = midi_utils.piano_roll(os.path.join(ROOT_FOLDER, FILE))[10]
    midi_utils.midi(data, os.path.join(TARGET_FOLDER, COMPARE_FILE))
    data = torch.tensor(data, dtype=torch.float32).cuda()

    orig_shape = data.squeeze().shape
    data = torch.reshape(data, (data.shape[0], 1, -1))

    model = vrae.VRAE(ENC_HIDDEN, LATENT_SIZE, num_layers_enc=1, num_layers_dec=1, input_size=data.shape[2],
                      bidirectional_enc=False)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(beta_1, beta_2))
    # cheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.9)
    reconstruction_loss = torch.nn.MSELoss()

    result = None
    for i in range(0, EPOCHS):
        model.zero_grad()
        result, mean, log_sigma = model(data)

        result = result.reshape(data.shape)
        # result = torch.reshape(result, (result.shape[0], 1, result.shape[1]))

        latent_loss = -0.5 * torch.sum(1 + log_sigma - torch.pow(mean, 2) - torch.exp(log_sigma))

        rec_loss = reconstruction_loss(result, data)
        # for j in range(result.shape[0]):
        # rec_loss -= data[j] * torch.log(1e-10 + result[j]) + (1 - data[j]) * torch.log(1e-10+ 1 - result[j])

        # rec_loss = data * torch.log(result) + (1 - data) * torch.log(1 - result)
        # rec_loss = data * torch.log(1e-10 + result) + (1 - data) * torch.log(1e-10 + 1 - result)
        # rec_loss = data * torch.log(1e-10 + result) + (1 - data) * torch.log(1e-10 + 1 - result)
        # rec_loss = -torch.sum(rec_loss)
        # rec_loss = - torch.sum(rec_loss)
        total_loss = latent_loss + rec_loss
        total_loss.backward()
        optimizer.step()
        # scheduler.step()

        if i % 10 == 0:
            print('loss: %.6f' % total_loss)
            print('KL loss: %.6f' % latent_loss)
            print('RC loss: %.6f' % rec_loss)
    result = torch.reshape(result, orig_shape)
    result = torch.round(result).int().cpu().detach().numpy()
    print(len(result[result == 1]))
    result = numpy.reshape(result, orig_shape)
    midi = midi_utils.midi(result, os.path.join(TARGET_FOLDER, FILE))


train()
