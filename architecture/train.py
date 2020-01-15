import midi_utils
import vrae

import os
import torch
import numpy

ROOT_FOLDER = 'C:\\pycharmProjects\\BC_2020\\midi_data\\game\\transposed'
TARGET_FOLDER = 'C:\\pycharmProjects\\BC_2020\\midi_data\\game\\results'
FILE = 'aug_04woodman.mid'
COMPARE_FILE = 'orig'
ENC_HIDDEN = 128
LATENT_SIZE = 64
EPOCHS = 1000
LR = 0.01
beta_1 = 0.05
beta_2 = 0.001
BATCH_SIZE = 2

def train():
    data1 = midi_utils.piano_roll(os.path.join(ROOT_FOLDER, FILE))
    data2 = midi_utils.piano_roll(os.path.join(ROOT_FOLDER, 'aug_06crashman.mid'))

    data2 = torch.tensor(data2, dtype=torch.float32)
    data2 = data2.reshape((data2.shape[0], 1, data2.shape[1], data2.shape[2]))

    data1 = torch.tensor(data1, dtype=torch.float32)
    orig_shape = data1.squeeze().shape
    data1 = data1.reshape((data1.shape[0], 1, data1.shape[1], data1.shape[2]))

    min_seq = min(data1.shape[0], data2.shape[0])
    data1 = data1[0:min_seq, :, :, :]
    data2 = data2[0:min_seq, :, :, :]
    data = torch.cat((data1, data2), 1)

    data = torch.reshape(data, (data.shape[0], data.shape[1], -1)).cuda()

    model = vrae.VRAE(ENC_HIDDEN, LATENT_SIZE, num_layers_enc=1, num_layers_dec=2, input_size=data.shape[2],
                      bidirectional_enc=False)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(beta_1, beta_2))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 70, gamma=0.9)
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
        scheduler.step()

        if i % 10 == 0:
            print('loss: %.6f' % total_loss)
            print('KL loss: %.6f' % latent_loss)
            print('RC loss: %.6f' % rec_loss)

    # save first song in batch
    result = result[:, 0, :]
    result = torch.reshape(result, orig_shape)
    result = torch.round(result).int().cpu().detach().numpy()
    # print(len(result[result == 1]))
    result = numpy.reshape(result, orig_shape)
    midi = midi_utils.midi(result, os.path.join(TARGET_FOLDER, 'generated.mid'))


train()
