import midi_utils
import vrae

import os
import torch
import numpy

ROOT_FOLDER = 'C:\\My Documents\\BC\\data\\game\\transposed\\'
TARGET_FOLDER = 'C:\\My Documents\\BC\\data\\game\\results\\'
FILE = 'aug_04woodman.mid'
COMPARE_FILE = 'orig'
ENC_HIDDEN = 512
LATENT_SIZE = 256
EPOCHS = 2000
LR = 1e-10


def train():
    data = midi_utils.piano_roll(os.path.join(ROOT_FOLDER, FILE))[10]
    data = numpy.reshape(data, (1, data.shape[0], data.shape[1]))
    midi_utils.midi(data, os.path.join(TARGET_FOLDER, COMPARE_FILE))
    data = torch.tensor(data, dtype=torch.float32).cuda()
    #print(data[data > 0])
    orig_shape = data.shape
    #data = torch.reshape(data, (data.shape[0], 1, -1))
    data = torch.reshape(data, (1, 1, -1))
    model = vrae.VRAE(ENC_HIDDEN, LATENT_SIZE, num_layers_enc=1, num_layers_dec=2)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.999)
    reconstruction_loss = torch.nn.CrossEntropyLoss()#torch.nn.MSELoss()
    standard_normal_dist = torch.distributions.normal.Normal(torch.tensor([0.0], device='cuda'),
                                                             torch.tensor([1.0], device='cuda'))
    result = None
    for i in range(0, EPOCHS):
        optimizer.zero_grad()
        result, mean, sigma = model(data)

        result = torch.squeeze(result).cuda().detach()
        #result = torch.reshape(result, (result.shape[0], 1, result.shape[1]))

        latent_loss = torch.mean(torch.distributions.kl_divergence(torch.distributions.normal.Normal(mean, sigma),
                                                                   standard_normal_dist))

        #rec_loss = reconstruction_loss(result, data).data
        rec_loss = 0
        #for j in range(result.shape[0]):
            #rec_loss -= data[j] * torch.log(1e-10 + result[j]) + (1 - data[j]) * torch.log(1e-10+ 1 - result[j])
        rec_loss = data * torch.log(1e-10 + result) + (1 - data) * torch.log(1e-10 + 1 - result)
        rec_loss = -torch.mean(rec_loss)
        total_loss = 0.01 * latent_loss + rec_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

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
