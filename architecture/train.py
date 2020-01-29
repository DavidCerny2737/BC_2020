from multiprocessing import freeze_support

import midi_utils
import vrae
from MidiDataset import MidiDataset, MidiNumpyDataset

import os
import torch
import torch.utils.data
import numpy
import matplotlib.pyplot as plt

TRAIN_FOLDER = 'C:\\pycharmProjects\\BC_2020\\midi_data\\game\\train'
VALID_FOLDER = 'C:\\pycharmProjects\\BC_2020\\midi_data\\game\\valid'
TARGET_FOLDER = 'C:\\pycharmProjects\\BC_2020\\midi_data\\game\\results'
SAVE_WEIGHTS = True
WEIGHTS_PATH = 'C:\\pycharmProjects\\BC_2020\\params.pth'
MEAN_PATH = 'C:\\pycharmProjects\\BC_2020\\mean.pth'
SIGMA_PATH = 'C:\\pycharmProjects\\BC_2020\\sigma.pth'

ENC_HIDDEN = 512
LATENT_SIZE = 256
EPOCHS = 40
LR = 0.009
beta_1 = 0.05
beta_2 = 0.001
BATCH_SIZE = 16
STEP_LR = 4
INPUT_SIZE = midi_utils.SAMPLE_LENGHT * midi_utils.NUM_NOTES

MODULO_PRINT = 2
DTYPE = torch.float64
DECODER_LAYERS = 2
ENCODER_LAYERS = 2

def test_reconstruction(model, seq_length):
    pr = midi_utils.get_random_piano_roll_sample(TRAIN_FOLDER)
    midi_utils.midi(pr, os.path.join(TARGET_FOLDER, 'test.mid'), npy=False)
    pr = torch.tensor(pr, dtype=DTYPE)
    pr = pr.reshape((pr.shape[0], 1, -1)).cuda()
    pr[pr == 80] = 1
    res = model(pr)[0].squeeze()
    res = torch.reshape(res, (seq_length, midi_utils.SAMPLE_LENGHT, midi_utils.NUM_NOTES))
    res = torch.round(res).int().cpu().detach().numpy()
    midi_utils.midi(res, os.path.join(TARGET_FOLDER, 'test_res.mid'), npy=False)


if __name__ == '__main__':
    freeze_support()
    train_data = torch.utils.data.DataLoader(MidiNumpyDataset(TRAIN_FOLDER), batch_size=BATCH_SIZE, shuffle=True)
    # batch size validace je 1 protoze soubory ve validaci jsou rozdilne velky
    valid_data = torch.utils.data.DataLoader(MidiNumpyDataset(VALID_FOLDER), batch_size=1, shuffle=True)

    model = vrae.VRAE(ENC_HIDDEN, LATENT_SIZE, num_layers_dec=DECODER_LAYERS, input_size=INPUT_SIZE, dtype=DTYPE)
    model.train_prep(num_layers_enc=ENCODER_LAYERS, bidirectional_enc=True)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(beta_1, beta_2))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, STEP_LR, gamma=0.9)
    reconstruction_loss = torch.nn.MSELoss()

    # na kazdou epochu prumerna hodnota pro kazdy ze tri lossu (pouze pro plot)
    train_loss_mean = numpy.ndarray(shape=(EPOCHS, 3), dtype=float)
    valid_loss_mean = numpy.ndarray(shape=(EPOCHS, 3), dtype=float)

    result = None
    # train
    # filename = 'part0aug_Portal_-_Still_Alive.mid.npz'
    # pr = midi_utils.piano_roll(os.path.join(TRAIN_FOLDER, filename), npy=True)
    # pr = torch.tensor(pr, dtype=DTYPE)
    # pr = pr.reshape((pr.shape[0], 1, -1)).cuda()
    for i in range(0, EPOCHS):
        # placeholder pro lossy pro pozdejsi prumerovani
        # kazdy list pro jeden ze tri lossu
        train_loss = [[], [], []]

        for local_batch in train_data:
            local_batch = local_batch.type(DTYPE).cuda()
            model.zero_grad()

            #result, mean, log_sigma = model(local_batch)
            result, mean, log_sigma = model(local_batch)
            result = result.reshape(local_batch.shape)
            latent_loss = -0.5 * torch.sum(1 + log_sigma - torch.pow(mean, 2) - torch.exp(log_sigma))
                # rec_loss = reconstruction_loss(result, pr)
                    # for j in range(result.shape[0]):
                    # rec_loss -= data[j] * torch.log(1e-10 + result[j]) + (1 - data[j]) * torch.log(1e-10+ 1 - result[j])

            # rec_loss = -torch.sum(
               # local_batch * torch.log(1e-10 + result) + (1 - local_batch) * torch.log(1e-10 + 1 - result))

            rec_loss = -torch.sum(
                local_batch * torch.log(1e-10 + result) + (1 - local_batch) * torch.log(1e-10 + 1 - result))
            total_loss = latent_loss + rec_loss
            total_loss.backward()

            train_loss[0].append(rec_loss.data.clone().cpu().detach())
            train_loss[1].append(latent_loss.data.clone().cpu().detach())
            train_loss[2].append(total_loss.data.clone().cpu().detach())

            optimizer.step()

        if i % MODULO_PRINT == 0:
            print('loss: %.6f' % total_loss)
            print('KL loss: %.6f' % latent_loss)
            print('RC loss: %.6f' % rec_loss)

        scheduler.step()

        train_loss_mean[i, 0] = torch.mean(torch.Tensor(train_loss[0])).numpy()
        train_loss_mean[i, 1] = torch.mean(torch.Tensor(train_loss[1])).numpy()
        train_loss_mean[i, 2] = torch.mean(torch.Tensor(train_loss[2])).numpy()

        # validace
        '''valid_loss = [[], [], []]
        for local_batch in valid_data:
            local_batch = local_batch.cuda()
            result, mean, log_sigma = model(local_batch)
            result = result.reshape(local_batch.shape)
            valid_loss[0].append(-0.5 * torch.sum(1 + log_sigma - torch.pow(mean, 2) - torch.exp(log_sigma)))
            valid_loss[1].append(reconstruction_loss(result, local_batch))
            valid_loss[2].append(latent_loss + rec_loss)

        valid_loss_mean[i, 0] = torch.mean(torch.Tensor(valid_loss[0])).numpy()
        valid_loss_mean[i, 1] = torch.mean(torch.Tensor(valid_loss[1])).numpy()
        valid_loss_mean[i, 2] = torch.mean(torch.Tensor(valid_loss[2])).numpy()'''

    # plot
    epochs = numpy.arange(EPOCHS)
    fig = plt.figure()

    ax1 = fig.add_subplot(131)
    ax1.plot(epochs, train_loss_mean[:, 0], label='Reconstruction loss - train')
    # ax1.plot(epochs, valid_loss_mean[:, 0], label='Reconstruction loss - valid')
    # ax1.legend(['Train', 'Valid'])
    ax1.legend(['Train'])
    ax1.title.set_text('Reconstruction loss')

    ax2 = fig.add_subplot(132)
    ax2.plot(epochs, train_loss_mean[:, 1], label='Latent loss - train')
    # ax2.plot(epochs, valid_loss_mean[:, 1], label='Latent loss - valid')
    # ax2.legend(['Train', 'Valid'])
    ax2.legend(['Train'])
    ax2.title.set_text('Latent loss')

    ax3 = fig.add_subplot(133)
    ax3.plot(epochs, train_loss_mean[:, 2], label='Total loss - train')
    # ax3.plot(epochs, valid_loss_mean[:, 2], label='Total loss - valid')
    # ax3.legend(['Train', 'Valid'])
    ax3.legend(['Train'])
    ax3.title.set_text('Total loss')

    plt.show()

    if SAVE_WEIGHTS:
        model.save_weights(WEIGHTS_PATH, MEAN_PATH, SIGMA_PATH)

    seq_length = 30
    test_reconstruction(model, seq_length)

    output = model.generate(seq_length).squeeze()
    result = torch.reshape(output, (seq_length, midi_utils.SAMPLE_LENGHT, midi_utils.NUM_NOTES))
    result = torch.round(result).int().cpu().detach().numpy()
    midi = midi_utils.midi(result, os.path.join(TARGET_FOLDER, 'generated.mid'), npy=False)
