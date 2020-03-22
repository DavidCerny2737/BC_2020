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
EPOCHS = 100
LR = 0.001
beta_1 = 0.05
beta_2 = 0.01
BETA = 50
BATCH_SIZE = 10
STEP_LR = 8
INPUT_SIZE = midi_utils.NUM_NOTES * midi_utils.SAMPLE_LENGHT

MODULO_PRINT = 4
DTYPE = torch.float64
DECODER_LAYERS = 3
ENCODER_LAYERS = 2


def test_reconstruction(model, seq_length):
    pr = midi_utils.get_random_piano_roll_sample(TRAIN_FOLDER)
    midi_utils.midi(pr, os.path.join(TARGET_FOLDER, 'test.mid'), npy=False)
    pr = torch.tensor(pr, dtype=DTYPE)
    # pr = pr.reshape((1, -1, pr.shape[2])).cuda()
    pr = pr.reshape((1, pr.shape[0], -1)).cuda()
    pr[pr == 80] = 1
    res = model(pr)[0].squeeze()
    print(res[res == 1].shape)
    res = torch.round(res).int().cpu().detach().numpy()
    midi_utils.midi(res, os.path.join(TARGET_FOLDER, 'test_res.mid'), npy=False)


def generate_sample(model, seq_length):
    output = model.generate(seq_length).squeeze()
    print(output[output == 1].shape)
    if len(output.shape) == 3:
        output = output.permute(1, 0, 2)
    output = torch.round(output).int().cpu().detach().numpy()
    midi = midi_utils.midi(output, os.path.join(TARGET_FOLDER, 'generated.mid'), npy=False)


def train_full_data(model):
    TEST_FOLDER = 'C:\\pycharmProjects\\BC_2020\\midi_data\\game\\test'
    #train_data = torch.utils.data.DataLoader(MidiNumpyDataset(TEST_FOLDER, DTYPE), batch_size=BATCH_SIZE, shuffle=True)
    train_data = torch.utils.data.DataLoader(MidiNumpyDataset(TRAIN_FOLDER, DTYPE), batch_size=BATCH_SIZE, shuffle=True)
    valid_data = torch.utils.data.DataLoader(MidiNumpyDataset(VALID_FOLDER, DTYPE), batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(beta_1, beta_2))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, STEP_LR, gamma=0.9)

    # na kazdou epochu prumerna hodnota pro kazdy ze tri lossu (pouze pro plot)
    train_loss_mean = numpy.ndarray(shape=(EPOCHS, 3), dtype=float)
    valid_loss_mean = numpy.ndarray(shape=(EPOCHS, 3), dtype=float)

    result = None
    # train
    for i in range(0, EPOCHS):
        # placeholder pro lossy pro pozdejsi prumerovani
        # kazdy list pro jeden ze tri lossu
        train_loss = [[], [], []]

        for local_batch in train_data:
            local_batch = local_batch.type(DTYPE).cuda()
            model.zero_grad()

            result, mean, log_sigma = model(local_batch)
            result = result.permute(1, 0, 2)
            latent_loss = -0.5 * torch.sum(1 + log_sigma - torch.pow(mean, 2) - torch.exp(log_sigma))

            rec_loss = -torch.sum(
                local_batch * torch.log(1e-10 + result) + (1 - local_batch) * torch.log(1e-10 + 1 - result))
            total_loss = BETA * latent_loss + rec_loss
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

            latent_loss = -0.5 * torch.sum(1 + log_sigma - torch.pow(mean, 2) - torch.exp(log_sigma))
            rec_loss = -torch.sum(
                local_batch * torch.log(1e-10 + result) + (1 - local_batch) * torch.log(1e-10 + 1 - result))
            total_loss = latent_loss + rec_loss

            valid_loss[0].append(latent_loss.clone().cpu().detach())
            valid_loss[1].append(rec_loss.clone().cpu().detach())
            valid_loss[2].append(total_loss.clone().cpu().detach())

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

    seq_length = 10
    test_reconstruction(model, seq_length)
    generate_sample(model, seq_length)


def train_single(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(beta_1, beta_2))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, STEP_LR, gamma=0.9)

    train_loss_mean = numpy.ndarray(shape=(EPOCHS, 3), dtype=float)
    result = None
    filename = 'part0aug_Portal_-_Still_Alive.mid.npz'
    # filename2 = 'part1aug_Portal_-_Still_Alive.mid.npz'
    pr = midi_utils.piano_roll(os.path.join(TRAIN_FOLDER, filename), npy=True)
    pr = torch.tensor(pr, dtype=DTYPE)
    # pr = pr.reshape(1, -1, pr.shape[2]).cuda()
    pr = pr.reshape(1, pr.shape[0], -1).cuda()

    # pr2 = midi_utils.piano_roll(os.path.join(TRAIN_FOLDER, filename2), npy=True)
    # pr2 = torch.tensor(pr2, dtype=DTYPE)
    # pr2 = pr2.reshape((1, pr2.shape[0], -1)).cuda()

    # pr = torch.cat((pr, pr2), 0)'''

    train_loss = [[], [], []]
    for i in range(0, EPOCHS):
        # pr = midi_utils.get_random_piano_roll_sample(TRAIN_FOLDER, npy=True)
        # pr = torch.tensor(pr, dtype=DTYPE)
        # pr = pr.reshape((1, pr.shape[0], -1)).cuda()

        model.zero_grad()

        # result, mean, log_sigma = model(local_batch)
        result, mean, log_sigma = model(pr)
        result = result.permute(1, 0, 2)
        latent_loss = -0.5 * torch.sum(1 + log_sigma - torch.pow(mean, 2) - torch.exp(log_sigma))

        rec_loss = -torch.sum(
            pr * torch.log(1e-10 + result) + (1 - pr) * torch.log(1e-10 + 1 - result))

        total_loss = latent_loss + rec_loss
        total_loss.backward()

        train_loss[0].append(rec_loss.data.clone().cpu().detach())
        train_loss[1].append(latent_loss.data.clone().cpu().detach())
        train_loss[2].append(total_loss.data.clone().cpu().detach())

        optimizer.step()
        scheduler.step()

        if i % MODULO_PRINT == 0:
            print('loss: %.6f' % total_loss)
            print('KL loss: %.6f' % latent_loss)
            print('RC loss: %.6f' % rec_loss)

    epochs = numpy.arange(EPOCHS)
    fig = plt.figure()

    ax1 = fig.add_subplot(131)
    ax1.plot(epochs, train_loss[0], label='Reconstruction loss - train')
    ax1.legend(['Train'])
    ax1.title.set_text('Reconstruction loss')

    ax2 = fig.add_subplot(132)
    ax2.plot(epochs, train_loss[1], label='Latent loss - train')
    ax2.legend(['Train'])
    ax2.title.set_text('Latent loss')

    ax3 = fig.add_subplot(133)
    ax3.plot(epochs, train_loss[2], label='Total loss - train')
    ax3.legend(['Train'])
    ax3.title.set_text('Total loss')
    plt.show()

    seq_length = 5
    test_reconstruction(model, seq_length)
    generate_sample(model, seq_length)


if __name__ == '__main__':
    freeze_support()

    model = vrae.VRAE(ENC_HIDDEN, LATENT_SIZE, num_layers_dec=DECODER_LAYERS, input_size=INPUT_SIZE, dtype=DTYPE)
    model.train_prep(num_layers_enc=ENCODER_LAYERS, bidirectional_enc=True)
    model.cuda()

    # train_single(model)
    train_full_data(model)
