from multiprocessing import freeze_support

import torch
import torch.utils.data
import numpy
import midi_utils

import time
import os


class MidiDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, dtype):
        super().__init__()
        if not os.path.isdir(root_dir):
            raise TypeError('cesta k midi souborum neni adresar')
        self.root_dir = root_dir
        self.dtype = dtype

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, item):
        file_name = os.listdir(self.root_dir)[item]
        file = os.path.join(self.root_dir, file_name)
        midi_data = midi_utils.piano_roll(file)
        pr = torch.tensor(midi_data, dtype=self.dtype)
        pr = pr.reshape((pr.shape[0], -1))
        return pr


class MidiNumpyDataset(MidiDataset):

    def __init__(self, root_dir, dtype):
        super().__init__(root_dir, dtype)


    def __getitem__(self, item):
        file_name = os.listdir(self.root_dir)[item]
        file = os.path.join(self.root_dir, file_name)
        midi_data = midi_utils.piano_roll(file, npy=True)
        pr = torch.tensor(midi_data, dtype=self.dtype)
        pr = pr.reshape((pr.shape[0], -1))
        return pr


# root_dir = 'C:\\pycharmProjects\\BC_2020\\midi_data\\game\\train'
# valid_dir = 'C:\\pycharmProjects\\BC_2020\\midi_data\\game\\valid'
# inspect_seq_len(root_dir, npy=True)

'''if __name__ == '__main__':
    freeze_support()
    start = time.time()
    data_loader = torch.utils.data.DataLoader(MidiNumpyDataset(root_dir), batch_size=64, shuffle=True)
    try:
        for batch_ndx, sample in enumerate(data_loader):
            print('batch')
    except KeyError as e:
        print(e)
    end = time.time()
    print(end - start)

    start = time.time()
    data_loader = torch.utils.data.DataLoader(MidiNumpyDataset(root_dir), batch_size=64, shuffle=True, num_workers=1)
    try:
        for batch_ndx, sample in enumerate(data_loader):
            print('batch')
    except KeyError as e:
        print(e)
    end = time.time()
    print(end - start)'''