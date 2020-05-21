import pretty_midi
from os import listdir, rename, remove
from os.path import isfile, join, exists
import os
import shutil
import random
import collections
import pypianoroll
import numpy as np
from music_transformer import LOWER_BOUND, UPPER_BOUND

NUM_NOTES = 60
SAMPLE_LENGHT = 96
MIDI_NOTE_MAX = 128


class Analyzer:
    key_signature_translate = {0: 'C-dur', 1: 'C#-dur', 2: 'D-dur', 3: 'D#-dur', 4: 'E-dur', 5: 'F-dur',
                               6: 'F#-dur',
                               7: 'G-dur', 8: 'G#-dur', 9: 'A-dur', 10: 'A#-dur', 11: 'H-dur', 12: 'C-moll',
                               13: 'C#-moll',
                               14: 'D-moll', 15: 'D#-moll', 16: 'E-moll', 17: 'F-moll', 18: 'F#-moll',
                               19: 'G-moll',
                               20: 'G#-moll', 21: 'A-moll', 22: 'A#-moll', 23: 'H-moll'}
    note_names_translate = {'A': [21, 33, 45, 57, 69, 81, 93, 105], 'A#': [22, 34, 46, 58, 70, 82, 94, 106],
                            'H': [23, 35, 47, 59, 71, 83, 95, 107], 'C': [24, 36, 48, 60, 72, 84, 96, 108],
                            'C#': [25, 37, 49, 61, 73, 85, 97], 'D': [26, 38, 50, 62, 74, 86, 98],
                            'D#': [27, 39, 51, 63, 75, 87, 99], 'E': [28, 40, 52, 64, 76, 88, 100],
                            'F': [29, 41, 53, 65, 77, 89, 101], 'F#': [30, 42, 54, 66, 78, 90, 102],
                            'G': [31, 43, 55, 67, 79, 91, 103], 'G#': [32, 44, 56, 68, 80, 92, 104]}

    def __init__(self, root_folder):
        self.root_folder = root_folder

    def pitch_analyze(self, note_names=False):
        pitch_dict = collections.OrderedDict()
        midi_files = [file for file in listdir(self.root_folder) if isfile(join(self.root_folder, file))]
        for file in midi_files:
            try:
                midi_data = pretty_midi.PrettyMIDI(join(self.root_folder, file))
                for instrument in midi_data.instruments:
                    if instrument.program > 8 or instrument.is_drum:
                        continue

                    for note in instrument.notes:
                        # slovnik jmen not
                        if note_names:
                            for note_name, note_numbers in Analyzer.note_names_translate.items():
                                if note.pitch in Analyzer.note_names_translate[note_name]:
                                    if note_name in pitch_dict:
                                        pitch_dict[note_name] += 1
                                    else:
                                        pitch_dict[note_name] = 1
                        # slovnik cisel not
                        else:
                            if note.pitch in pitch_dict:
                                pitch_dict[note.pitch] += 1
                            else:
                                pitch_dict[note.pitch] = 1
            except IOError as e:
                print(str(e))
                print('File: ' + file)
        one_percent = sum(pitch_dict.values()) / 100
        for key, value in sorted(pitch_dict.items()):
            print(key, value, "{:10.4f}".format(value / one_percent) + '%')

    def pitch_analyze_single(self, file_name, note_names=False):
        pitch_dict = collections.OrderedDict()
        try:
            midi_data = pretty_midi.PrettyMIDI(join(self.root_folder, file_name))
            for instrument in midi_data.instruments:
                if instrument.program > 8 or instrument.is_drum:
                    continue

                for note in instrument.notes:
                    # slovnik jmen not
                    if note_names:
                        for note_name, note_numbers in Analyzer.note_names_translate.items():
                            if note.pitch in Analyzer.note_names_translate[note_name]:
                                if note_name in pitch_dict:
                                    pitch_dict[note_name] += 1
                                else:
                                    pitch_dict[note_name] = 1
                    # slovnik cisel not
                    else:
                        if note.pitch in pitch_dict:
                            pitch_dict[note.pitch] += 1
                        else:
                            pitch_dict[note.pitch] = 1
        except IOError as e:
            print(str(e))
        one_percent = sum(pitch_dict.values()) / 100
        for key, value in sorted(pitch_dict.items()):
            print(key, value, "{:10.4f}".format(value / one_percent) + '%')

    def time_analyze(self):
        time_dict = {}
        midi_files = [file for file in listdir(self.root_folder) if isfile(join(self.root_folder, file))]
        for file in midi_files:
            try:
                midi_data = pretty_midi.PrettyMIDI(join(self.root_folder, file))
                time_end = midi_data.get_end_time()
                if time_end in time_dict:
                    time_dict[time_end] += 1
                else:
                    time_dict[time_end] = 0
            except IOError as e:
                print(str(e))
                print('File: ' + file)
        for key in sorted(time_dict):
            print(str(key) + " : " + str(time_dict[key]))


    def time_signature_analyze(self):
        time_signature_dict = {}
        midi_files = [file for file in listdir(self.root_folder) if isfile(join(self.root_folder, file))]
        no_time_signature_files = 0
        for file in midi_files:
            try:
                midi_data = pretty_midi.PrettyMIDI(join(self.root_folder, file))
                time_signatures = midi_data.time_signature_changes

                # spousta souboru nema definovano time_signature
                if len(time_signatures) == 0:
                    no_time_signature_files += 1
                    continue

                for time_signature in time_signatures:
                    str_signature = str(time_signature.numerator) + '/' + str(time_signature.denominator)
                    if str_signature in time_signature_dict:
                        time_signature_dict[str_signature] += 1
                    else:
                        time_signature_dict[str_signature] = 1
            except IOError as e:
                print(str(e))
                print('File: ' + file)
        print(time_signature_dict)
        print("Bez time signature: " + str(no_time_signature_files))

    def key_signature_analyze(self):
        key_signature_dict = {}
        midi_files = [file for file in listdir(self.root_folder) if isfile(join(self.root_folder, file))]
        for file in midi_files:
            try:
                midi_data = pretty_midi.PrettyMIDI(join(self.root_folder, file))
                key_signatures = midi_data.key_signature_changes

                for key_signature in key_signatures:
                    signature_name = Analyzer.key_signature_translate[key_signature.key_number]
                    if signature_name in key_signature_dict:
                        key_signature_dict[signature_name] += 1
                    else:
                        key_signature_dict[signature_name] = 1
            except IOError as e:
                print(str(e))
                print('File: ' + file)
        print(key_signature_dict)
        print(len(key_signature_dict))


def instrument_analysis(self):
    instrument_dict = {}
    midi_files = [file for file in listdir(self.root_folder) if isfile(join(self.root_folder, file))]
    for file in midi_files:
        try:
            midi_data = pretty_midi.PrettyMIDI(join(self.root_folder, file))
            for instrument in midi_data.instruments:
                if instrument.name in instrument_dict:
                    instrument_dict[instrument.name] += 1
                else:
                    instrument_dict[instrument.name] = 1
        except IOError as e:
            print(str(e))
            print('File: ' + file)
    print(instrument_dict)


def file_clear(self, number_of_chars):
    midi_files = [file for file in listdir(self.root_folder) if isfile(join(self.root_folder, file))]
    for file in midi_files:
        new_name = file[:-number_of_chars]
        rename(join(self.root_folder, file), join(self.root_folder, new_name))


def piano_roll(file_name, npy=False):
    pr = None
    if npy:
        pr = pypianoroll.load(file_name)
    else:
        pr = pypianoroll.parse(file_name, beat_resolution=24)
    tracks = []
    try:
        for track in pr.tracks:
            if track.program > 8 or track.is_drum or len(track.pianoroll) == 0:
                continue
            tracks.append(track)
        if len(tracks) == 0:
            return None
        piano_roll = [np.zeros((SAMPLE_LENGHT, NUM_NOTES), dtype=np.uint8) for i in
                      range(0, np.ceil(len(tracks[0].pianoroll) / SAMPLE_LENGHT).astype(np.uint8))]
    except IndexError as e:
        print(e)
        print(file_name)
    for track in pr.tracks:
        if track.program > 8 or track.is_drum:
            continue
        lower_time = 0
        upper_time = SAMPLE_LENGHT
        for num, sample_roll in enumerate(piano_roll):
            if num == len(piano_roll) - 1:
                pass
            pr_track = track.pianoroll[lower_time:upper_time, LOWER_BOUND:UPPER_BOUND]
            if pr_track.shape[0] != sample_roll.shape[0]:
                help = np.copy(pr_track)
                pr_track = np.zeros_like(sample_roll)
                pr_track[:help.shape[0], :help.shape[1]] = help
            sample_roll[pr_track > 0] = 1
            lower_time += SAMPLE_LENGHT
            upper_time += SAMPLE_LENGHT
    return piano_roll


def midi(piano_roll, file_name, npy=False):
    piano_roll_joined = piano_roll
    piano_roll_joined = None
    for pr in piano_roll:
        if len(pr.shape) == 1:
            pr = np.reshape(pr, (SAMPLE_LENGHT, NUM_NOTES))
            '''help = np.ndarray((SAMPLE_LENGHT, NUM_NOTES), dtype=np.int)
            for i in range(0, len(pr) // SAMPLE_LENGHT):
                help[:, i] = pr[SAMPLE_LENGHT * i: SAMPLE_LENGHT * (i + 1)]
            pr = help'''
        if piano_roll_joined is None:
            piano_roll_joined = pr
            continue
        else:
            piano_roll_joined = np.concatenate((piano_roll_joined, pr))
    piano_roll_joined[piano_roll_joined == 1] = 80
    left_missing = np.zeros((piano_roll_joined.shape[0], LOWER_BOUND))
    right_missing = np.zeros((piano_roll_joined.shape[0], MIDI_NOTE_MAX - UPPER_BOUND))
    result = np.c_[left_missing, piano_roll_joined, right_missing]
    obj = pypianoroll.Multitrack(beat_resolution=24)
    track = pypianoroll.Track(result, program=1, is_drum=False)
    obj.tracks.append(track)
    if exists(file_name):
        remove(file_name)
    if npy:
        pypianoroll.save(file_name, obj)
    else:
        pypianoroll.write(obj, file_name)


def get_random_piano_roll_sample(root_dir, npy=False):
    file_names = os.listdir(root_dir)
    idx = random.randrange(len(file_names) - 1)
    return piano_roll(os.path.join(root_dir, file_names[idx]), npy=npy)


def debug_fce(file_name):
    data = pretty_midi.PrettyMIDI(file_name)
    for instrument in data.instruments:
        print()


def split_midi(root_dir, result_dir, too_small_dir, threshold, npy=False):
    if not os.path.isdir(root_dir):
        raise TypeError('cesta k midi souborum neni adresar')
    if not os.path.exists(too_small_dir):
        os.mkdir(too_small_dir)
    else:
        if not os.path.isdir(too_small_dir):
            raise TypeError('adresar pro ulozeni prilis malych souboru neni platny dir')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    else:
        if not os.path.isdir(result_dir):
            raise TypeError('adresar pro ulozeni vysledku neni platny dir')

    file_names = os.listdir(root_dir)
    for file_name in file_names:
        file = os.path.join(root_dir, file_name)
        try:
            pr = piano_roll(file)
        except Exception as e:
            print(e)
            continue
        if pr is None:
            continue

        # prilis kratky song
        '''if len(pr) < threshold:
            if npy:
                new_file_name = os.path.join(too_small_dir, 'valid_' + file_name)
                midi(pr, new_file_name, npy=npy)
            else:
                shutil.copyfile(file, os.path.join(too_small_dir, file_name))
            continue'''

        chunks = len(pr) // threshold
        for i in range(0, chunks):
            chunk = pr[i * threshold:(i + 1) * threshold]
            new_file_name = os.path.join(result_dir, 'part' + str(i) + file_name)
            midi(chunk, new_file_name, npy=True)


def inspect_seq_len(root_dir, threshold, npy=False):
    file_names = os.listdir(root_dir)
    min_seq = 10000
    count = 0
    for file_name in file_names:
        try:
            file = os.path.join(root_dir, file_name)
            midi = piano_roll(file, npy=npy)
            if min_seq > len(midi):
                min_seq = len(midi)
            if threshold > len(midi):
                print(file_name)
                count += 1
        except ValueError as e:
            print(file_name)
            print(e)
    print(min_seq)
    print(count)
    return min_seq


root_dir = 'C:\\pycharmProjects\\BC_2020\\midi_data\\game\\transposed'
# root_dir = 'C:\\pycharmProjects\\BC_2020\\midi_data\\new\\transposed'
too_small_dir = 'C:\\pycharmProjects\\BC_2020\\midi_data\\game\\valid'
result_dir = 'C:\\pycharmProjects\\BC_2020\\midi_data\\game\\train'
threshold = 5
# a = Analyzer(root_dir)
# a.key_signature_analyze()
# a.time_signature_analyze()
split_midi(root_dir, result_dir, too_small_dir, threshold, npy=True)
#inspect_seq_len(result_dir, threshold, npy=True)
# file_names = ['part0aug_Aion_Fairy_Of_The_Peace.mid', 'part1aug_Aion_Fairy_Of_The_Peace.mid',
#              'part2aug_Aion_Fairy_Of_The_Peace.mid', 'part0aug_AT.mid', 'part1aug_AT.mid', 'part0aug_whoareyou.mid']
# pr1 = piano_roll(os.path.join(result_dir, 'part0AdventureOfLink_Bossbattle.mid.npz'), npy=True)
# pr2 = piano_roll(os.path.join(result_dir, file_names[1]))
# pr3 = piano_roll(os.path.join(result_dir, file_names[2]))
# pr4 = piano_roll(os.path.join(result_dir, file_names[3]))
# pr5 = piano_roll(os.path.join(result_dir, file_names[4]))
# pr6 = piano_roll(os.path.join(result_dir, file_names[5]))
# print('yes')
