import pretty_midi
from os import listdir, rename, remove
from os.path import isfile, join, exists
import collections
import music21
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


def time_signature_analyze(self):
    time_signature_dict = {}
    midi_files = [file for file in listdir(self.root_folder) if isfile(join(self.root_folder, file))]
    for file in midi_files:
        try:
            midi_data = pretty_midi.PrettyMIDI(join(self.root_folder, file))
            time_signatures = midi_data.time_signature_changes

            # spousta souboru nema definovano time_signature
            if len(time_signatures) == 0:
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


def piano_roll(file_name):
    pr = pypianoroll.parse(file_name, beat_resolution=24)
    tracks = []
    for track in pr.tracks:
        if track.program > 8 or track.is_drum:
            continue
        tracks.append(track)
    piano_roll = [np.zeros((SAMPLE_LENGHT, NUM_NOTES), dtype=np.uint8) for i in
                  range(0, np.ceil(len(pr.tracks[0].pianoroll) / SAMPLE_LENGHT).astype(np.uint8))]
    for track in pr.tracks:
        if track.program > 8 or track.is_drum:
            continue
        lower_time = 0
        upper_time = 96
        for num, sample_roll in enumerate(piano_roll):
            if num == len(piano_roll) - 1:
                print()
            pr_track = track.pianoroll[lower_time:upper_time, LOWER_BOUND:UPPER_BOUND]
            if pr_track.shape[0] != sample_roll.shape[0]:
                help = np.copy(pr_track)
                pr_track = np.zeros_like(sample_roll)
                pr_track[:help.shape[0], :help.shape[1]] = help
            sample_roll[pr_track > 0] = 1
            lower_time += SAMPLE_LENGHT
            upper_time += SAMPLE_LENGHT
    last_sample = piano_roll[-1]
    if not np.any(last_sample):
        piano_roll.pop()
    return piano_roll


def midi(piano_roll, file_name):
    piano_roll_joined = piano_roll
    piano_roll_joined = None
    for pr in piano_roll:
        if piano_roll_joined is None:
            piano_roll_joined = pr
            continue
        else:
            piano_roll_joined = np.concatenate((piano_roll_joined, pr))
    piano_roll_joined[piano_roll_joined == 1] = 80
    left_missing = np.zeros((piano_roll_joined.shape[0], LOWER_BOUND))
    right_missing = np.zeros((piano_roll_joined.shape[0], MIDI_NOTE_MAX - UPPER_BOUND))
    result = np.c_[left_missing, piano_roll_joined, right_missing]
    obj = pypianoroll.Multitrack()
    track = pypianoroll.Track(result, program=1, is_drum=False)
    obj.tracks.append(track)
    if exists(file_name):
        remove(file_name)
    pypianoroll.write(obj, file_name)


def debug_fce(file_name):
    data = pretty_midi.PrettyMIDI(file_name)
    for instrument in data.instruments:
        print()

# midi_dir = 'C:\\My Documents\\BC\\data\\game\\orig\\'
# target_dir = 'C:\\My Documents\\BC\\data\\game\\transposed\\'
# file_name = '2_-_A_brave_Warrior.mid'
# midi_transposed_dir = 'C:\\My Documents\\BC\\data\\game\\transposed\\'
# analyzer = Analyzer(midi_dir)
# analyzer = Analyzer(target_dir)
# analyzer.pitch_analyze_single(file_name, note_names=False)
# analyzer2.pitch_analyze_single('C_' + file_name, note_names=False)
# analyzer.pitch_analyze(note_names=True) # 26 - 96
# analyzer.time_signature_analyze()
# analyzer.key_signature_analyze()
# analyzer.instrument_analysis()
# analyzer.pith_print()
# analyzer.file_clear(4)

# root_folder = 'C:\\My Documents\\BC\\data\\game\\transposed\\'
# target_folder = 'C:\\My Documents\\BC\\data\\game\\results\\'
# file = 'aug_04woodman.mid'
# p_roll = piano_roll(join(root_folder, file))
# midi(piano_roll(join(root_folder, file)), join(target_folder, file))
# debug_fce(join(root_folder, file))
# midi_data = pretty_midi.PrettyMIDI(join(root_folder, file))
