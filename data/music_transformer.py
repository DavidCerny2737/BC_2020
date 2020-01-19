import music21
import os
import math
from music21 import stream
import pretty_midi

LOWER_BOUND = 36
UPPER_BOUND = 96


def get_step_from_number(midi_number):
    if midi_number not in Transformer.midi_scale_translate.keys():
        key = Transformer.midi_scale_translate[midi_number - 12]
        step = Transformer.minors[key]
    # major
    else:
        key = Transformer.midi_scale_translate[midi_number]
        step = Transformer.majors[key]
    return step


def pitch_clip(data, lower_bound, upper_bound):
    try:
        for instrument in data.instruments:
            if instrument.program > 8 or instrument.is_drum:
                continue
            for note in instrument.notes:
                if note.pitch < lower_bound:
                    note.pitch += 12
                if note.pitch > upper_bound:
                    note.pitch -= 12
    except IOError as e:
        print(e)


def pitch_print(score, lower_bound, upper_bound):
    result = {}
    for part in score.parts:
        for pitch in part.pitches:
            if pitch.midi < lower_bound or pitch.midi > upper_bound:
                if pitch.nameWithOctave in result:
                    result[pitch.nameWithOctave] += 1
                else:
                    result[pitch.nameWithOctave] = 1
    print(result)


class Transformer:
    majors = dict(
        [("A-", 4), ("A", 3), ("A#", 2), ("B-", 2), ("B", 1),  ("B#", 0), ("C-", 1), ("C", 0), ("C#", -1),
         ("D-", -1), ("D", -2), ("D#", -3), ("E-", -3), ("E", -4), ("E#", -5), ("F-", -4), ("F", -5), ("F#", -6),
         ("G-", 6), ("G", 5), ("G#", 4)])
    minors = dict(
        [("A-", 1), ("A", 0), ("A#", -1), ("B-", -1), ("B", -2), ("B#", -3), ("C-", -2), ("C", -3), ("C#", -4),
         ("D-", -4), ("D", -5), ("D#", 6), ("E-", 6), ("E", 5), ("E#", 4), ("F-", 5), ("F", 4), ("F#", 3),
         ("G-", 3), ("G", 2), ("G#", 1)])
    midi_scale_translate = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A',
                            10: 'A#', 11: 'B'}

    def __init__(self, root_folder, target_folder, file_prefix='aug_'):
        self.root_folder = root_folder
        self.target_folder = target_folder
        self.file_prefix = file_prefix
        self.files = [file for file in os.listdir(self.root_folder) if
                      os.path.isfile(os.path.join(self.root_folder, file))]

    def start_process(self, scale_transpose=True, pitch_clip_augumentation=True, lower_bound=26, upper_bound=96):
        for file in self.files:
            data = pretty_midi.PrettyMIDI(os.path.join(self.root_folder, file))
            if scale_transpose:
                # print(score.analyze('key'))
                self.transpose_to_c_major(data, file)
                # print(score.analyze('key'))
            if pitch_clip_augumentation:
                # self.pith_print(score, lower_bound, upper_bound)
                pitch_clip(data, lower_bound, upper_bound)
                # self.pith_print(score, lower_bound, upper_bound)
            self.commit(data, file)

    def transpose_to_c_major(self, data, file_name):
        try:
            key_signatures = data.key_signature_changes
            if len(key_signatures) == 0:
                music21_obj = music21.converter.parseFile(os.path.join(self.root_folder, file_name))
                key = music21_obj.analyze('key')
                if key.mode == "major":
                    step = Transformer.majors[key.tonic.name]
                else:
                    step = Transformer.minors[key.tonic.name]
                for instrument in data.instruments:
                    if instrument.program > 8 or instrument.is_drum:
                        continue
                    for note in instrument.notes:
                        note.pitch += step
            elif len(key_signatures) > 1:
                for instrument in data.instruments:
                    if instrument.program > 8 or instrument.is_drum:
                        continue
                    i = 0
                    actual_step = get_step_from_number(key_signatures[i].key_number)
                    last_signature = False
                    end = key_signatures[i + 1].time
                    for note in instrument.notes:
                        if not last_signature and note.start >= end:
                            i += 1
                            actual_step = get_step_from_number(key_signatures[i].key_number)
                            if len(key_signatures) - 1 == i:
                                last_signature = True
                            else:
                                end = key_signatures[i + 1].time
                        note.pitch += actual_step
            else:
                step = get_step_from_number(key_signatures[0].key_number)
                for instrument in data.instruments:
                    if instrument.program > 8 or instrument.is_drum:
                        continue
                    for note in instrument.notes:
                        note.pitch += step
        except IOError as e:
            print(e)

    def commit(self, data, file_name):
        new_file_name = os.path.join(self.target_folder, self.file_prefix + file_name)
        data.write(new_file_name)


# root_folder = 'C:\\pycharmProjects\\BC_2020\\midi_data\\game\\orig\\'
# target_folder = 'C:\\pycharmProjects\\BC_2020\\midi_data\\game\\transposed\\'
# transformer = Transformer(root_folder, target_folder)
# transformer.start_process(lower_bound=36, upper_bound=96)
