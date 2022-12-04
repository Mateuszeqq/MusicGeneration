import pypianoroll
import os
import glob
import numpy as np
from tqdm import tqdm

from src.constants import LOWEST_PITCH, NUMBER_OF_PITCHES, LPD_FILE_EXTENSION, MAESTRO_FILE_EXTENSION


def is_divisible_by(pianoroll, N):
    return pianoroll.shape[1] % N == 0


def do_padding(pianoroll, chunk):
    x = chunk - (pianoroll.shape[1] % chunk)
    padding = np.zeros((pianoroll.shape[0], x))
    return np.concatenate((pianoroll, padding), axis=1)


def how_much_music_data(array):
    array = array.flatten().astype(int)
    return sum(array) / len(array)


def is_good_sample(array, percentage_of_music_data_recquired):
    return how_much_music_data(array) >= percentage_of_music_data_recquired


def is_empty(array):
    return np.all(array == False)


def prepare_data(file_path, length, music_info_threshold, file_extension, pianoroll_idx=1, do_filtration=True):
    data = []
    for midi_file in tqdm([y for x in os.walk(file_path) for y in glob.glob(os.path.join(x[0], file_extension))]):
        if file_extension == LPD_FILE_EXTENSION:
            multitrack = pypianoroll.load(midi_file)
        elif file_extension == MAESTRO_FILE_EXTENSION:
            multitrack = pypianoroll.read(midi_file)
        else:
            raise Exception(f'Unknown file extension: {file_extension}')
        multitrack.binarize()

        pr_array = multitrack.tracks[pianoroll_idx].pianoroll
        pr_T = np.transpose(pr_array)
        x = do_padding(pr_T, length)
        batches = int(x.shape[1] / length)
        split_pr = np.array_split(x, batches, axis=1)
        for chunk in split_pr:
            if is_empty(chunk):
                continue
            if do_filtration and not is_good_sample(chunk, music_info_threshold):
                continue
            data.append(chunk[LOWEST_PITCH:LOWEST_PITCH + NUMBER_OF_PITCHES, :].transpose())
    return np.stack(data)