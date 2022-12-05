import pypianoroll
import os
import glob
import numpy as np
from tqdm import tqdm

from src.constants import LOWEST_PITCH, NUMBER_OF_PITCHES, LPD_FILE_EXTENSION, MAESTRO_FILE_EXTENSION


def is_divisible_by(pianoroll: np.ndarray, N: int) -> bool:
    """
    Method that checks if an array can be divisible into N arrays with the same length.
    :param pianoroll: pianoroll object to check
    :param N: the number by which to divide
    :return bool: return True or False - is divisible or not
    """
    return pianoroll.shape[1] % N == 0


def do_padding(pianoroll: np.ndarray, chunk: int) -> np.ndarray:
    """
    Method that does padding on a given array (pianoroll) based on the chunk.
    :param pianoroll: an array on which to do the padding
    :param chunk: an array must be divided by the chunk, if its not then we do the padding
    :return np.ndarray: padded array (pianoroll)
    """
    x = chunk - (pianoroll.shape[1] % chunk)
    padding = np.zeros((pianoroll.shape[0], x))
    return np.concatenate((pianoroll, padding), axis=1)


def how_much_music_data(array: np.ndarray) -> float:
    """
    Method which checks how many % of the pixels have music information in them (how much white pixels).
    :param array: ann array to check
    :return float: return % of music information in an array
    """
    array = array.flatten().astype(int)
    return sum(array) / len(array)


def is_good_sample(array: np.ndarray, percentage_of_music_data_recquired: float) -> bool:
    """
    Method that checks if an array has enough music information in it (% of white pixels).
    :param array: an array to check
    :param percentage_of_music_data_recquired:
    :return bool: return True or False - is sample good or not
    """
    return how_much_music_data(array) >= percentage_of_music_data_recquired


def is_empty(array: np.ndarray) -> bool:
    """
    Method that check if an array is 'empty' (has no music information).
    :param array: an array to check
    :return bool: return True or False - is array empty or not
    """
    return np.all(array == False)


def prepare_data(file_path: str, length: int, music_info_threshold: float, file_extension: str,\
     pianoroll_idx: int=1, do_filtration: bool=True) -> np.ndarray:
    """
    Method that prepares training data.
    :param file_path: path to a file with data.
    :param length: how long the samples will be.
    :param music_info_threshold: how much music information (% white pixels) must a sample have
    :param file_extension: .midi or .npz files are acceptable
    :param pianoroll_idx: in LPD it's for example 1 and in MAESTRO it's 0
    :param do_filtration: segregation of data based on the amount of musical information it has
    :return np.ndarray: prepared data as an array
    """
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