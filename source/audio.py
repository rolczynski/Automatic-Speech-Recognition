import numpy as np
import python_speech_features
import scipy.io.wavfile as wav


def get_features_mfcc(files: list):
    """ Extract MFCC features from the files list. """
    mfccs = [make_mfcc(file) for file in files]
    X = align(mfccs)
    return X


def make_mfcc(file_path: str):
    """ Use `python_speech_features` lib to extract MFCC features from the audio file. """
    fs, audio = wav.read(file_path)
    mfcc = python_speech_features.mfcc(audio, samplerate=fs, numcep=26)
    return mfcc


def align(arrays: list, default=0):
    """ Pad arrays along time dimensions. Return the single array (batch_size, time, features). """
    max_array = max(arrays, key=len)
    X = np.full(shape=[len(arrays), *max_array.shape], fill_value=default)
    for index, array in enumerate(arrays):
        time_dim, features_dim = array.shape
        X[index, :time_dim] = array
    return X
