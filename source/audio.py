import numpy as np
import python_speech_features
import scipy.io.wavfile as wav
from keras.preprocessing.sequence import pad_sequences


def get_features_mfcc(files):
    """ Extract MFCC features """
    x_val = [get_sample_length(file) for file in files]
    max_val = max(x_val)
    X = np.array([make_mfcc(file, padlen=max_val) for file in files])
    return X


def get_sample_length(file_path):
    """ Read .wav file and return the sample length """
    fs, audio = wav.read(file_path)
    r = python_speech_features.mfcc(audio, samplerate=fs, numcep=26)
    return r.shape[0]


def make_mfcc(filename, padlen):
    """ Use `python_speech_features` lib to extract MFCC features """
    fs, audio = wav.read(filename)
    r = python_speech_features.mfcc(audio, samplerate=fs, numcep=26)
    t = np.transpose(r)
    X = pad_sequences(t, maxlen=padlen, dtype='float', padding='post', truncating='post').T
    return X