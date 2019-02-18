import numpy as np
import python_speech_features
import scipy.io.wavfile as wav


class FeaturesExtractor:

    def __init__(self, params: dict):
        if 'winfunc' in params and params['winfunc'] == 'hamming':
            params['winfunc'] = np.hamming
        self.params = params


    def get_features_mfcc(self, files: list) -> np.ndarray:
        """ Extract MFCC features from the files list. """
        mfccs = [self.make_mfcc(file, **self.params) for file in files]
        X = self.align(mfccs)
        return X


    @staticmethod
    def make_mfcc(file_path: str, **kwargs) -> np.ndarray:
        """ Use `python_speech_features` lib to extract MFCC features from the audio file. """
        fs, audio = wav.read(file_path)
        mfcc = python_speech_features.mfcc(audio, samplerate=fs, **kwargs)
        return mfcc


    @staticmethod
    def align(arrays: list, default=0) -> np.ndarray:
        """ Pad arrays along time dimensions. Return the single array (batch_size, time, features). """
        max_array = max(arrays, key=len)
        X = np.full(shape=[len(arrays), *max_array.shape], fill_value=default, dtype=np.float64)
        for index, array in enumerate(arrays):
            time_dim, features_dim = array.shape
            X[index, :time_dim] = array
        return X
