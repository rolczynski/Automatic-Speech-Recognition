import h5py
import numpy as np
import pandas as pd
from keras.utils import Sequence
from source import audio, text


class DataGenerator(Sequence):
    """
    Generates data for Keras

    `Sequence` are a safer way to do multiprocessing. This structure
    guarantees that the network will only train once on each sample per epoch
    which is not the case with generators.

    References:
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
    """
    def __init__(self, source_indicators, alphabet, shuffle_after_epoch=1, batch_size=30, features_store=False):
        self._source_indicators = source_indicators
        self._features_store = features_store
        self._alphabet = alphabet
        self._batch_size = batch_size
        self._shuffle_after_epoch = shuffle_after_epoch

        self.epoch = 0
        self.indices = np.arange(len(self))


    @classmethod
    def from_audio_files(cls, file_path, **kwargs):
        """ Create generator from csv file. The file contains audio file paths
        with corresponding transcriptions. """
        source_indicators = pd.read_csv(file_path, names=['name', 'transcript'],
                                        sep=',', encoding='utf-8', header=0)
        return cls(source_indicators=source_indicators, **kwargs)


    @classmethod
    def from_prepared_features(cls, file_path, **kwargs):
        """ Create generator from prepared features saved in the HDF5 format. """
        features_store = h5py.File(file_path, mode='r')
        source_indicators = pd.HDFStore(file_path, mode='r')['source_indicators']   # Read DataFrame via PyTables
        return cls(source_indicators=source_indicators, features_store=features_store, **kwargs)


    def __len__(self):
        """ Denotes the number of batches per epoch. """
        return int(np.floor(len(self._source_indicators.index) / self._batch_size))


    def __getitem__(self, next_index):
        """ Operator to get the batch data. """
        batch_index = self.indices[next_index]
        return self._get_batch(batch_index)


    def _get_batch(self, index):
        """ Read (if features store exist) or generate features and labels batch. """
        start, end = index*self._batch_size, (index+1)*self._batch_size
        batch_indicators = self._source_indicators[start:end]
        names, transcripts = batch_indicators.name, batch_indicators.transcript

        labels = text.get_batch_labels(transcripts, self._alphabet)
        if self._features_store:
            features = self._read_features(names)
        else:
            features = self._extract_features(names)
        return features, labels


    def _read_features(self, names):
        """ Read already prepared features from the audio store. """
        features = [self._features_store[name][:] for name in names]
        return audio.align(features)


    def _extract_features(self, names):
        """ Extract features from the audio files (mono 16kHz). """
        return audio.get_features_mfcc(files=names)


    def on_epoch_end(self):
        """ Invoke methods at the end of the each epoch. """
        self.epoch += 1
        self._shuffle_indices()


    def _shuffle_indices(self):
        """ Set up the order of next batches """
        if self.epoch >= self._shuffle_after_epoch:
            np.random.shuffle(self.indices)
