import h5py
import numpy as np
import pandas as pd
from keras.utils import Sequence
from source import audio, text
from source.text import Alphabet
from source.audio import FeaturesExtractor


class DataGenerator(Sequence):
    """
    Generates data for Keras

    `Sequence` are a safer way to do multiprocessing. This structure
    guarantees that the network will only train once on each sample per epoch
    which is not the case with generators.

    References:
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
    """
    def __init__(self,
                 metadata: pd.DataFrame,
                 alphabet: Alphabet,
                 features_extractor: FeaturesExtractor,
                 shuffle_after_epoch=1,
                 batch_size=30,
                 features_store=False):
        self._metadata = metadata
        self._features_store = features_store
        self._alphabet = alphabet
        self._features_extractor = features_extractor
        self._batch_size = batch_size
        self._shuffle_after_epoch = shuffle_after_epoch

        self.epoch = 0
        self.indices = np.arange(len(self))


    @classmethod
    def from_audio_files(cls, file_path, **kwargs):
        """ Create generator from csv file. The file contains audio file paths
        with corresponding transcriptions. """
        metadata = pd.read_csv(file_path, usecols=['path', 'transcript'], sep=',', encoding='utf-8', header=0)
        return cls(metadata=metadata, **kwargs)


    @classmethod
    def from_prepared_features(cls, file_path, **kwargs):
        """ Create generator from prepared features saved in the HDF5 format.
        The hdf5 file has the hierarchy with /-separator and also can be invoke via `path`. """
        features_store = h5py.File(file_path, mode='r')
        metadata = pd.HDFStore(file_path, mode='r')['metadata']   # Read DataFrame via PyTables
        return cls(metadata=metadata, features_store=features_store, **kwargs)


    def __len__(self):
        """ Denotes the number of batches per epoch. """
        return int(np.floor(len(self._metadata.index) / self._batch_size))


    def __getitem__(self, next_index):
        """ Operator to get the batch data. """
        batch_index = self.indices[next_index]
        return self._get_batch(batch_index)


    def _get_batch(self, index):
        """ Read (if features store exist) or generate features and labels batch. """
        start, end = index*self._batch_size, (index+1)*self._batch_size
        metadata = self._metadata[start:end]
        paths, transcripts = metadata.path, metadata.transcript

        labels = self._alphabet.get_batch_labels(transcripts)
        if self._features_store:
            features = self._read_features(paths)
        else:
            features = self._extract_features(paths)
        return features, labels


    def _read_features(self, paths):
        """ Read already prepared features from the store. """
        features = [self._features_store[path][:] for path in paths]
        return self._features_extractor.align(features)


    def _extract_features(self, paths):
        """ Extract features from the audio files (mono 16kHz). """
        return self._features_extractor.get_features_mfcc(files=paths)


    def on_epoch_end(self):
        """ Invoke methods at the end of the each epoch. """
        self.epoch += 1
        self._shuffle_indices()


    def _shuffle_indices(self):
        """ Set up the order of next batches """
        if self.epoch >= self._shuffle_after_epoch:
            np.random.shuffle(self.indices)
