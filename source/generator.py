import numpy as np
import pandas as pd
from keras.utils import Sequence
from source import audio, text
from source.utils import cache_features_to_files


class DataGenerator(Sequence):
    """
    Generates data for Keras

    Every `Sequence` must implement the `__getitem__` and the `__len__`
    methods. If you want to modify your dataset between epochs you may
    implement `on_epoch_end`. The method `__getitem__` should return a
    complete batch.

    `Sequence` are a safer way to do multiprocessing. This structure
    guarantees that the network will only train once on each sample per epoch
    which is not the case with generators.

    Sequence implementation:
    https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py

    Example:
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
    """
    def __init__(self, name, csv_path, alphabet, shuffle_after_epoch=1,
                 sort_source=True, max_length=False, batch_size=30,
                 cache=True, cache_dir=None, cache_files_count=None,
                 cache_verbose=False):
        self._epoch = 0
        self._name = name
        self._csv_path = csv_path
        self._alphabet = alphabet
        self._batch_size = batch_size
        self._sort_source = sort_source
        self._max_length = max_length
        self._shuffle_after_epoch = shuffle_after_epoch

        self._cache = cache
        self._cache_dir = cache_dir
        self._cache_files_count = cache_files_count
        self._cache_verbose = cache_verbose

        # Wrap method and keep all features in the cache files
        if self._cache:
            self.__get_batch = cache_features_to_files(
                self._name, self._cache_dir, self._cache_files_count
            )(self.__get_batch)

        self.__set_data_from_csv()
        self.indices = np.arange(len(self))
        self.__shuffle_indices()


    def __set_data_from_csv(self):
        """ Process csv files: `file_name`, `size`, `transcript` """
        data = pd.read_csv(
            self._csv_path,
            names=['file_name', 'size', 'transcript'],
            sep=',',
            encoding='utf-8',
            header=0
        )

        if self._max_length:
            correct_records = data.transcript.map(len) <= self._max_length
            data = data[correct_records]

        if self._sort_source:
            data.sort_values(by='size', ascending=True, inplace=True)

        self._data = data


    def __len__(self):
        """ Denotes the number of batches per epoch """
        return int(np.floor(len(self._data.index) / self._batch_size))


    def __getitem__(self, next_index):
        """ Operator to get batch data - can be cached """
        index = self.indices[next_index]
        return self.__get_batch(index)


    def __get_batch(self, index):
        """ Generate one batch of data """
        start, end = index*self._batch_size, (index+1)*self._batch_size
        batch_data = self._data[start:end]
        return self.__generate_data(batch_data)


    def __generate_data(self, data):
        """ Extract features from the source batch """
        features = audio.get_features_mfcc(data.file_name)
        labels = text.get_batch_labels(data.transcript, self._alphabet)
        return features, labels


    def on_epoch_end(self):
        """ Invoke methods on the end of each epoch """
        self._epoch += 1
        self.__shuffle_indices()
        self.__cache_summary()


    def __shuffle_indices(self):
        """ Set up the order of next batches """
        if self._epoch >= self._shuffle_after_epoch:
            np.random.shuffle(self.indices)


    def __cache_summary(self):
        """ Summary to check the cache usage """
        if self._cache and self._cache_verbose:
            print('\n' + str(self.__get_batch.cache_info()))
