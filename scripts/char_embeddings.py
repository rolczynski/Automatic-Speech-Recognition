# Data source: PolEval (Language Models)
# 2018.poleval.pl/task3/task3_train.txt.gz
import io
import os
import itertools
from functools import partial
from multiprocessing.pool import Pool
from typing import Iterable
from typing import List
from typing import Tuple

import h5py
import numpy as np
from keras import Model
from keras.layers import Dense
from keras.layers import CuDNNLSTM
from keras.layers import Embedding
from keras.models import Sequential
from keras.utils import Sequence
from keras.utils import to_categorical
from tqdm import tqdm

from text import Alphabet


def load_tokens(fname: str, alphabet: Alphabet) -> Iterable[str]:
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    correct_chars = set(alphabet._str_to_label.keys())
    for line in fin:
        tokens = line.rstrip().split(' ')
        for token in tokens:
            is_correct = not set(token) - correct_chars
            if is_correct:
                yield token


def create_features(token, alphabet: Alphabet, context_size: int) -> List[Tuple[np.ndarray, np.int]]:
    labels = np.array([alphabet.blank_token, alphabet.label_from_string(' '),
                       *(alphabet.label_from_string(char) for char in token),
                       alphabet.label_from_string(' '), alphabet.blank_token])
    max_index = len(labels) - context_size
    samples = np.array([labels[i: i + context_size] for i in range(0, max_index + 1)], dtype=np.int8)
    return samples


def maybe_create_features(fname: str, tokens: Iterable[str], alphabet: Alphabet, context_size: int):
    if os.path.isfile(fname):
        return
    _create_features = partial(create_features, alphabet=alphabet, context_size=context_size)
    workers = os.cpu_count()
    with h5py.File(fname, mode='w') as store, Pool(processes=workers) as pool:
        index, tokens_chunk, pbar, = 0, True, tqdm()
        while tokens_chunk:
            tokens_chunk = list(itertools.islice(tokens, int(1e6)))
            features_chunk = pool.map(_create_features, tokens_chunk, chunksize=int(1e4))
            features = np.concatenate(features_chunk)
            store.create_dataset(name=str(index), data=features)
            index += 1
            pbar.update(int(1e6))
        store.attrs['chunks_num'] = index
        store.attrs['chunks_size'] = int(1e6)


class DataGenerator(Sequence):

    def __init__(self, fname: str, batch_size: int, num_classes: int):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.store = h5py.File(fname, mode='r')
        self.chunks_num = self.store.attrs['chunks_num']
        self.chunks_size = self.store.attrs['chunks_size']
        self.batches_per_chunk = self.chunks_size // self.batch_size

    def __len__(self):
        """ Denotes the number of batches per epoch. """
        return self.batches_per_chunk * self.chunks_num

    def __getitem__(self, next_index):
        """ Operator to get the batch data. """
        chunk_num, i = divmod(next_index, self.batches_per_chunk)
        data = self.store[str(chunk_num)][i*self.batch_size: (i+1)*self.batch_size]
        target_index = data.shape[1] // 2
        X, y = np.delete(data, target_index, axis=1), data[:, target_index]
        return X, to_categorical(y, self.num_classes)


def create_model(embed_size: int, lstm_size: int, input_length: int, voc_size: int) -> Model:
    model = Sequential()
    model.add(Embedding(voc_size, embed_size, input_length=input_length))
    model.add(CuDNNLSTM(lstm_size))
    model.add(Dense(voc_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def save_char_embeddings(fname: str, model: Model):
    embeddings, = model.get_layer('embedding_1').get_weights()
    embeddings[-1, :] = embeddings[:-1, :].mean(axis=0)
    np.savetxt(fname, embeddings.T)


def main():
    alphabet = Alphabet(alphabet_path)
    tokens = load_tokens(corpus_path, alphabet)
    maybe_create_features(features_path, tokens, alphabet, CONTEXT_SIZE)
    generator = DataGenerator(features_path, BATCH_SIZE, alphabet.size)
    model = create_model(EMBED_SIZE, LSTM_SIZE, input_length=CONTEXT_SIZE-1, voc_size=alphabet.size)
    model.fit_generator(generator, epochs=EPOCHS)
    model.save(model_path)
    save_char_embeddings(embeddings_path, model)


if __name__ == '__main__':
    root_dir = os.path.dirname(__file__)
    corpus_path = os.path.join(root_dir, 'task3_train.txt')
    features_path = os.path.join(root_dir, 'embed_features.hdf5')
    model_path = os.path.join(root_dir, 'model-36-128.hdf5')
    embeddings_path = os.path.join('data', 'char-embeddings-36.txt')
    alphabet_path = 'tests/models/test/alphabet.txt'
    EMBED_SIZE = 36
    LSTM_SIZE = 128
    BATCH_SIZE = 2048
    CONTEXT_SIZE = 5    # Should be odd
    EPOCHS = 2
    main()
