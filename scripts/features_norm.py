import os
import argparse
import h5py
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Iterable
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
mem = joblib.Memory('cache')


def extract(args) -> np.ndarray:
    store_path, feature_index, path = args
    with h5py.File(store_path, mode='r') as store:
        data = store[path]
        return data[:, feature_index]


def extract_feature_stats(store_path: str, feature_index: int, iterator: Iterable, pool: Pool) -> Tuple[float, float]:
    iterator_with_args = [[store_path, feature_index, i] for i in iterator]
    results = pool.map(extract, iterator_with_args)     # Map do not accept Series
    array = np.hstack(results)
    return array.mean(), array.std()


@mem.cache
def extract_stats(store_path: str, processes: int = cpu_count()) -> np.ndarray:
    with pd.HDFStore(store_path, mode='r') as store:
        references = store['references']
        info = store['info']

    paths = references['path']
    features_num = info['numcep'][0]
    with Pool(processes) as pool:
        stats = [extract_feature_stats(store_path, feature_index, paths, pool)
                 for feature_index in range(features_num)]
    return stats


def copy_normalized_datasets(ref_store, store, stats: np.ndarray, iterator: Iterable):
    mean, std = stats.T
    for path in tqdm(iterator):
        data = ref_store[path][:]
        normalized_data = (data - mean) / std
        store.create_dataset(path, data=normalized_data)


def main(ref_store_path: str, word_min: int = 3, max_time: int = 7):
    """
    Normalize training examples on a per utterance basis in order to make the total power of each
    example consistent. From each frequency bin remove the global mean over the training set and
    divide by the global standard deviation, primarily so the inputs are well scaled during the
    early stages of training.
    """
    stats = np.array(extract_stats(ref_store_path))

    directory = os.path.dirname(ref_store_path)
    file_name = os.path.basename(ref_store_path)
    name, ext = file_name.split('.')
    store_path = os.path.join(directory, f'{name}-normalized.{ext}')

    with pd.HDFStore(ref_store_path, mode='r') as ref_store, \
            pd.HDFStore(store_path, mode='w') as store:

        store['info'] = ref_store['info']
        ref_references = ref_store['references']
        word_counts = ref_references.transcript.str.split().map(len)
        store['references'] = ref_references[(word_counts >= word_min) & (ref_references['size'] < max_time * 16000)]
        paths = store['references']['path']

    with h5py.File(ref_store_path, mode='r') as ref_store, \
            h5py.File(store_path, mode='r+') as store:

        copy_normalized_datasets(ref_store, store, stats, iterator=paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', required=True, help='File hdf5 keeps all created features')
    args = parser.parse_args()
    main(args.store)
