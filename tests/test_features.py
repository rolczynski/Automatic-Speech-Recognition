import os
import h5py
import numpy as np
import pandas as pd
from typing import Iterable, Callable


store_paths = [os.path.join('../data', file_name)
               for file_name in os.listdir('../data') if '.hdf5' in file_name]


def for_each(iterator: Iterable) -> Callable:
    def for_each_element(func) -> Callable:
        def wrapper(*args):
            for i in iterator:
                func(i, *args)
        return wrapper
    return for_each_element


@for_each(store_paths)
def test_store(store_path: str):
    with pd.HDFStore(store_path, mode='r') as store:
        info = store['info']
        assert list(info.columns) == ['winlen', 'winstep', 'numcep', 'winfunc']

        metadata = store['metadata']
        assert list(metadata.columns) == ['path', 'size', 'transcript']

    random_record = metadata.sample()
    record_path, = random_record.path
    with h5py.File(store_path, mode='r') as store:
        sample = store[record_path]
        assert sample.dtype == np.float
        assert sample.ndim == 2
