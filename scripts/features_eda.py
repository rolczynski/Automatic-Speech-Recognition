import argparse
import h5py
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from typing import Iterable


def save_histogram(x: Iterable, file_name: str, label: str, fig_size=(8, 8), **kwargs):
    fig, ax = plt.subplots(figsize=fig_size)
    ax = sns.distplot(x, **kwargs)
    ax.set_xlabel(label)
    fig.savefig(file_name, format='svg', dpi=1200)


def main(store_path: str):
    file_name = os.path.basename(store_path)
    name, extension = file_name.split('.')

    with pd.HDFStore(store_path, mode='r') as store:
        metadata = store['metadata']
        sizes = metadata['size'].astype(int)
        times = pd.Series(sizes / 16e3, name='time')    # Divide by sample rate
        save_histogram(times, file_name=f'images/{name}-histogram-time.svg',
                       label='Time [s]', fig_size=(10, 6))

        transcripts = metadata['transcript']
        word_counts = transcripts.str.split().map(len)
        save_histogram(word_counts, file_name=f'images/{name}-histogram-word_counts.svg',
                       label='Words count', fig_size=(10, 6), kde=False)

    paths = metadata['path']
    with h5py.File(store_path, mode='r') as store:
        lens = [len(store[path]) for path in paths]     # Should looks as scaled time histogram
        save_histogram(lens, file_name=f'images/{name}-histogram-steps.svg',
                       label='Time steps', fig_size=(10, 6))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', required=True, help='File hdf5 keeps all created features')
    args = parser.parse_args()
    main(args.store)
