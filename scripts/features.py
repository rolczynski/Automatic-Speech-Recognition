import argparse
import h5py
import pandas as pd
import numpy as np
import python_speech_features
import scipy.io.wavfile as wav
from collections import namedtuple
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
from tqdm import tqdm
from typing import List, Iterable
Sample = namedtuple('Sample', ['size', 'transcript', 'features'])


def generate_samples_from(segmented_phrases: pd.DataFrame, audio_data: np.ndarray, fs: int, mfcc_params: dict) -> Iterable[Sample]:
    """ Each phrase is represented as record which indicate how to split the sample. """
    def split(array: np.ndarray, splitter: pd.DataFrame, fs: int):
        start, end = int(splitter.start * fs), int(splitter.end * fs)
        audio_part = array[int(start): int(end)]
        return audio_part

    def extract_mfcc_from(data: np.ndarray, **mfcc_params):
        return python_speech_features.mfcc(data, **mfcc_params)

    for segmented_phrase in segmented_phrases.itertuples():
        audio_phrase = split(audio_data, splitter=segmented_phrase, fs=fs)
        size = audio_phrase.size
        features = extract_mfcc_from(audio_phrase, samplerate=fs, **mfcc_params)
        transcript = ''.join(segmented_phrase.transcript)
        yield Sample(size, transcript, features)


def read(audio_file: str):
    """ Simple read .wav audio file. """
    fs, audio = wav.read(audio_file)
    return fs, audio


def divide(sentence: pd.DataFrame, max_words: int) -> List[pd.DataFrame]:
    """ The sub-sentence division is well-balanced. """
    words_count, _ = sentence.shape
    phrases_count = words_count // max_words + 1
    groups_indices = np.array_split(sentence.index, phrases_count)
    groups = [sentence.loc[indices] for indices in groups_indices]
    return groups


def concatenate(groups: List[pd.DataFrame]) -> pd.DataFrame:
    """ Each group has a detailed information about words occurred in the phrase. """
    phrases = [[f'{group.audio_id.iloc[0]}_{index}',
                group.start.iloc[0],
                group.end.iloc[-1] - group.start.iloc[0],
                group.end.iloc[-1],
                ' '.join(group.token)]
               for index, group in enumerate(groups)]
    return pd.DataFrame(phrases, columns=['phrase_id', 'start', 'length', 'end', 'transcript'])


def create_sub_samples(file_path: str, segmented_audio: pd.DataFrame, max_words: int, mfcc_params: dict) -> List[Sample]:
    """ Generate samples which are divided to sub-samples (better performance). """
    fs, audio = read(file_path)
    phrase_groups = divide(segmented_audio, max_words)
    segmented_phrases = concatenate(phrase_groups)
    samples_generator = generate_samples_from(segmented_phrases, audio, fs, mfcc_params)
    samples = list(samples_generator)
    return samples


def process_one_sentence(args) -> List[Sample]:     # One parameter function is required by pool.map
    audio_id, file_path, transcript, max_words, segmented, mfcc_params = args
    try:
        segmented_audio = segmented.get_group(audio_id)
        return create_sub_samples(file_path, segmented_audio, max_words, mfcc_params)

    except Exception as exception:
        print(f'Error occurs: {file_path}')
        print(exception)


def save_in(store, samples: List[Sample], references: pd.DataFrame):
    """ Save the next dataset in the store. Track the crucial information using `references` table.
    The hdf5 file has the hierarchy with /-separator. Features are saved in the `features` group. """
    for sample in samples:
        index = len(references)
        path = f'features/{index}'
        references.loc[index] = path, sample.size, sample.transcript
        store.create_dataset(path, data=sample.features)


def extract_features(store_path: str, audio_path: str, segmented_path: str, max_words: int, mfcc_params: dict, workers: int = cpu_count()):
    """ The script preprocess features before the training. It supports the sample segmentation."""
    audio = pd.read_csv(audio_path, index_col='id')
    segmented_data = pd.read_csv(segmented_path)
    segmented = segmented_data.groupby('audio_id')

    references = pd.DataFrame(columns=['path', 'size', 'transcript'])
    with h5py.File(store_path, mode='w') as store, Pool(processes=workers) as pool:

        parts = len(audio) // workers // 500   # This number can be reduce if the RAM is a limitation
        parts = 1 if parts < 1 else parts
        iterator = np.array_split(audio, parts)

        for chunk in tqdm(iterator):
            chunk_iterator = [list(row) + [max_words, segmented, mfcc_params] for row in chunk.itertuples()]
            batch_samples = pool.map(process_one_sentence, chunk_iterator)
            correct_samples = (sample for samples in batch_samples if samples
                               for sample in samples if sample)
            save_in(store, correct_samples, references)

    references['transcript'] = references.transcript.str.lower()
    references.sort_values(by='size', inplace=True)
    info = pd.DataFrame.from_dict({k: [v] for k, v in mfcc_params.items()})

    with pd.HDFStore(store_path, mode='r+') as store:
        store.put('info', info)
        store.put('references', references)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', required=True, help='File hdf5 keeps all created features')
    parser.add_argument('--audio', required=True, help='File csv keeps information where audio files are')
    parser.add_argument('--segmented', required=True, help='Text file contains words timing (used for samples division)')
    parser.add_argument('--max_words', type=int, help='Max count words in the sample')
    parser.add_argument('--winlen', default=0.025, type=float, help='The length of the analysis window in seconds. Default is 0.025s (25 milliseconds)')
    parser.add_argument('--winstep', default=0.01, type=float, help='The step between successive windows in seconds. Default is 0.01s (10 milliseconds)')
    parser.add_argument('--numcep', default=26, type=int, help='The number of cepstrum to return, default 26')
    parser.add_argument('--winfunc', choices=['hamming', 'linear'], help='The analysis window to apply to each frame')
    args = parser.parse_args()

    args_mfcc_params = dict(
        winlen=args.winlen,
        winstep=args.winstep,
        numcep=args.numcep,
        winfunc=np.hamming if args.winfunc == 'hamming' else lambda x: np.ones((x,)),
    )
    extract_features(args.store, args.audio, args.segmented, args.max_words, args_mfcc_params)
