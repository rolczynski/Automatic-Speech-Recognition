import os
import argparse
import h5py
import pandas as pd
import numpy as np
import python_speech_features
import scipy.io.wavfile as wav
from functools import partial
from collections import namedtuple
from typing import Callable
import sys  # Add `source` module (needed when it runs via terminal)
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from source.utils import chdir

Sample = namedtuple('Sample', ['path', 'size', 'transcript', 'features'])


def read_dataframes(audio_path: str, segmented_path=None):
    """ Read dataframes. """
    audio_source = pd.read_csv(audio_path, usecols=['name', 'transcript'])
    if segmented_path:
        cmt = pd.read_csv(segmented_path,
                          names=['sentence_id', 'nb', 'start', 'length', 'transcript'],
                          sep=' ', index_col=False)
        cmt.drop(columns=['nb'], inplace=True)
        cmt['end'] = cmt.start + cmt.length
        cmt = cmt.groupby('sentence_id')
        return audio_source, cmt
    else:
        return audio_source, None


def read_(audio_file: str):
    """ Simple read .wav audio file. """
    fs, audio = wav.read(audio_file)
    return fs, audio


def get_next_path(metadata: pd.DataFrame):
    """ The hdf5 file has the hierarchy with /-separator. Features are saved in the `features` group. """
    index = len(metadata)
    path = f'features/{index}'
    return path


def retrieve_sentence_id(audio_file: str):
    """ The sentences are divided into sessions. This method helps to encode it. """
    file = os.path.basename(audio_file)
    name, ext = file.split('.')
    directory_path = os.path.dirname(audio_file)
    directory = os.path.basename(directory_path)
    return f'{directory}_{name}'


def divide_(sentence: pd.DataFrame, max_words: int):
    """ The sub-sentence division is well-balanced. """
    words_count, _ = sentence.shape
    phrases_count = words_count // max_words + 1
    groups_indices = np.array_split(sentence.index, phrases_count)
    groups = [sentence.loc[indices] for indices in groups_indices]
    return groups


def concatenate_(groups: list):
    """ Each group has a detailed information about words occurred in the phrase. """
    phrases = [[f'{group.sentence_id.iloc[0]}_{index}',
                group.start.iloc[0],
                group.end.iloc[-1] - group.start.iloc[0],
                group.end.iloc[-1],
                ' '.join(group.transcript)]
               for index, group in enumerate(groups)]
    return pd.DataFrame(phrases, columns=['phrase_id', 'start', 'length', 'end', 'transcript'])


def create_samples_from(segmented_phrases: pd.DataFrame, audio: np.ndarray, fs: int, next_path: Callable):
    """ Each phrase is represented as record which indicate how to split the sample. """

    def split_(array: np.ndarray, splitter: pd.DataFrame):
        start, end = int(splitter.start * fs), int(splitter.end * fs)
        audio_part = array[int(start): int(end)]
        return audio_part

    def extract_mfcc_from(data: np.ndarray):
        return python_speech_features.mfcc(data, samplerate=fs, numcep=26)


    for segmented_phrase in segmented_phrases.itertuples():
        path = next_path()
        audio_phrase = split_(audio, splitter=segmented_phrase)
        size = audio_phrase.size
        features = extract_mfcc_from(audio_phrase)
        transcript = ''.join(segmented_phrase.transcript)

        yield Sample(path, size, transcript, features)


def save_in(store, data: Sample, references: pd.DataFrame):
    """ Save the next dataset in the store. Track the crucial information using `references` table. """
    references.loc[len(references)] = data.path, data.size, data.transcript
    store.create_dataset(data.path, data=data.features)


def generate_segmented_samples(audio_file: str, segmented_source: pd.DataFrame, max_words: int, reference: pd.DataFrame):
    """ Generate samples which are divided to sub-samples (better performance). """
    fs, audio = read_(audio_file)
    sentence_id = retrieve_sentence_id(audio_file)
    segmented_sentence = segmented_source.get_group(sentence_id)
    phrase_groups = divide_(segmented_sentence, max_words)
    segmented_phrases = concatenate_(phrase_groups)

    next_path = partial(get_next_path, reference)
    samples = create_samples_from(segmented_phrases, audio, fs, next_path)
    return samples


def generate_sample(audio_file: str, transcript: str, reference: pd.DataFrame):
    """ Generate the sample which contains precomputed features. """
    fs, audio = read_(audio_file)
    path = get_next_path(reference)
    size = audio.size
    features = python_speech_features.mfcc(audio, samplerate=fs, numcep=26)
    sample = Sample(path, size, transcript, features)
    return sample


def main(store_path: str, audio_path: str, segmented_path: str, max_words: int):
    """ The script preprocess features before the training. It supports the sample segmentation."""
    metadata = pd.DataFrame(columns=['path', 'size', 'transcript'])
    audio_source, segmented_source = read_dataframes(audio_path, segmented_path)

    errors = 0
    with h5py.File(store_path, mode='w') as store:
        for index, (audio_file, transcript) in audio_source.iterrows():
            try:
                if bool(segmented_source) and max_words:
                    samples = generate_segmented_samples(audio_file, segmented_source, max_words, metadata)
                else:
                    samples = [generate_sample(audio_file, transcript, metadata)]
            except:
                errors += 1
                print(f'can not find ({errors}): {audio_file}')
                continue

            for sample in samples:
                save_in(store, sample, references=metadata)

            if index % 1e3 == 0:
                print(index)

    metadata['transcript'] = metadata.transcript.str.lower()
    metadata.sort_values(by='size', inplace=True)

    with pd.HDFStore(store_path, mode='r+') as store:
        store.put('metadata', metadata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', required=True, help='File hdf5 keeps all created features')
    parser.add_argument('--audio', required=True, help='File csv keeps information where audio files are')
    parser.add_argument('--segmented', help='Text file contains words timing (used for samples division)')
    parser.add_argument('--max_words', type=int, help='Max count words in the sample')
    args = parser.parse_args()

    chdir(to='ROOT')
    main(args.store, args.audio, args.segmented, args.max_words)
