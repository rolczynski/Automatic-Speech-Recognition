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
from typing import List
Sample = namedtuple('Sample', ['size', 'transcript', 'features'])


def create_samples_from(segmented_phrases: pd.DataFrame, audio: np.ndarray, fs: int, **mfcc_params):
    """ Each phrase is represented as record which indicate how to split the sample. """

    def split_(array: np.ndarray, splitter: pd.DataFrame):
        start, end = int(splitter.start * fs), int(splitter.end * fs)
        audio_part = array[int(start): int(end)]
        return audio_part

    def extract_mfcc_from(data: np.ndarray):
        return python_speech_features.mfcc(data, samplerate=fs, **mfcc_params)

    for segmented_phrase in segmented_phrases.itertuples():
        audio_phrase = split_(audio, splitter=segmented_phrase)
        size = audio_phrase.size
        features = extract_mfcc_from(audio_phrase)
        transcript = ''.join(segmented_phrase.transcript)

        return Sample(size, transcript, features)


def generate_segmented_samples(audio_file: str, sentence_ctm: pd.DataFrame, max_words: int, **mfcc_params):
    """ Generate samples which are divided to sub-samples (better performance). """

    def read_(audio_file: str):
        """ Simple read .wav audio file. """
        fs, audio = wav.read(audio_file)
        return fs, audio

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

    fs, audio = read_(audio_file)
    phrase_groups = divide_(sentence_ctm, max_words)
    segmented_phrases = concatenate_(phrase_groups)
    samples = create_samples_from(segmented_phrases, audio, fs, **mfcc_params)
    return samples


def process_one_sentence(args):
    sentence_id, file_path, max_words, ctm = args
    try:
        sentence_ctm = ctm.get_group(sentence_id)
        return generate_segmented_samples(file_path, sentence_ctm, max_words, **mfcc_params)

    except Exception as exception:
        print(f'Error occurs: {file_path}')
        print(exception)


def main(store_path: str, audio_path: str, ctm_path: str, max_words: int, **mfcc_params):
    """ The script preprocess features before the training. It supports the sample segmentation."""

    def read_ctm(segmented_path: str) -> pd.DataFrame:
        """ Read dataframes. """
        ctm = pd.read_csv(segmented_path,
                          names=['sentence_id', 'nb', 'start', 'length', 'transcript'],
                          sep=' ', index_col=False)
        ctm.drop(columns=['nb'], inplace=True)
        ctm['end'] = ctm.start + ctm.length
        ctm = ctm.groupby('sentence_id')
        return ctm


    def save_in(store, samples: List[Sample], references: pd.DataFrame):
        """ Save the next dataset in the store. Track the crucial information using `references` table.
        The hdf5 file has the hierarchy with /-separator. Features are saved in the `features` group. """
        for sample in samples:
            index = len(references)
            path = f'features/{index}'
            references.loc[index] = path, sample.size, sample.transcript
            store.create_dataset(path, data=sample.features)

    metadata = pd.DataFrame(columns=['path', 'size', 'transcript'])
    audio = pd.read_csv(audio_path, sep=' ', names=['sentence_id', 'file_path'], index_col='sentence_id')
    ctm = read_ctm(ctm_path)

    available_workers = cpu_count()
    with h5py.File(store_path, mode='w') as store, Pool(processes=available_workers) as pool:

        parts = len(audio) // available_workers // 500   # This number can be reduce if the RAM is a limitation
        parts = 1 if parts < 1 else parts
        iterator = np.array_split(audio, parts)

        for chunk in tqdm(iterator):
            chunk_iterator = [list(row) + [max_words, ctm] for row in chunk.itertuples()]
            samples = pool.map(process_one_sentence, chunk_iterator)
            correct_samples = (sample for sample in samples if sample)
            save_in(store, correct_samples, references=metadata)

    metadata['transcript'] = metadata.transcript.str.lower()
    metadata.sort_values(by='size', inplace=True)
    info = pd.DataFrame.from_dict({k: [v] for k, v in mfcc_params.items()})

    with pd.HDFStore(store_path, mode='r+') as store:
        store.put('info', info)
        store.put('metadata', metadata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', required=True, help='File hdf5 keeps all created features')
    parser.add_argument('--audio', required=True, help='File csv keeps information where audio files are')
    parser.add_argument('--ctm', help='Text file contains words timing (used for samples division)')
    parser.add_argument('--max_words', type=int, help='Max count words in the sample')
    parser.add_argument('--winlen', default=0.025, type=float, help='The length of the analysis window in seconds. Default is 0.025s (25 milliseconds)')
    parser.add_argument('--winstep', default=0.01, type=float, help='The step between successive windows in seconds. Default is 0.01s (10 milliseconds)')
    parser.add_argument('--numcep', default=26, type=int, help='The number of cepstrum to return, default 26')
    parser.add_argument('--winfunc', default='', help='The analysis window to apply to each frame')
    args = parser.parse_args()

    mfcc_params = dict(
        winlen=args.winlen,
        winstep=args.winstep,
        numcep=args.numcep,
        winfunc=np.hamming if args.winfunc else lambda x: np.ones((x,)),
    )
    main(args.store, args.audio, args.ctm, args.max_words, **mfcc_params)
