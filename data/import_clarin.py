# Dataset: http://mowa.clarin-pl.eu/korpusy/audio.tar.gz
import os
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from urllib import request
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
from keras.utils import Progbar

np.random.seed(40)
ROOT_DIR = os.path.join(os.getcwd(), 'clarin')
DATA_DIR = os.path.join(ROOT_DIR, 'source/audio')

TRAIN_SESSIONS_URL = 'https://github.com/danijel3/ClarinStudioKaldi/raw/master/local_clarin/train.sessions'
TEST_SESSIONS_URL = 'https://github.com/danijel3/ClarinStudioKaldi/raw/master/local_clarin/test.sessions'


def __read_and_splitlines(url):
    """ Read the train / test division among sessions """
    session_list = request.urlopen(url).read().decode('utf-8').splitlines()
    return {session_name: True for session_name in session_list}


def __filter_wav_txt_pairs(files):
    """ Filter and retrieve pairs .wav sample and .txt transcript """
    table = {}
    for file in files:
        name, extension = file.split('.')
        if 'sent' not in name:
            continue

        if extension not in ['txt', 'wav']:
            Warning('Unrecognised file extension inside the folder: {extension}')

        if name not in table:
            table[name] = 1
        else:
            table[name] += 1

    return [(name+'.wav', name+'.txt') for name, v in table.items() if v == 2]


def __add_paths_to_names(pairs, dir):
    """ Add paths to transcript file names """
    return [(os.path.join(dir, wav), os.path.join(dir, txt))
            for wav, txt in pairs]


def __process_pair_wav_txt(pair):
    """ Method designed to compute in parallel threads - O/I overload """
    wav_name, txt_name = pair
    with open(txt_name) as txt_file:
        transcript = txt_file.read().strip()

    fs, audio = wav.read(wav_name)
    assert fs == 16000
    return [wav_name, audio.size, transcript]


def __sort_and_save(*args):
    for name, df in args:
        df.sort_values(by='size').to_csv(os.path.join(ROOT_DIR, name), index=False)


if __name__ == '__main__':
    train_sessions = __read_and_splitlines(url=TRAIN_SESSIONS_URL)
    test_sessions = __read_and_splitlines(url=TEST_SESSIONS_URL)

    column_names = ['file_name', 'size', 'transcript']
    train = pd.DataFrame(columns=column_names)
    test = pd.DataFrame(columns=column_names)
    target_steps = len(train_sessions) + len(test_sessions)
    progress = Progbar(target=target_steps)

    available_workers = cpu_count()
    with Pool(processes=available_workers) as pool:

        for step, (root, dirs, files) in enumerate(os.walk(DATA_DIR)):
            if not files:
                continue

            pairs = __filter_wav_txt_pairs(files)
            pairs_with_paths = __add_paths_to_names(pairs, dir=root)
            chunk = pool.map(__process_pair_wav_txt, pairs_with_paths)
            chunk_df = pd.DataFrame(chunk, columns=column_names)

            session_name = os.path.basename(root)
            if session_name in train_sessions:
                train = train.append(chunk_df, ignore_index=True)
            elif session_name in test_sessions:
                test = test.append(chunk_df, ignore_index=True)
            else:
                Warning(f'Unrecognised session name: {session_name}')

            progress.update(step)

    intersection = pd.merge(train, test, how='inner')
    assert intersection.size == 0, "Train and test sets have to be disjoint"

    train_size, _ = train.shape
    dev_indices = np.random.choice(train.index, size=train_size // 10)
    dev = train.iloc[dev_indices]
    train = train.drop(dev_indices)

    __sort_and_save(('train.csv', train), ('dev.csv', dev), ('test.csv', test))
    print('Successfully complete')
