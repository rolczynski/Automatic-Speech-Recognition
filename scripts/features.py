import argparse
import h5py
import pandas as pd
from source import audio
from utils import chdir


def main(store_path, audio_indicators_path):
    audio_indicators = pd.read_csv(audio_indicators_path)
    source_indicators = pd.DataFrame(columns=['name', 'transcript'])

    with h5py.File(store_path, mode='w') as store:
        for index, (audio_file_path, transcript) in audio_indicators.iterrows():

            name = f'features/{index}'
            features = audio.make_mfcc(audio_file_path)
            store.create_dataset(name, data=features)

            source_indicators.loc[index] = name, transcript

    with pd.HDFStore(store_path, mode='r+') as store:
        store.put('source_indicators', source_indicators)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--store_path', required=True, help='The hdf5 file keeps all created features')
    parser.add_argument('--audio_indicators', required=True, help='The csv file keeps information where audio files are')
    args = parser.parse_args()

    chdir(to='ROOT')
    main(args.store_path, args.audio_indicators)
