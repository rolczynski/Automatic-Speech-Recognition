from typing import Tuple, List

import numpy as np
import pandas as pd
import h5py

from decoding import BestPathDecoder, CTCDecoder, naive_decode
from text import Alphabet
from metric import get_metrics


ACTIVATION_PATH = '../scripts/evaluation-clarin-without-activations.hdf5'
ALPHABET = Alphabet(file_path='../models/pl/alphabet.txt')


def load_data(file_path: str, limit=None) -> Tuple[pd.Series, pd.Series, List[np.ndarray]]:
    with pd.HDFStore(file_path, mode='r') as hdfs_store:
        transcripts = hdfs_store['references']['transcript']
        cer = hdfs_store['references']['cer']
    with h5py.File(file_path, 'r') as file:
        activations = file['outputs/1']
        activations = list(activation[:] for activation in activations.values())
    if limit is not None:
        transcripts, cer, activations = transcripts[:limit], cer[:limit], activations[:limit]
    return transcripts, cer, activations


transcripts, cer, activations = load_data(file_path=ACTIVATION_PATH, limit=10)


decoders = [
    BestPathDecoder(config={'alphabet': ALPHABET}),
    CTCDecoder(language_model=None, config={'alphabet': ALPHABET}),
]
for decoder in decoders:
    print()
    print(decoder)
    print()
    for probs, transcript in zip(activations, transcripts):
        decoded_transcript, loss = decoder.decode(probs)
        metrics = get_metrics([decoded_transcript], [transcript])
        print(transcript, '<->', decoded_transcript, list(metrics))

