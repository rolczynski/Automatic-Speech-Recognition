import os
import pytest
import numpy as np
from typing import List

from keras import Model

from deepspeech import DeepSpeech, Configuration, Alphabet, DataGenerator
is_same = lambda A, B: all(np.array_equal(a, b) for a, b in zip(A, B))


@pytest.fixture
def test_dir() -> str:
    return 'tests'


@pytest.fixture
def config_path(test_dir) -> str:
    return os.path.join(test_dir, 'configuration.yaml')


@pytest.fixture
def alphabet_path(test_dir) -> str:
    return os.path.join(test_dir, 'alphabet.txt')


@pytest.fixture
def config(config_path) -> Configuration:
    return DeepSpeech.get_configuration(config_path)


@pytest.fixture
def alphabet(alphabet_path: str) -> Alphabet:
    return DeepSpeech.get_alphabet(alphabet_path)


@pytest.fixture
def deepspeech(config_path: str, alphabet_path: str) -> DeepSpeech:
    return DeepSpeech.construct(config_path, alphabet_path)


@pytest.fixture
def model(deepspeech: DeepSpeech) -> Model:
    return deepspeech.compiled_model


@pytest.fixture
def audio_file_paths() -> List[str]:
    return ['tests/data/audio/sent000.wav', 'tests/data/audio/sent001.wav',
            'tests/data/audio/sent002.wav', 'tests/data/audio/sent003.wav']


@pytest.fixture
def generator(deepspeech: DeepSpeech) -> DataGenerator:
    return DataGenerator.from_audio_files(file_path='tests/data/audio.csv',
                                          alphabet=deepspeech.alphabet,
                                          features_extractor=deepspeech.features_extractor,
                                          batch_size=2)
