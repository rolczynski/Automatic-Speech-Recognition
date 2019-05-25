import os
import pytest
import numpy as np
from typing import List
from deepspeech import DeepSpeech, Configuration, FeaturesExtractor, Alphabet, DataGenerator, History
is_same = lambda A, B: all(np.array_equal(a, b) for a, b in zip(A, B))


@pytest.fixture
def model_dir() -> str:
    return 'tests/models/test/'


@pytest.fixture
def config_path(model_dir: str) -> str:
    return os.path.join(model_dir, 'configuration.yaml')


@pytest.fixture
def alphabet_path(model_dir: str) -> str:
    return os.path.join(model_dir, 'alphabet.txt')


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
def audio_file_paths() -> List[str]:
    return ['tests/data/audio/sent000.wav', 'tests/data/audio/sent001.wav']


@pytest.fixture
def generator(deepspeech: DeepSpeech) -> DataGenerator:
    return deepspeech.create_generator(file_path='tests/data/features.hdf5', source='from_prepared_features', batch_size=2)
