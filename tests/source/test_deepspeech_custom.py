import os
import pytest
import numpy as np
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)    # Tensorflow warnings
from deepspeech import DeepSpeech, Configuration


@pytest.fixture
def model_dir() -> str:
    return 'tests/models/test_custom/'


@pytest.fixture
def config_path(model_dir: str) -> str:
    return os.path.join(model_dir, 'configuration.yaml')


@pytest.fixture
def alphabet_path(model_dir: str) -> str:
    return os.path.join(model_dir, 'alphabet.txt')


@pytest.fixture
def deepspeech(config_path: str, alphabet_path: str) -> DeepSpeech:
    return DeepSpeech.construct(config_path, alphabet_path)


@pytest.fixture
def config(config_path) -> Configuration:
    return DeepSpeech.get_configuration(config_path)


def test_compile_model(config: Configuration, deepspeech: DeepSpeech):
    embeddings_path = config.model['embeddings']['file']
    char_embeddings = np.loadtxt(embeddings_path)
    assert char_embeddings.shape == (128, 36)
    softmax = deepspeech.model.layers[-1]  # The last TimeDistributed layer contains Softmax layer
    softmax_weights, = softmax.layer.get_weights()
    assert np.array_equal(char_embeddings, softmax_weights)
    softmax = deepspeech.distributed_model.get_layer('DeepSpeech').layers[-1]
    softmax_weights, = softmax.layer.get_weights()
    assert np.array_equal(char_embeddings, softmax_weights)
