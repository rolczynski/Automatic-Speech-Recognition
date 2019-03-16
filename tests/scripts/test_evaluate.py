import os
from typing import List, Tuple, Iterable
import pytest
import h5py
import pandas as pd
import numpy as np
from keras.models import Model
from source.utils import load
from source.deepspeech import DeepSpeech, DataGenerator
from source.metric import Metric, get_metrics
from scripts.features import extract_features
from scripts.evaluate import get_activations_function, save_in, evaluate
np.random.seed(123)


@pytest.fixture
def deepspeech() -> DeepSpeech:
    return load('tests/models/test')


@pytest.fixture
def model(deepspeech: DeepSpeech) -> Model:
    return deepspeech.model


def test_start():
    """ Evaluation depends on the features store. """
    extract_features('tests/data/features.hdf5', 'tests/data/audio.csv', 'tests/data/segmented.csv', 7,
                     dict(winlen=0.032, winstep=0.02, numcep=26, winfunc=np.hamming))


@pytest.fixture
def features_store_path() -> str:
    return 'tests/data/features.hdf5'


@pytest.fixture
def generator(features_store_path: str, deepspeech: DeepSpeech) -> DataGenerator:
    return deepspeech.create_generator(features_store_path, source='from_prepared_features', batch_size=3)


@pytest.fixture
def batch(generator: DataGenerator) -> Tuple[np.ndarray, np.ndarray]:
    return generator[0]


def test_get_activations_function(model: Model, batch: Tuple[np.ndarray, np.ndarray]):
    X, y = batch
    get_activations = get_activations_function(model)
    *activations, y_hat = get_activations([X, 0])
    assert len(activations) == len(model.layers)-2    # Without input and output layer
    batch, time, features = X.shape
    assert activations[0].shape == (batch, time, features, 1)
    assert activations[1].shape == (batch, time+2*7, features, 1)
    assert activations[2].shape == (batch, time, 1, 64)
    assert all(activation.shape == (batch, time, 64) for activation in activations[3:])
    assert y_hat.shape == (3, 94, 36)


@pytest.fixture
def layer_outputs(model: Model, batch: Tuple[np.ndarray, np.ndarray]) -> List[np.ndarray]:
    X, y = batch
    get_activations = get_activations_function(model)
    *activations, y_hat = get_activations([X, 0])
    return [X, *activations, y_hat]


@pytest.fixture
def metrics(deepspeech: DeepSpeech, layer_outputs: List[np.ndarray], batch: Tuple[np.ndarray, np.ndarray]) -> Iterable[Metric]:
    X, y = batch
    y_hat = layer_outputs[-1]
    predict_sentences = deepspeech.decode(y_hat)
    true_sentences = deepspeech.get_transcripts(y)
    return get_metrics(sources=predict_sentences, destinations=true_sentences)


@pytest.fixture
def store_path() -> str:
    return 'tests/models/test/evaluation.hdf5'


@pytest.fixture
def references() -> pd.DataFrame:
    return pd.DataFrame(columns=['sample_id', 'transcript', 'prediction', 'wer', 'cer']).set_index('sample_id')


def test_save_in(store_path: str, layer_outputs: List[np.ndarray], metrics: Iterable[Metric], references: pd.DataFrame):
    with h5py.File(store_path, mode='w') as store:
        metrics = list(metrics)
        save_in(store, layer_outputs, metrics, references)
        assert len(references) == 3
        assert all(references.columns.values == np.array(['transcript', 'prediction', 'wer', 'cer']))
        save_in(store, layer_outputs, metrics, references)
        assert len(references) == 6

    sample_id = np.random.choice(references.index)
    with h5py.File(store_path, mode='r') as store:
        output_index = 1
        sample_X = store[f'outputs/{output_index}/{sample_id}']
        assert sample_X.shape == (94, 26, 1), 'Input layer and one additional dim'


def test_evaluate(deepspeech: DeepSpeech, generator: Iterable, store_path: str) -> pd.DataFrame:
    metrics = evaluate(deepspeech, generator, save_activations=True, store_path=store_path)
    with pd.HDFStore(store_path, mode='r') as store:
        references = store['references']

    assert len(references) == len(metrics) == 12
    assert all(references.columns.values == np.array(['transcript', 'prediction', 'wer', 'cer']))


def test_end(features_store_path: str, store_path: str):
    os.remove(features_store_path)
    os.remove(store_path)
