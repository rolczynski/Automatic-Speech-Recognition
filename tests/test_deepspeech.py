import os
import shutil
import pytest
import numpy as np
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)    # Tensorflow warnings
from typing import List
from keras.engine.training import Model
from deepspeech import DeepSpeech, Configuration, FeaturesExtractor, Alphabet, DataGenerator, History


@pytest.fixture
def config_path() -> str:
    return 'test_model/configuration.yaml'


@pytest.fixture
def alphabet_path() -> str:
    return 'test_model/alphabet.txt'


@pytest.fixture
def config(config_path) -> Configuration:
    return DeepSpeech.get_configuration(config_path)


@pytest.fixture
def alphabet(alphabet_path: str) -> Alphabet:
    return DeepSpeech.get_alphabet(alphabet_path)


def test_config(config: Configuration):
    required_attrs = ['features_extractor', 'model', 'callbacks', 'optimizer', 'decoder']
    assert all(hasattr(config, attr) for attr in required_attrs)


def test_get_model(config: Configuration):
    model = DeepSpeech.get_model(**config.model, is_gpu=False)
    assert type(model) == Model
    new_model = DeepSpeech.get_model(**config.model, is_gpu=False)
    assert is_same(model.get_weights(), new_model.get_weights())    # Test random seed


def test_get_features_extractor(config: Configuration):
    features_extractor = DeepSpeech.get_features_extractor(**config.features_extractor)
    assert type(features_extractor) == FeaturesExtractor


def test_get_decoder(config: Configuration, alphabet: Alphabet):
    model = DeepSpeech.get_model(**config.model, is_gpu=False)
    decoder = DeepSpeech.get_decoder(alphabet=alphabet, model=model, **config.decoder)
    assert callable(decoder)


def test_get_callbacks(config: Configuration):
    callbacks = DeepSpeech.get_callbacks(home_dir='test_model', configurations=config.callbacks)
    assert len(callbacks) == 6


def test_compile_model(config: Configuration):
    model = DeepSpeech.get_model(**config.model, is_gpu=False)
    optimizer = DeepSpeech.get_optimizer(**config.optimizer)
    loss = DeepSpeech.get_loss()
    compiled_model = DeepSpeech.compile_model(model, optimizer, loss, gpus=[])
    assert compiled_model._is_compiled


@pytest.fixture
def deepspeech(config_path: str, alphabet_path: str) -> DeepSpeech:
    return DeepSpeech.construct(config_path, alphabet_path)


@pytest.fixture
def audio_file_paths() -> List[str]:
    return ['test_data/audio/0000.wav', 'test_data/audio/0001.wav']


def test_get_features(deepspeech: DeepSpeech, audio_file_paths: List[str]):
    features = deepspeech.get_features(audio_file_paths)
    assert features.shape == (2, 132, 26)


def test_get_labels_and_get_transcripts(deepspeech: DeepSpeech):
    bad_transcripts = ['to jest je$st!', 'test']
    correct_transcripts = ['to jest jest', 'test']
    labels = deepspeech.get_labels(bad_transcripts)
    assert labels.dtype == np.int64
    assert labels.shape == (2, 12)
    assert labels[1, 4] == deepspeech.alphabet.blank_token
    transformed_transcripts = deepspeech.get_transcripts(labels)
    assert transformed_transcripts == correct_transcripts


def test_create_generator_from_audio_files(deepspeech: DeepSpeech):
    generator = deepspeech.create_generator(file_path='test_data/audio.csv', source='from_audio_files', batch_size=5)
    assert len(generator) == 2
    X, y = generator[0]
    assert X.shape == (5, 180, 26)
    assert y.shape == (5, 44)


def test_create_generator_from_prepared_features(deepspeech: DeepSpeech):
    generator = deepspeech.create_generator(file_path='test_data/features.hdf5', source='from_prepared_features', batch_size=5)
    assert len(generator) == 2
    X, y = generator[0]
    assert X.shape == (5, 278, 26)
    assert y.shape == (5, 28)


@pytest.fixture
def generator(deepspeech: DeepSpeech) -> DataGenerator:
    return deepspeech.create_generator(file_path='test_data/features.hdf5', source='from_prepared_features', batch_size=5)


def test_fit(deepspeech: DeepSpeech, generator: DataGenerator, config_path: str, alphabet_path: str):
    # Test save best weights (overwrite the best result)
    deepspeech.save('test_model/test_weights.hdf5')
    distributed_weights = deepspeech.distributed_model.get_weights()
    model_checkpoint = deepspeech.callbacks[2]
    model_checkpoint.best_result = 0
    model_checkpoint.best_weights_path = 'test_model/test_weights.hdf5'

    history = deepspeech.fit(train_generator=generator, dev_generator=generator, epochs=2)
    assert type(history) == History

    # Test the returned model has `test_weights`
    deepspeech_weights = deepspeech.model.get_weights()
    new_deepspeech = DeepSpeech.construct(config_path, alphabet_path)
    new_deepspeech.load(model_checkpoint.best_weights_path)
    new_deepspeech_weights = new_deepspeech.model.get_weights()
    assert is_same(deepspeech_weights, new_deepspeech_weights)

    # Test that distributed model appropriate update weights
    new_distributed_weights = deepspeech.distributed_model.get_weights()
    assert is_same(distributed_weights, new_distributed_weights)


def test_predict():
    # Load pretrained model and predict
    pass


def test_decode():
    # Load pretrained model and decode
    pass


def test_save_load(deepspeech: DeepSpeech, config: Configuration, config_path: str, alphabet_path: str):
    weights_path = os.path.join('test_model', 'weights.hdf5')
    model_weights = deepspeech.model.get_weights()
    deepspeech.save(weights_path)

    new_deepspeech = DeepSpeech.construct(config_path, alphabet_path)
    new_deepspeech.model = deepspeech.get_model(**config.model, is_gpu=False, random_state=123)
    new_model_weights = new_deepspeech.model.get_weights()
    assert not is_same(model_weights, new_model_weights)

    new_deepspeech.load(weights_path)
    new_model_weights = new_deepspeech.model.get_weights()
    assert is_same(model_weights, new_model_weights)


def test_call(deepspeech: DeepSpeech, audio_file_paths: List[str]):
    # sentences = deepspeech(audio_file_paths)
    # assert len(sentences) == 2
    pass


def test_utils_load():
    from utils import load, get_root_dir
    deepspeech = load('pl')                                 # Call via: model name
    assert type(deepspeech) == DeepSpeech

    root_dir = get_root_dir()
    model_dir = os.path.join(root_dir, 'models', 'pl')
    deepspeech_dir = load(model_dir)                        # or model directory
    assert type(deepspeech_dir) == DeepSpeech


def test_end():
    """ Clean the directory at the end. """
    os.rename('test_model/alphabet.txt', 'alphabet.txt')
    os.rename('test_model/configuration.yaml', 'configuration.yaml')
    shutil.rmtree('test_model')
    os.mkdir('test_model')
    os.rename('alphabet.txt', 'test_model/alphabet.txt')
    os.rename('configuration.yaml', 'test_model/configuration.yaml')


is_same = lambda A, B: all(np.array_equal(a, b) for a, b in zip(A, B))
