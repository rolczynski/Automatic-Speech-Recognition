import numpy as np
from typing import List
from keras.engine.training import Model
from source.utils import chdir
from source.deepspeech import DeepSpeech, Configuration, FeaturesExtractor, Alphabet
is_same = lambda A, B: all(np.array_equal(a, b) for a, b in zip(A, B))
chdir(to='ROOT')


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


def test_get_callbacks(test_dir: str, config: Configuration):
    model = DeepSpeech.get_model(**config.model, is_gpu=False)
    callbacks = DeepSpeech.get_callbacks(home_dir=test_dir, configurations=config.callbacks, model=model)
    assert len(callbacks) == 2


def test_compile_model(config: Configuration):
    model = DeepSpeech.get_model(**config.model, is_gpu=False)
    optimizer = DeepSpeech.get_optimizer(**config.optimizer)
    loss = DeepSpeech.get_loss()
    DeepSpeech.compile_model(model, optimizer, loss)
    assert model._is_compiled


def test_get_features(deepspeech: DeepSpeech, audio_file_paths: List[str]):
    features = deepspeech.get_features(audio_file_paths)
    assert features.shape == (4, 299, 80)


def test_get_labels_and_get_transcripts(deepspeech: DeepSpeech):
    bad_transcripts = ['to jest je$st!', 'test']
    correct_transcripts = ['to jest jest', 'test']
    labels = deepspeech.get_labels(bad_transcripts)
    assert labels.dtype == np.int64
    assert labels.shape == (2, 12)
    assert labels[1, 4] == deepspeech.alphabet.blank_token
    transformed_transcripts = deepspeech.get_transcripts(labels)
    assert transformed_transcripts == correct_transcripts


def test_predict():
    # Load pretrained model and predict
    pass


def test_decode():
    # Load pretrained model and decode
    pass


def test_call():
    pass
