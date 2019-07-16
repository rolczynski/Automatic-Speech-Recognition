import os
import shutil
import numpy as np
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)    # Tensorflow warnings
from typing import List
from keras.engine.training import Model
from source.utils import chdir
from source.deepspeech import DeepSpeech, Configuration, FeaturesExtractor, Alphabet, DataGenerator, History
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
    callbacks = DeepSpeech.get_callbacks(home_dir=test_dir, configurations=config.callbacks)
    assert len(callbacks) == 2


def test_compile_model(config: Configuration):
    model = DeepSpeech.get_model(**config.model, is_gpu=False)
    optimizer = DeepSpeech.get_optimizer(**config.optimizer)
    loss = DeepSpeech.get_loss()
    compiled_model = DeepSpeech.compile_model(model, optimizer, loss, gpus=[])
    assert compiled_model._is_compiled


def test_get_features(deepspeech: DeepSpeech, audio_file_paths: List[str]):
    features = deepspeech.get_features(audio_file_paths)
    assert features.shape == (4, 1477, 80)


def test_get_labels_and_get_transcripts(deepspeech: DeepSpeech):
    bad_transcripts = ['to jest je$st!', 'test']
    correct_transcripts = ['to jest jest', 'test']
    labels = deepspeech.get_labels(bad_transcripts)
    assert labels.dtype == np.int64
    assert labels.shape == (2, 12)
    assert labels[1, 4] == deepspeech.alphabet.blank_token
    transformed_transcripts = deepspeech.get_transcripts(labels)
    assert transformed_transcripts == correct_transcripts


def test_fit(deepspeech: DeepSpeech, generator: DataGenerator, config_path: str, alphabet_path: str, test_dir: str):
    # Test save best weights (overwrite the best result)
    weights_path = os.path.join(test_dir, 'weights_copy.hdf5')
    deepspeech.save(weights_path)
    distributed_weights = deepspeech.compiled_model.get_weights()
    model_checkpoint = deepspeech.callbacks[1]
    model_checkpoint.best_result = 0
    model_checkpoint.best_weights_path = weights_path

    history = deepspeech.fit(train_generator=generator, dev_generator=generator, epochs=1, shuffle=False)
    assert type(history) == History

    # Test the returned model has `test_weights`
    deepspeech_weights = deepspeech.model.get_weights()
    new_deepspeech = DeepSpeech.construct(config_path, alphabet_path)
    new_deepspeech.load(model_checkpoint.best_weights_path)
    new_deepspeech_weights = new_deepspeech.model.get_weights()
    assert is_same(deepspeech_weights, new_deepspeech_weights)

    # Test that distributed model appropriate update weights
    new_distributed_weights = deepspeech.compiled_model.get_weights()
    assert is_same(distributed_weights, new_distributed_weights)
    shutil.rmtree('tests/checkpoints')
    os.remove('tests/weights_copy.hdf5')


def test_predict():
    # Load pretrained model and predict
    pass


def test_decode():
    # Load pretrained model and decode
    pass


def test_call(deepspeech: DeepSpeech, audio_file_paths: List[str]):
    # sentences = deepspeech(audio_file_paths)
    # assert len(sentences) == 2
    pass
