import os
import numpy as np
import deepspeech
from deepspeech import DeepSpeech
from source.utils import load, get_root_dir, chdir, freeze
is_same = lambda A, B: all(np.array_equal(a, b) for a, b in zip(A, B))
chdir(to='ROOT')


def test_freeze():
    gpus = deepspeech.get_available_gpus()
    base_configuration = DeepSpeech.get_configuration('tests/models/base/configuration.yaml')
    base_model = DeepSpeech.get_model(is_gpu=len(gpus) > 0, **base_configuration.model)

    fname = 'weights.hdf5'
    base_model.save_weights(fname)
    freeze(base_model, weights_path=fname)
    assert all(not base_model.get_layer(name).trainable for name in ['base_1', 'base_2', 'base_3'])

    # Freeze only base layers in the extended model
    extended_configuration = DeepSpeech.get_configuration('tests/models/extended/configuration.yaml')
    extended_model = DeepSpeech.get_model(is_gpu=len(gpus) > 0, **extended_configuration.model)
    extended_model.load_weights(fname, by_name=True)
    extended_model_weights = base_model.get_weights()
    freeze(extended_model, weights_path=fname)

    assert all(not extended_model.get_layer(name).trainable for name in ['base_1', 'base_2', 'base_3'])
    assert all(extended_model.get_layer(name).trainable for name in ['extension_1', 'extension_2'])
    assert is_same(extended_model.get_weights(), extended_model_weights)    # Keep the same, pretrained weights
    os.remove(fname)


def test_get_root_dir():
    root_dir = get_root_dir()
    assert os.path.isdir(root_dir)
    assert os.path.basename(root_dir) == 'DeepSpeech-Keras'


def test_load():
    deepspeech = load('tests')                          # Or call via: model name
    assert isinstance(deepspeech, DeepSpeech)           # (but has to be in the models directory)
