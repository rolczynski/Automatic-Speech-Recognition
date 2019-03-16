import os
from deepspeech import DeepSpeech
from source.utils import load, get_root_dir


def test_get_root_dir():
    root_dir = get_root_dir()
    assert os.path.isdir(root_dir)
    assert os.path.basename(root_dir) == 'DeepSpeech-Keras'


def test_utils_load():
    deepspeech = load('tests/models/test')      # Or call via: model name
    assert isinstance(deepspeech, DeepSpeech)   # (but has to be in the models directory)
