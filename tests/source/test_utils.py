import os
from source.utils import get_root_dir, chdir
chdir(to='ROOT')


def test_get_root_dir():
    root_dir = get_root_dir()
    assert os.path.isdir(root_dir)
    assert os.path.basename(root_dir) == 'DeepSpeech-Keras'
