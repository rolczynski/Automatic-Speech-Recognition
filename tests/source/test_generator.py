from deepspeech import DataGenerator
from utils import chdir

chdir(to='ROOT')


def test_create_generator_from_audio_files(generator: DataGenerator):
    assert len(generator) == 2
    X, y = generator[0]
    assert X.shape == (2, 1477, 80)
    assert y.shape == (2, 206)
