from copy import deepcopy as copy

from generator import DistributedDataGenerator
from source.generator import DataGenerator
from source.utils import chdir

chdir(to='ROOT')


def test_create_generator_from_audio_files(generator: DataGenerator):
    assert len(generator) == 2
    X, y = generator[0]
    assert X.shape == (2, 299, 80)
    assert y.shape == (2, 45)


def test_distributed_generator(generator: DataGenerator):
    generator = DistributedDataGenerator(
        generators=[copy(generator) for i in range(5)]
    )
    assert len(generator) == 10
    assert generator[0]
    assert generator[9]
